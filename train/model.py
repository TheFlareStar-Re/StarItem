"""文本条件的小型 UNet，直接在 RGBA 像素上做扩散；全卷积结构，16x16 和 32x32 共用一套权重。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = 32


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(GROUPS, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, dim, ctx_dim=None, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(ctx_dim or dim, dim, bias=False)
        self.v = nn.Linear(ctx_dim or dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        b, n, c = x.shape
        split = lambda t: t.view(b, -1, self.heads, c // self.heads).transpose(1, 2)  # noqa: E731
        if mask is not None:
            mask = mask[:, None, None, :]  # True 表示可以关注
        o = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(ctx)), split(self.v(ctx)), attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(b, n, c))


class SpatialTransformer(nn.Module):
    """特征图上的自注意力 + 对文本 token 的交叉注意力。"""

    def __init__(self, dim, ctx_dim, heads):
        super().__init__()
        self.norm = nn.GroupNorm(GROUPS, dim)
        self.proj_in = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, ctx_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, ctx, mask):
        b, c, hh, ww = x.shape
        h = self.proj_in(self.norm(x).flatten(2).transpose(1, 2))
        h = h + self.attn1(self.norm1(h))
        h = h + self.attn2(self.norm2(h), ctx, mask)
        h = h + self.ff(self.norm3(h))
        return x + self.proj_out(h).transpose(1, 2).reshape(b, c, hh, ww)


class Block(nn.Module):
    def __init__(self, res, attn=None, upsample=False):
        super().__init__()
        self.res = res
        self.attn = attn
        c = res.conv2.out_channels
        self.up = nn.Conv2d(c, c, 3, padding=1) if upsample else None

    def forward(self, h, temb, ctx, mask):
        h = self.res(h, temb)
        if self.attn is not None:
            h = self.attn(h, ctx, mask)
        if self.up is not None:
            h = self.up(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return h


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, h, temb, ctx, mask):
        return self.conv(h)


class UNet(nn.Module):
    def __init__(self, ctx_dim, in_ch=4, ch=64, mult=(1, 2, 3), num_res=2, attn_levels=(1, 2), heads=4,
                 dropout=0.1, num_sizes=2, num_sources=0, num_tokens=0):
        super().__init__()
        assert ch % GROUPS == 0, "ch must be a multiple of 32"
        self.ch = ch
        tdim = ch * 4
        self.time = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # 分辨率作为条件：0 = 16x16，1 = 32x32
        self.size_emb = nn.Embedding(num_sizes, tdim)
        # 图的来源作为条件：0 = 手绘的模组贴图，1 = 大模型生成后缩小的合成图。旧模型没有这一项
        self.src_emb = nn.Embedding(num_sources, tdim) if num_sources else None
        # 每个词一组可训练的向量，加在冻结的文本特征上：编码器把近义的物品编得很近，这一项让模型能按字面把它们分开。
        # 0 号是没见过的词和补齐位，恒为零。旧模型没有这一项
        self.tok_emb = nn.Embedding(num_tokens, ctx_dim, padding_idx=0) if num_tokens else None
        if self.tok_emb is not None:
            nn.init.zeros_(self.tok_emb.weight)
        self.pool = nn.Sequential(nn.Linear(ctx_dim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)

        def stage(cin, cout, lvl, upsample=False):
            attn = SpatialTransformer(cout, ctx_dim, heads) if lvl in attn_levels else None
            return Block(ResBlock(cin, cout, tdim, dropout), attn, upsample)

        self.down = nn.ModuleList()
        chans = [ch]
        c = ch
        for lvl, m in enumerate(mult):
            for _ in range(num_res):
                self.down.append(stage(c, ch * m, lvl))
                c = ch * m
                chans.append(c)
            if lvl != len(mult) - 1:
                self.down.append(Downsample(c))
                chans.append(c)

        last = len(mult) - 1
        self.mid = nn.ModuleList([stage(c, c, last), Block(ResBlock(c, c, tdim, dropout))])

        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(num_res + 1):
                self.up.append(stage(c + chans.pop(), ch * m, lvl, upsample=lvl != 0 and i == num_res))
                c = ch * m

        self.out_norm = nn.GroupNorm(GROUPS, c)
        self.out = nn.Conv2d(c, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, ctx, mask, size_idx, src_idx=None, tok=None):
        """x: (B,4,S,S)；t: (B,) 时间步；ctx: (B,L,D) 文本 token 特征；mask: (B,L) bool；size_idx、src_idx: (B,)；
        tok: (B,L) 每个位置的词编号（模型带 tok_emb 时才用）。"""
        if self.tok_emb is not None and tok is not None:
            ctx = ctx + self.tok_emb(tok).to(ctx.dtype)
        temb = self.time(timestep_embedding(t, self.ch).to(x.dtype)) + self.size_emb(size_idx)
        if self.src_emb is not None:
            if src_idx.is_floating_point():
                # 浮点数表示两种来源的混合比例：0 = 手绘，1 = 合成
                w = src_idx[:, None].to(temb.dtype)
                temb = temb + (1 - w) * self.src_emb.weight[0] + w * self.src_emb.weight[1]
            else:
                temb = temb + self.src_emb(src_idx)
        m = mask[..., None].to(ctx.dtype)
        temb = temb + self.pool((ctx * m).sum(1) / m.sum(1).clamp(min=1.0))

        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = blk(h, temb, ctx, mask)
            hs.append(h)
        for blk in self.mid:
            h = blk(h, temb, ctx, mask)
        for blk in self.up:
            h = blk(torch.cat([h, hs.pop()], dim=1), temb, ctx, mask)
        return self.out(F.silu(self.out_norm(h)))
