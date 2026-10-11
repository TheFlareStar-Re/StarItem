"""余弦噪声表、v-prediction 目标和 DDIM 采样。"""
import math

import torch

T = 1000


def cosine_alphas_cumprod(steps=T, s=0.008):
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    f = torch.cos((x / steps + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def q_sample(x0, t, noise, ac):
    a = ac[t].sqrt()[:, None, None, None]
    s = (1 - ac[t]).sqrt()[:, None, None, None]
    return a * x0 + s * noise, a * noise - s * x0  # 加噪后的图, v 目标


def min_snr_weight(t, ac, gamma=5.0):
    snr = ac[t] / (1 - ac[t])
    return torch.clamp(snr, max=gamma) / (snr + 1)  # v-prediction 下的 min-SNR 权重


@torch.no_grad()
def ddim_sample(model, ctx, mask, null_ctx, null_mask, size, size_idx, ac, steps=50, guidance=3.0, generator=None,
                source=0, tok=None, null_tok=None):
    """ctx/mask: (B,L,D)/(B,L)；null_*: 空提示词的 (1,L,D)/(1,L)；tok/null_tok: 词编号，模型带 tok_emb 时才给。
    返回 [-1,1] 的 (B,4,size,size)。"""
    b = ctx.shape[0]
    x = torch.randn(b, 4, size, size, device=ctx.device, generator=generator)
    sidx = torch.full((2 * b,), size_idx, device=ctx.device, dtype=torch.long)
    # source 是整数时按下标取，是小数时表示两种来源的混合比例
    src = torch.full((2 * b,), source, device=ctx.device, dtype=torch.long if isinstance(source, int) else torch.float32)
    ts = torch.linspace(T - 1, 0, steps, device=ctx.device).round().long()
    ctx2 = torch.cat([ctx, null_ctx.expand(b, -1, -1)])
    mask2 = torch.cat([mask, null_mask.expand(b, -1)])
    extra = {} if tok is None else {"tok": torch.cat([tok, null_tok.expand(b, -1)])}
    for i, t in enumerate(ts):
        tt = t.expand(b)
        v_c, v_u = model(torch.cat([x, x]), torch.cat([tt, tt]), ctx2, mask2, sidx, src, **extra).float().chunk(2)
        v = v_u + guidance * (v_c - v_u)
        a, s = ac[t].sqrt(), (1 - ac[t]).sqrt()
        x0 = (a * x - s * v).clamp(-1, 1)
        eps = (x - a * x0) / s
        if i == steps - 1:
            return x0
        ap = ac[ts[i + 1]]
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
