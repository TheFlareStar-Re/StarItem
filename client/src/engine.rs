//! 推理：分词、文本编码、DDIM 采样。和 train/export_onnx.py 里的 ort_sample 逐步对应。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use tokenizers::Tokenizer;

use crate::i18n::{english, tr};
use crate::models::{self, Manifest, Source};

pub struct Request {
    pub prompt: String,
    pub size: usize,
    pub count: usize,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    /// 是否用名称补全表把提示词补成训练时见过的写法。
    pub expand: bool,
    /// 画风：None 表示自动（按认出的物品决定），Some(x) 是手动指定，0 = 手绘，1 = 合成。
    pub style: Option<f32>,
    /// 照着画的参考图，没有就是纯按提示词画。
    pub reference: Option<Reference>,
    /// 用哪个出图网络。
    pub network: Choice,
}

/// 谁来画。SIC-1.2 起模型包里有两个出图网络，对外叫 Mo 和 Jang：Mo（unet.onnx）更听颜色词、对没见过的组合更稳，
/// Jang（unet_names.onnx）认名字更准。Relay 是接力：前 RELAY 比例的步数（定轮廓的阶段）由 Mo 画，其余交给 Jang，
/// 形状像 Mo 那样稳，颜色像 Jang 那样准。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Network {
    Mo,
    Jang,
    Relay,
}

/// 接力时 Mo 画的步数占比。Jang 的碎块出在最开始的几步，五分之一就够把它治住。
const RELAY: f64 = 0.2;
/// 提示词不是恰好认得的名字时，第 k 张图由谁画：接力是最好的单一选择，但带颜色词时 Mo 更听话、Jang 更保得住物品，各掺一张。
const MIXED: [Network; 4] = [Network::Relay, Network::Mo, Network::Relay, Network::Jang];

/// 用哪个网络：按提示词自动决定，或者手动指定一个。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Auto,
    Only(Network),
}

/// 第二个网络用到时才加载：很多提示词只需要其中一个。
enum Lazy {
    Missing,
    Pending { dir: Option<PathBuf>, name: String },
    Ready(Session),
}

/// 参考图缩成的底稿，以及照着它画时的两个力度。做法和 skill 里的 generate.py 一致。
pub struct Reference {
    pub draft: Sprite,
    /// 0..1，模型可以偏离底稿多远；1 等于不看底稿。
    pub strength: f32,
    /// 0..1，把结果的颜色往底稿拉回多少。
    pub colors: f32,
}

/// 一次生成的结果。
pub struct Output {
    pub sprites: Vec<Sprite>,
    /// 实际送进模型的提示词。
    pub prompt: String,
    /// 实际使用的画风。
    pub style: f32,
    /// 每张图是哪个网络画的。
    pub networks: Vec<Network>,
}

/// 一张生成结果，RGBA 逐像素排列。
#[derive(Clone)]
pub struct Sprite {
    pub size: usize,
    pub rgba: Vec<u8>,
}

pub struct Engine {
    unet: Session,
    unet_names: Lazy,
    /// 手绘训练图的名字（小写）。提示词和其中一个完全相同时交给认名字的网络。
    known: HashSet<String>,
    text: Session,
    tokenizer: Tokenizer,
    manifest: Manifest,
    alphas_cumprod: Vec<f64>,
    /// 物品的各种叫法 -> ("中文名, 英文名", 这个物品适合的画风)。
    aliases: HashMap<String, (String, f32)>,
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// text 是否以 key 这个词结尾。中英文的中心词都在最后（"感恩节火鸡"、"thanksgiving turkey"），
/// 只认结尾可以避免"钻石剑"因为含有"钻石"被当成钻石。英文还要求 key 前面不是字母数字。
fn ends_with_word(text: &str, key: &str) -> bool {
    let Some(head) = text.strip_suffix(key) else {
        return false;
    };
    key.chars().any(is_cjk) || head.chars().next_back().map_or(true, |c| !c.is_alphanumeric())
}

/// 从目录里的文件或内置的字节加载一个 ONNX 模型。
fn session(dir: Option<&Path>, name: &str, embedded: &[u8]) -> Result<Session> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads)?;
    Ok(match dir {
        Some(dir) => builder.commit_from_file(dir.join(name))?,
        None => builder.commit_from_memory(embedded)?,
    })
}

/// 余弦噪声表，和训练时的 cosine_alphas_cumprod 一致。
fn cosine_alphas_cumprod(steps: usize) -> Vec<f64> {
    let f = |i: usize| {
        let x = (i as f64 / steps as f64 + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2;
        x.cos().powi(2)
    };
    let mut acc = 1.0;
    (0..steps)
        .map(|i| {
            let beta = (1.0 - f(i + 1) / f(i)).min(0.999);
            acc *= 1.0 - beta;
            acc
        })
        .collect()
}

/// 用显著物体模型给一张图片分出前景。px 是 w*h 个 RGBA 像素，返回每个像素是不是物体。和 generate.py 的 matting_mask 对应。
fn matting_mask(px: &[[f32; 4]], w: usize, h: usize) -> Result<Vec<bool>> {
    const N: usize = 320;
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    // 模型只在用到时加载一次
    static SESSION: Mutex<Option<Session>> = Mutex::new(None);

    // 双线性缩放到 320x320（像素中心对齐，边缘截住），和 Pillow 的 BILINEAR 放大一致
    let axis = |len: usize| -> Vec<(usize, usize, f32)> {
        (0..N)
            .map(|o| {
                let f = ((o as f32 + 0.5) * len as f32 / N as f32 - 0.5).clamp(0.0, (len - 1) as f32);
                let a = f.floor() as usize;
                (a, (a + 1).min(len - 1), f - a as f32)
            })
            .collect()
    };
    let (xs, ys) = (axis(w), axis(h));
    let mut input = vec![0f32; 3 * N * N];
    for (oy, &(y0, y1, ty)) in ys.iter().enumerate() {
        for (ox, &(x0, x1, tx)) in xs.iter().enumerate() {
            for c in 0..3 {
                let top = px[y0 * w + x0][c] * (1.0 - tx) + px[y0 * w + x1][c] * tx;
                let bottom = px[y1 * w + x0][c] * (1.0 - tx) + px[y1 * w + x1][c] * tx;
                input[c * N * N + oy * N + ox] = (top * (1.0 - ty) + bottom * ty).round();
            }
        }
    }
    let peak = input.iter().copied().fold(1e-6f32, f32::max);
    for c in 0..3 {
        for v in &mut input[c * N * N..(c + 1) * N * N] {
            *v = (*v / peak - MEAN[c]) / STD[c];
        }
    }

    let mut guard = SESSION.lock().map_err(|_| anyhow!("matting session lock"))?;
    if guard.is_none() {
        *guard = Some(session(None, "", models::EMBEDDED_MATTING)?);
    }
    let model = guard.as_mut().unwrap();
    let (name, first) = (model.inputs[0].name.clone(), model.outputs[0].name.clone());
    let outputs = model.run(ort::inputs![name.as_str() => Tensor::from_array(([1usize, 3, N, N], input))?])?;
    let (_, pred) = outputs[first.as_str()].try_extract_tensor::<f32>()?;
    let (lo, hi) = pred.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (hi - lo).max(1e-6);
    Ok((0..w * h).map(|i| (pred[((i / w) * N / h).min(N - 1) * N + ((i % w) * N / w).min(N - 1)] - lo) / span >= 0.5).collect())
}

/// 去掉比最大的一块小得多的零碎块：模型顺带标出来的别的东西。
fn main_parts(mask: Vec<bool>, w: usize, h: usize) -> Vec<bool> {
    let mut label = vec![0usize; w * h];
    let mut sizes = vec![0usize];
    for start in 0..w * h {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let n = sizes.len();
        label[start] = n;
        let (mut stack, mut count) = (vec![start], 0);
        while let Some(i) = stack.pop() {
            count += 1;
            let (x, y) = (i % w, i / w);
            let around = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in around.into_iter().flatten() {
                if mask[j] && label[j] == 0 {
                    label[j] = n;
                    stack.push(j);
                }
            }
        }
        sizes.push(count);
    }
    let least = (0.05 * *sizes.iter().max().unwrap() as f32).max(1.0);
    label.iter().map(|&n| sizes[n] as f32 >= least).collect()
}

/// 对一批图做完整的去噪。x 是 n 张 (4,S,S) 的图排在一起，原地更新成结果。
/// 第 handoff 步之前用 Mo，从它起用 Jang；ctx 和 null_ctx 是两个网络各自的文本特征，[Mo, Jang]。
#[allow(clippy::too_many_arguments)]
fn denoise(
    (mo, mut jang, handoff): (&mut Session, Option<&mut Session>, usize),
    x: &mut [f32],
    (b, s): (usize, usize),
    (ctx, mask): (&[Vec<f32>; 2], &[bool]),
    (null_ctx, null_mask): (&[Vec<f32>; 2], &[bool]),
    (len, dim): (usize, usize),
    style: Option<f32>,
    guidance: f32,
    (ts, first): (&[usize], usize),
    alphas_cumprod: &[f64],
    tick: &mut dyn FnMut(),
) -> Result<()> {
    let pixels = 4 * s * s;
    let size_idx = if s == 32 { 1i64 } else { 0 };
    // 前 b 份是带提示词的，后 b 份是空提示词（classifier-free guidance）
    let doubled = |slot: usize| {
        let mut out = Vec::with_capacity(2 * b * len * dim);
        for c in [&ctx[slot], &null_ctx[slot]] {
            for _ in 0..b {
                out.extend_from_slice(c);
            }
        }
        out
    };
    let ctx2 = [doubled(0), doubled(1)];
    let mut mask2 = Vec::with_capacity(2 * b * len);
    for m in [mask, null_mask] {
        for _ in 0..b {
            mask2.extend_from_slice(m);
        }
    }
    for (i, &t) in ts.iter().enumerate().skip(first) {
        let (unet, slot) = match &mut jang {
            Some(jang) if i >= handoff => (&mut **jang, 1),
            _ => (&mut *mo, 0),
        };
        let mut xx = Vec::with_capacity(2 * b * pixels);
        xx.extend_from_slice(x);
        xx.extend_from_slice(x);
        let x_in = Tensor::from_array(([2 * b, 4, s, s], xx))?;
        let t_in = Tensor::from_array(([2 * b], vec![t as i64; 2 * b]))?;
        let ctx_in = Tensor::from_array(([2 * b, len, dim], ctx2[slot].clone()))?;
        let mask_in = Tensor::from_array(([2 * b, len], mask2.clone()))?;
        let size_in = Tensor::from_array(([2 * b], vec![size_idx; 2 * b]))?;
        let outputs = match style {
            Some(style) => {
                let src_in = Tensor::from_array(([2 * b], vec![style; 2 * b]))?;
                unet.run(ort::inputs![
                    "x" => x_in, "t" => t_in, "ctx" => ctx_in, "mask" => mask_in, "size_idx" => size_in, "src_mix" => src_in,
                ])?
            }
            None => unet.run(ort::inputs!["x" => x_in, "t" => t_in, "ctx" => ctx_in, "mask" => mask_in, "size_idx" => size_in])?,
        };
        let (_, v) = outputs["v"].try_extract_tensor::<f32>()?;
        let (v_cond, v_null) = v.split_at(b * pixels);

        let ac = alphas_cumprod[t];
        let (a, sg) = (ac.sqrt() as f32, (1.0 - ac).sqrt() as f32);
        let next = ts.get(i + 1).map(|&n| alphas_cumprod[n]);
        for k in 0..b * pixels {
            let v = v_null[k] + guidance * (v_cond[k] - v_null[k]);
            let x0 = (a * x[k] - sg * v).clamp(-1.0, 1.0);
            x[k] = match next {
                Some(ap) => {
                    let eps = (x[k] - a * x0) / sg;
                    ap.sqrt() as f32 * x0 + (1.0 - ap).sqrt() as f32 * eps
                }
                None => x0,
            };
        }
        tick();
    }
    Ok(())
}

/// 把一张物体的图片变成 16x16 的底稿：抠掉背景，裁出物体，按面积平均缩小，边缘压暗。
/// 透明或纯色的背景直接去掉；别的背景（实拍图）用抠图模型找出主体，这时返回的第二项是 true。
pub fn reference_draft(path: &Path) -> Result<(Sprite, bool)> {
    const SIDE: usize = 16;
    const TOLERANCE: f32 = 40.0;
    let img = image::open(path).map_err(|e| anyhow!("{}{e}", tr("读不了这张图片：", "Cannot read this image: ")))?.to_rgba8();
    let (mut w, mut h) = (img.width() as usize, img.height() as usize);
    let mut px: Vec<[f32; 4]> = img.pixels().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]).collect();

    // 比 256 更细的细节在 16x16 里留不住，先缩小，后面找背景也快
    let k = w.max(h).div_ceil(256);
    if k > 1 {
        let solid = px.iter().all(|p| p[3] == 255.0);
        let (nw, nh) = ((w / k).max(1), (h / k).max(1));
        let mut small = Vec::with_capacity(nw * nh);
        for y in 0..nh {
            for x in 0..nw {
                if solid {
                    let mut sum = [0f32; 3];
                    for dy in 0..k {
                        for dx in 0..k {
                            let p = px[(y * k + dy).min(h - 1) * w + (x * k + dx).min(w - 1)];
                            for c in 0..3 {
                                sum[c] += p[c];
                            }
                        }
                    }
                    let n = (k * k) as f32;
                    small.push([sum[0] / n, sum[1] / n, sum[2] / n, 255.0]);
                } else {
                    // 带透明的图不做平均，免得透明处的颜色渗进来
                    small.push(px[(y * k).min(h - 1) * w + (x * k).min(w - 1)]);
                }
            }
        }
        (px, w, h) = (small, nw, nh);
    }

    let total = (w * h) as f32;
    let mut opaque: Vec<bool> = px.iter().map(|p| p[3] >= 128.0).collect();
    let mut by_model = false;
    if opaque.iter().filter(|&&o| !o).count() as f32 / total < 0.02 {
        // 没有透明区域：把四边的颜色当背景，从边缘往里把相连的同色区域去掉
        let border: Vec<usize> = (0..w).flat_map(|x| [x, (h - 1) * w + x]).chain((0..h).flat_map(|y| [y * w, y * w + w - 1])).collect();
        let mut bg = [0f32; 3];
        for (c, value) in bg.iter_mut().enumerate() {
            let mut channel: Vec<f32> = border.iter().map(|&i| px[i][c]).collect();
            channel.sort_by(f32::total_cmp);
            *value = channel[channel.len() / 2];
        }
        let close = |i: usize| (0..3).map(|c| (px[i][c] - bg[c]).powi(2)).sum::<f32>().sqrt() <= TOLERANCE;
        if border.iter().filter(|&&i| close(i)).count() as f32 / (border.len() as f32) < 0.85 {
            // 背景不是纯色（实拍图）：让模型找出主体
            opaque = main_parts(matting_mask(&px, w, h)?, w, h);
            by_model = true;
        } else {
            let mut reached = vec![false; w * h];
            let mut stack: Vec<usize> = border.into_iter().filter(|&i| close(i)).collect();
            while let Some(i) = stack.pop() {
                if reached[i] {
                    continue;
                }
                reached[i] = true;
                let (x, y) = (i % w, i / w);
                let around = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
                stack.extend(around.into_iter().flatten().filter(|&j| !reached[j] && close(j)));
            }
            opaque = reached.iter().map(|&r| !r).collect();
        }
    }
    let kept = opaque.iter().filter(|&&o| o).count() as f32 / total;
    if !(0.01..=0.97).contains(&kept) {
        return Err(anyhow!(
            "{}",
            tr(
                "没能从这张图里分出一个物体。换一张物体更突出、背景更简单的图试试。",
                "Could not pick out an object in this image. Try one where the object stands out against a simpler background.",
            )
        ));
    }

    // 裁到物体的外接框，补成正方形，再按面积平均缩到 16x16（颜色按不透明部分的占比加权，背景不会渗进来）
    let cells = || (0..w * h).filter(|&i| opaque[i]);
    let (x0, x1) = (cells().map(|i| i % w).min().unwrap(), cells().map(|i| i % w).max().unwrap());
    let (y0, y1) = (cells().map(|i| i / w).min().unwrap(), cells().map(|i| i / w).max().unwrap());
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let side = bw.max(bh);
    let (ox, oy) = ((side - bw) / 2, (side - bh) / 2);
    let at = |sx: usize, sy: usize| -> Option<[f32; 4]> {
        if sx < ox || sy < oy || sx >= ox + bw || sy >= oy + bh {
            return None;
        }
        let i = (y0 + sy - oy) * w + x0 + sx - ox;
        opaque[i].then(|| px[i])
    };
    let scale = side as f64 / SIDE as f64;
    let mut color = vec![[0f32; 3]; SIDE * SIDE];
    let mut solid = vec![false; SIDE * SIDE];
    for cy in 0..SIDE {
        for cx in 0..SIDE {
            let (fx0, fx1, fy0, fy1) = (cx as f64 * scale, (cx + 1) as f64 * scale, cy as f64 * scale, (cy + 1) as f64 * scale);
            let (mut sum, mut cover) = ([0f64; 3], 0f64);
            for sy in fy0.floor() as usize..(fy1.ceil() as usize).min(side) {
                let wy = (fy1.min(sy as f64 + 1.0) - fy0.max(sy as f64)).max(0.0);
                for sx in fx0.floor() as usize..(fx1.ceil() as usize).min(side) {
                    let weight = wy * (fx1.min(sx as f64 + 1.0) - fx0.max(sx as f64)).max(0.0);
                    if let Some(p) = at(sx, sy) {
                        cover += weight;
                        for c in 0..3 {
                            sum[c] += weight * p[c] as f64;
                        }
                    }
                }
            }
            if cover / (scale * scale) >= 0.5 {
                solid[cy * SIDE + cx] = true;
                color[cy * SIDE + cx] = [(sum[0] / cover) as f32, (sum[1] / cover) as f32, (sum[2] / cover) as f32];
            }
        }
    }

    // 物品贴图的边缘一般更深，把底稿的边缘格压暗，模型更容易往那个画法靠
    let filled = |x: isize, y: isize| x >= 0 && y >= 0 && x < SIDE as isize && y < SIDE as isize && solid[y as usize * SIDE + x as usize];
    let mut rgba = vec![0u8; 4 * SIDE * SIDE];
    for cy in 0..SIDE as isize {
        for cx in 0..SIDE as isize {
            if !filled(cx, cy) {
                continue;
            }
            let inside = filled(cx - 1, cy) && filled(cx + 1, cy) && filled(cx, cy - 1) && filled(cx, cy + 1);
            let i = cy as usize * SIDE + cx as usize;
            for c in 0..3 {
                let value = if inside { color[i][c] } else { color[i][c] * 0.55 };
                rgba[i * 4 + c] = value.round().clamp(0.0, 255.0) as u8;
            }
            rgba[i * 4 + 3] = 255;
        }
    }
    Ok((Sprite { size: SIDE, rgba }, by_model))
}

/// 把生成结果的色相往底稿拉回去，明暗保留生成结果自己的。不这么做的话颜色容易跑偏（粉色茶壶变成米色）。
fn keep_colors(sprite: &mut Sprite, draft: &Sprite, amount: f32) {
    if amount <= 0.0 || sprite.size != draft.size {
        return;
    }
    let ycc = |p: &[u8]| {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        [0.299 * r + 0.587 * g + 0.114 * b, -0.168736 * r - 0.331264 * g + 0.5 * b, 0.5 * r - 0.418688 * g - 0.081312 * b]
    };
    for (out, base) in sprite.rgba.chunks_exact_mut(4).zip(draft.rgba.chunks_exact(4)) {
        if out[3] == 0 || base[3] == 0 {
            continue;
        }
        let (a, b) = (ycc(out), ycc(base));
        // 底稿的边缘被压暗过，色度也跟着变小了，借用之前按亮度比例还原
        let gain = (a[0] / b[0].max(1.0)).clamp(0.0, 2.0);
        let cb = (1.0 - amount) * a[1] + amount * b[1] * gain;
        let cr = (1.0 - amount) * a[2] + amount * b[2] * gain;
        let rgb = [a[0] + 1.402 * cr, a[0] - 0.344136 * cb - 0.714136 * cr, a[0] + 1.772 * cb];
        for c in 0..3 {
            out[c] = rgb[c].round().clamp(0.0, 255.0) as u8;
        }
    }
}

impl Engine {
    pub fn load(source: &Source, manifest: Manifest) -> Result<Self> {
        let dir = match source {
            Source::Dir(dir) => Some(dir.as_path()),
            Source::Embedded => None,
        };
        let tokenizer = match dir {
            Some(dir) => Tokenizer::from_file(dir.join(&manifest.files.tokenizer.name)),
            None => Tokenizer::from_bytes(models::EMBEDDED_TOKENIZER),
        }
        .map_err(|e| anyhow!("{}{e}", tr("分词器加载失败：", "Failed to load the tokenizer: ")))?;
        let aliases = match (dir, &manifest.files.aliases) {
            (Some(dir), Some(entry)) => std::fs::read_to_string(dir.join(&entry.name)).unwrap_or_default(),
            (Some(_), None) => String::new(),
            (None, _) => models::EMBEDDED_ALIASES.to_string(),
        };
        let aliases = serde_json::from_str(&aliases).unwrap_or_default();
        let unet_names = match (&manifest.files.unet_names, &manifest.text.names_output) {
            (Some(entry), Some(_)) => Lazy::Pending { dir: dir.map(Path::to_path_buf), name: entry.name.clone() },
            _ => Lazy::Missing,
        };
        let known = match (dir, &manifest.files.known_names) {
            (Some(dir), Some(entry)) => std::fs::read_to_string(dir.join(&entry.name)).unwrap_or_default(),
            (None, Some(_)) => models::EMBEDDED_KNOWN_NAMES.to_string(),
            _ => String::new(),
        };
        let known = serde_json::from_str::<Vec<String>>(&known).unwrap_or_default().into_iter().collect();
        Ok(Self {
            aliases,
            unet_names,
            known,
            unet: session(dir, &manifest.files.unet.name, models::embedded_unet())?,
            text: session(dir, &manifest.files.text_encoder.name, models::embedded_text_encoder())?,
            tokenizer,
            alphas_cumprod: cosine_alphas_cumprod(manifest.diffusion.timesteps),
            manifest,
        })
    }

    /// 模型包里是不是有两个出图网络。
    pub fn two_networks(&self) -> bool {
        !matches!(self.unet_names, Lazy::Missing)
    }

    /// 返回 ([两个网络各自的 max_len * dim 的 token 特征], max_len 的 mask)。只有一个网络时两份相同。
    fn encode(&mut self, prompt: &str) -> Result<([Vec<f32>; 2], Vec<bool>)> {
        let text = if self.manifest.text.lowercase { prompt.trim().to_lowercase() } else { prompt.trim().to_string() };
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("{}{e}", tr("分词失败：", "Tokenization failed: ")))?;
        let len = self.manifest.text.max_len;
        if enc.get_ids().len() != len {
            let got = enc.get_ids().len();
            return Err(if english() {
                anyhow!("tokenizer produced {got} tokens but the model expects {len}")
            } else {
                anyhow!("分词结果长度 {got} 和模型要求的 {len} 不一致")
            });
        }
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let att: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        let mask = att.iter().map(|&m| m != 0).collect();
        let outputs = self.text.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, len], ids))?,
            "attention_mask" => Tensor::from_array(([1usize, len], att))?,
        ])?;
        let mo = outputs["last_hidden_state"].try_extract_tensor::<f32>()?.1.to_vec();
        let jang = match (&self.manifest.text.names_output, &self.unet_names) {
            (Some(name), Lazy::Pending { .. } | Lazy::Ready(_)) => outputs[name.as_str()].try_extract_tensor::<f32>()?.1.to_vec(),
            _ => mo.clone(),
        };
        Ok(([mo, jang], mask))
    }

    /// 看提示词是不是以某个已知物品的叫法结尾（取最长的一个）。返回 (补全后的提示词, 这个物品适合的画风)。
    /// 补全就是补上物品的中英文正式名称，训练时有一部分提示词就是这种"叫法, 中文名, 英文名"的写法。
    /// 第三项：提示词是不是恰好就是这个名字（前面没有别的词）。
    fn lookup(&self, text: &str) -> Option<(String, f32, bool)> {
        let (key, (pair, style)) = self
            .aliases
            .iter()
            .filter(|(key, _)| ends_with_word(text, key))
            .max_by_key(|(key, _)| key.chars().count())?;
        let expanded = if key == text && pair.split(", ").any(|part| part == text) {
            pair.clone()
        } else if text.contains(pair.as_str()) {
            text.to_string()
        } else {
            format!("{text}, {pair}")
        };
        Some((expanded, *style, key == text || text.contains(pair.as_str())))
    }

    pub fn generate(&mut self, req: &Request, mut progress: impl FnMut(usize, usize)) -> Result<Output> {
        let text = req.prompt.trim().to_lowercase();
        let known = self.lookup(&text);
        let style = req.style.or(known.as_ref().map(|k| k.1)).unwrap_or(self.manifest.defaults.source);
        let two = self.two_networks();
        let named = known.as_ref().is_some_and(|k| k.2) || self.known.contains(&text);
        let networks: Vec<Network> = (0..req.count)
            .map(|k| match req.network {
                _ if !two => Network::Mo,
                Choice::Only(network) => network,
                // 恰好是训练时见过的名字（日常物品或模组物品）：交给认名字的 Jang
                Choice::Auto if named => Network::Jang,
                // 名字前面带了别的词，或者是没见过的组合
                Choice::Auto => MIXED[k % MIXED.len()],
            })
            .collect();
        let prompt = match known {
            Some((expanded, _, _)) if req.expand => expanded,
            _ => text,
        };
        let (ctx, mask) = self.encode(&prompt)?;
        let (null_ctx, null_mask) = self.encode("")?;
        let (b, s, len, dim) = (req.count, req.size, self.manifest.text.max_len, self.manifest.text.dim);
        let pixels = 4 * s * s;

        let mut rng = StdRng::seed_from_u64(req.seed);
        let mut x: Vec<f32> = (0..b * pixels).map(|_| rng.sample(StandardNormal)).collect();
        let total = self.manifest.diffusion.timesteps;
        let steps = req.steps.max(2);
        let ts: Vec<usize> = (0..steps)
            .map(|i| ((total - 1) as f64 * (1.0 - i as f64 / (steps - 1) as f64)).round_ties_even() as usize)
            .collect();

        // 有参考图时不从纯噪声开始：给底稿加上中途那一步的噪声，从那一步接着去噪
        let mut first = 0;
        let reference = req.reference.as_ref().filter(|r| r.strength < 1.0 && r.draft.size == s);
        if let Some(r) = reference {
            first = (((1.0 - r.strength as f64) * steps as f64).round_ties_even() as usize).min(steps - 1);
            let ac = self.alphas_cumprod[ts[first]];
            let (keep, noise) = (ac.sqrt() as f32, (1.0 - ac).sqrt() as f32);
            let plane = s * s;
            for n in 0..b {
                for p in 0..plane {
                    let cell = &r.draft.rgba[p * 4..p * 4 + 4];
                    for c in 0..4 {
                        // 透明格的颜色按训练时的约定取 -1
                        let value = if cell[3] == 0 { -1.0 } else { cell[c] as f32 / 127.5 - 1.0 };
                        let k = n * pixels + c * plane + p;
                        x[k] = keep * value + noise * x[k];
                    }
                }
            }
        }

        // 每张图的噪声是自己的，所以第 k 张图不随张数和两个网络的分配而变
        let all = [Network::Mo, Network::Jang, Network::Relay];
        let (mut done, work) = (0, (steps - first) * all.iter().filter(|n| networks.contains(n)).count());
        for network in all {
            let rows: Vec<usize> = (0..b).filter(|&k| networks[k] == network).collect();
            if rows.is_empty() {
                continue;
            }
            if network != Network::Mo {
                if let Lazy::Pending { dir, name } = &self.unet_names {
                    self.unet_names = Lazy::Ready(session(dir.as_deref(), name, models::embedded_unet_names())?);
                }
            }
            let jang = match &mut self.unet_names {
                Lazy::Ready(names) => Some(names),
                _ => None,
            };
            // 从第几步起交给 Jang：只用 Mo 时永不交接，只用 Jang 时从头就是它
            let handoff = match network {
                Network::Mo => steps,
                Network::Jang => 0,
                Network::Relay => (RELAY * steps as f64).round() as usize,
            };
            let mut part: Vec<f32> = rows.iter().flat_map(|&k| x[k * pixels..(k + 1) * pixels].iter().copied()).collect();
            denoise(
                (&mut self.unet, jang, handoff),
                &mut part,
                (rows.len(), s),
                (&ctx, &mask),
                (&null_ctx, &null_mask),
                (len, dim),
                self.manifest.has_source.then_some(style),
                req.guidance,
                (&ts, first),
                &self.alphas_cumprod,
                &mut || {
                    done += 1;
                    progress(done, work);
                },
            )?;
            for (n, &k) in rows.iter().enumerate() {
                x[k * pixels..(k + 1) * pixels].copy_from_slice(&part[n * pixels..(n + 1) * pixels]);
            }
        }

        // (B,4,S,S) 的 [-1,1] -> RGBA 字节；alpha 二值化，透明像素清零
        let plane = s * s;
        let sprites = (0..b)
            .map(|n| {
                let img = &x[n * pixels..(n + 1) * pixels];
                let mut rgba = vec![0u8; pixels];
                for p in 0..plane {
                    let byte = |c: usize| ((img[c * plane + p].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
                    if byte(3) >= 128 {
                        rgba[p * 4..p * 4 + 4].copy_from_slice(&[byte(0), byte(1), byte(2), 255]);
                    }
                }
                let mut sprite = Sprite { size: s, rgba };
                if let Some(r) = reference {
                    keep_colors(&mut sprite, &r.draft, r.colors);
                }
                sprite
            })
            .collect();
        Ok(Output { sprites, prompt, style, networks })
    }
}
