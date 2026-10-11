//! 模型文件的定位、下载和校验。

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::i18n::{english, tr};

/// 随客户端一起打包的模型（client/models/ 下的文件，由 train/export_onnx.py 导出）。
pub static EMBEDDED_MANIFEST: &str = include_str!("../models/manifest.json");
pub static EMBEDDED_KNOWN_NAMES: &str = include_str!("../models/known_names.json");

/// 几百 MB 的模型文件不用 include_bytes!：那么大的常量会让编译器把内存用光。
/// 这里让汇编器在生成目标文件时把文件原样放进只读段，再按首尾两个符号取出来。路径相对于 client 目录。
macro_rules! blob {
    ($func:ident, $start:ident, $end:ident, $file:literal) => {
        std::arch::global_asm!(
            ".section .rdata,\"dr\"",
            ".p2align 4",
            concat!(".globl ", stringify!($start)),
            concat!(stringify!($start), ":"),
            concat!(".incbin \"", $file, "\""),
            concat!(".globl ", stringify!($end)),
            concat!(stringify!($end), ":"),
            ".text",
        );
        pub fn $func() -> &'static [u8] {
            extern "C" {
                static $start: u8;
                static $end: u8;
            }
            // 两个符号由上面的汇编定义，中间就是文件内容
            unsafe {
                let (first, last) = (std::ptr::addr_of!($start), std::ptr::addr_of!($end));
                std::slice::from_raw_parts(first, last as usize - first as usize)
            }
        }
    };
}

blob!(embedded_unet, STARITEM_UNET_START, STARITEM_UNET_END, "models/unet.onnx");
// 第二个出图网络（认名字更准的那个），SIC-1.2 起才有
blob!(embedded_unet_names, STARITEM_UNET_NAMES_START, STARITEM_UNET_NAMES_END, "models/unet_names.onnx");
blob!(embedded_text_encoder, STARITEM_TEXT_START, STARITEM_TEXT_END, "models/text_encoder.onnx");
pub static EMBEDDED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
pub static EMBEDDED_ALIASES: &str = include_str!("../models/aliases.json");
/// 抠图用的显著物体模型 U²-Net-p（Apache-2.0），只在参考图的背景不是纯色时用。和出图模型的版本无关，始终用内置的。
pub static EMBEDDED_MATTING: &[u8] = include_bytes!("../models/matting.onnx");
/// 抠图模型的出处说明和 Apache-2.0 全文，随 exe 一起分发，在"许可"窗口里显示。
pub static MATTING_LICENSE: &str = include_str!("../models/matting.LICENSE.txt");

/// 模型从哪里加载。
pub enum Source {
    Embedded,
    Dir(PathBuf),
}

/// 模型发布地址（GitHub Release 的 latest/download 目录）。也可以在 exe 旁边放 staritem.json 覆盖：
/// {"model_urls": ["https://.../releases/latest/download", "https://mirror/..."]}
/// 没有配置地址时不联网，直接用内置模型。
const DEFAULT_MODEL_URLS: &[&str] = &[];

#[derive(Deserialize, Clone)]
pub struct FileEntry {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Deserialize, Clone)]
pub struct Files {
    pub unet: FileEntry,
    pub text_encoder: FileEntry,
    pub tokenizer: FileEntry,
    /// 名称补全表，旧模型没有。
    #[serde(default)]
    pub aliases: Option<FileEntry>,
    /// 第二个出图网络和模组物品名表，SIC-1.2 之前的模型没有。
    #[serde(default)]
    pub unet_names: Option<FileEntry>,
    #[serde(default)]
    pub known_names: Option<FileEntry>,
}

#[derive(Deserialize, Clone)]
pub struct TextConfig {
    pub max_len: usize,
    pub dim: usize,
    pub lowercase: bool,
    /// 文本编码器给第二个网络用的那个输出的名字。
    #[serde(default)]
    pub names_output: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct Diffusion {
    pub timesteps: usize,
}

#[derive(Deserialize, Clone)]
pub struct Defaults {
    pub steps: usize,
    pub guidance: f32,
    /// 画风：0 = 手绘的模组贴图，1 = 合成图，中间值是两者的混合。
    #[serde(default)]
    pub source: f32,
}

#[derive(Deserialize, Clone)]
pub struct Manifest {
    pub format: u32,
    pub model_version: String,
    pub files: Files,
    pub text: TextConfig,
    pub diffusion: Diffusion,
    pub trained_sizes: Vec<u32>,
    /// 模型是否有"画风"这个输入（src_mix）。
    #[serde(default)]
    pub has_source: bool,
    pub defaults: Defaults,
}

#[derive(Deserialize, Default)]
struct LocalConfig {
    #[serde(default)]
    model_urls: Vec<String>,
}

impl Manifest {
    fn entries(&self) -> Vec<&FileEntry> {
        let mut all = vec![&self.files.unet, &self.files.text_encoder, &self.files.tokenizer];
        all.extend(self.files.aliases.as_ref());
        all.extend(self.files.unet_names.as_ref());
        all.extend(self.files.known_names.as_ref());
        all
    }
}

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."))
}

fn read_manifest(dir: &Path) -> Result<Manifest> {
    parse_manifest(&fs::read_to_string(dir.join("manifest.json"))?)
}

fn parse_manifest(text: &str) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_str(text).context(tr("manifest.json 格式不对", "manifest.json is malformed"))?;
    if manifest.format != 1 {
        if english() {
            bail!("model format version {} is newer than this client; please update the client", manifest.format);
        }
        bail!("模型格式版本 {} 比这个客户端新，请更新客户端", manifest.format);
    }
    Ok(manifest)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// 目录里的文件是否齐全。只比大小，哈希在下载时已经校验过。
fn complete(dir: &Path, manifest: &Manifest) -> bool {
    manifest.entries().iter().all(|e| fs::metadata(dir.join(&e.name)).map(|m| m.len() == e.size).unwrap_or(false))
}

fn model_urls() -> Vec<String> {
    let local: LocalConfig = fs::read_to_string(exe_dir().join("staritem.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    if !local.model_urls.is_empty() {
        return local.model_urls;
    }
    DEFAULT_MODEL_URLS.iter().map(|s| s.to_string()).collect()
}

fn download(url: &str, dest: &Path, expect: &FileEntry, status: &mut dyn FnMut(String)) -> Result<()> {
    let resp = ureq::get(url).call().map_err(|e| anyhow!("{url}: {e}"))?;
    let part = dest.with_extension("part");
    let mut out = fs::File::create(&part)?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    let mut last = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        if done - last >= (1 << 20) {
            last = done;
            let (name, done, size) = (&expect.name, done as f64 / 1e6, expect.size as f64 / 1e6);
            status(if english() { format!("Downloading {name}: {done:.0} / {size:.0} MB") } else { format!("正在下载 {name}：{done:.0} / {size:.0} MB") });
        }
    }
    drop(out);
    if sha256_file(&part)? != expect.sha256 {
        fs::remove_file(&part).ok();
        bail!("{} {}", expect.name, tr("校验失败", "failed its checksum"));
    }
    fs::rename(&part, dest)?;
    Ok(())
}

fn fetch_manifest(base: &str) -> Result<(String, Manifest)> {
    let text = ureq::get(&format!("{base}/manifest.json"))
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| anyhow!("{e}"))?
        .into_string()?;
    let manifest: Manifest = serde_json::from_str(&text)?;
    Ok((text, manifest))
}

/// 决定用哪份模型。顺序：命令行 --models、exe 旁的 models 目录、配置了发布地址时的本地缓存（会检查更新）、内置模型。
pub fn locate(status: &mut dyn FnMut(String)) -> Result<(Source, Manifest)> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--models") {
        let dir = PathBuf::from(args.get(i + 1).ok_or_else(|| anyhow!(tr("--models 后面要跟目录", "--models needs a directory")))?);
        let manifest = read_manifest(&dir).with_context(|| {
            if english() {
                format!("no usable manifest.json in {}", dir.display())
            } else {
                format!("{} 里没有可用的 manifest.json", dir.display())
            }
        })?;
        return Ok((Source::Dir(dir), manifest));
    }
    let beside = exe_dir().join("models");
    if let Ok(manifest) = read_manifest(&beside) {
        if complete(&beside, &manifest) {
            return Ok((Source::Dir(beside), manifest));
        }
    }

    let urls = model_urls();
    if urls.is_empty() {
        return Ok((Source::Embedded, parse_manifest(EMBEDDED_MANIFEST)?));
    }
    let cache = dirs::data_local_dir().unwrap_or_else(exe_dir).join("StarItem").join("models");
    fs::create_dir_all(&cache)?;
    let cached = read_manifest(&cache).ok().filter(|m| complete(&cache, m));
    for base in &urls {
        let base = base.trim_end_matches('/');
        status(tr("正在检查模型更新…", "Checking for model updates…").to_string());
        let Ok((text, remote)) = fetch_manifest(base) else {
            continue;
        };
        if remote.format != 1 {
            break;
        }
        if let Some(local) = &cached {
            if local.model_version == remote.model_version {
                return Ok((Source::Dir(cache), local.clone()));
            }
        }
        let mut ok = true;
        for entry in remote.entries() {
            let dest = cache.join(&entry.name);
            let have = fs::metadata(&dest).map(|m| m.len() == entry.size).unwrap_or(false)
                && sha256_file(&dest).map(|h| h == entry.sha256).unwrap_or(false);
            if have {
                continue;
            }
            if download(&format!("{base}/{}", entry.name), &dest, entry, status).is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            // manifest 最后写，保证它出现时文件已经齐全
            fs::write(cache.join("manifest.json"), text)?;
            return Ok((Source::Dir(cache), remote));
        }
    }
    // 没网或下载失败：有下载过的就用下载过的，否则退回内置模型
    match cached {
        Some(local) => Ok((Source::Dir(cache), local)),
        None => Ok((Source::Embedded, parse_manifest(EMBEDDED_MANIFEST)?)),
    }
}
