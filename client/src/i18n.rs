//! 界面语言：中文和英文两套。默认跟随系统，手动切换后记在本地。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static ENGLISH: AtomicBool = AtomicBool::new(false);

pub fn english() -> bool {
    ENGLISH.load(Ordering::Relaxed)
}

/// 按当前语言二选一。
pub fn tr<'a>(zh: &'a str, en: &'a str) -> &'a str {
    if english() {
        en
    } else {
        zh
    }
}

fn saved_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|dir| dir.join("StarItem").join("lang.txt"))
}

#[cfg(windows)]
fn system_is_chinese() -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // 语言 ID 的低 10 位是主语言，0x04 是中文
    unsafe { GetUserDefaultUILanguage() & 0x3ff == 0x04 }
}

#[cfg(not(windows))]
fn system_is_chinese() -> bool {
    std::env::var("LANG").map(|lang| lang.starts_with("zh")).unwrap_or(false)
}

/// 启动时定语言：命令行 --lang 优先，其次是上次手动选的，最后跟随系统（中文系统用中文，其余用英文）。
pub fn init(arg: Option<&str>) {
    let saved = saved_path().and_then(|path| std::fs::read_to_string(path).ok());
    let english = match arg.or(saved.as_deref().map(str::trim)) {
        Some("en") => true,
        Some("zh") => false,
        _ => !system_is_chinese(),
    };
    ENGLISH.store(english, Ordering::Relaxed);
}

/// 手动切换语言，并记住这次选择。
pub fn choose(english: bool) {
    ENGLISH.store(english, Ordering::Relaxed);
    if let Some(path) = saved_path() {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(path, if english { "en" } else { "zh" }).ok();
    }
}
