//! 把图标和版本信息编进 exe：资源管理器和任务栏显示应用图标，文件属性里能看到产品名、版本和作者。
//! 直接调用 Windows SDK 的 rc.exe，不引入额外的构建依赖。版本号和作者取自 Cargo.toml。

use std::path::PathBuf;
use std::process::Command;

fn find_rc() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var("ProgramFiles(x86)").ok()?).join("Windows Kits").join("10").join("bin");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(root).ok()?.flatten().map(|e| e.path().join("x64").join("rc.exe")).filter(|p| p.is_file()).collect();
    versions.sort();
    versions.pop()
}

/// 生成资源脚本：图标加一份 VERSIONINFO。
fn resource_script() -> String {
    let env = |name: &str| std::env::var(name).unwrap_or_default();
    let icon = PathBuf::from(env("CARGO_MANIFEST_DIR")).join("assets").join("icon.ico").display().to_string().replace('\\', "/");
    let (major, minor, patch) = (env("CARGO_PKG_VERSION_MAJOR"), env("CARGO_PKG_VERSION_MINOR"), env("CARGO_PKG_VERSION_PATCH"));
    let version = env("CARGO_PKG_VERSION");
    // Cargo.toml 里有多位作者时用冒号隔开，这里换成逗号
    let author = env("CARGO_PKG_AUTHORS").replace(':', ", ");
    format!(
        r#"#pragma code_page(65001)
1 ICON "{icon}"
1 VERSIONINFO
FILEVERSION {major},{minor},{patch},0
PRODUCTVERSION {major},{minor},{patch},0
FILEOS 0x40004L
FILETYPE 0x1L
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "ProductName", "StarItem\0"
      VALUE "FileDescription", "StarItem\0"
      VALUE "CompanyName", "{author}\0"
      VALUE "Author", "{author}\0"
      VALUE "LegalCopyright", "Author: {author}\0"
      VALUE "FileVersion", "{version}\0"
      VALUE "ProductVersion", "{version}\0"
      VALUE "InternalName", "staritem\0"
      VALUE "OriginalFilename", "StarItem.exe\0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#
    )
}

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Some(rc) = find_rc() else {
        println!("cargo:warning=rc.exe not found, the exe will have no icon or version info");
        return;
    };
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let (script, res) = (out.join("app.rc"), out.join("app.res"));
    if std::fs::write(&script, resource_script()).is_err() {
        println!("cargo:warning=could not write the resource script, the exe will have no icon or version info");
        return;
    }
    let status = Command::new(rc).arg("/nologo").arg("/fo").arg(&res).arg(&script).status();
    match status {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg-bins={}", res.display()),
        _ => println!("cargo:warning=rc.exe failed, the exe will have no icon or version info"),
    }
}
