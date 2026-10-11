//! 去掉系统标题栏之后，窗口在 Windows 11 上默认是直角、没有阴影。这里向 DWM 要回圆角和阴影。

#[cfg(windows)]
pub fn polish(window: &impl raw_window_handle::HasWindowHandle) {
    use raw_window_handle::RawWindowHandle;

    #[repr(C)]
    struct Margins {
        left: i32,
        right: i32,
        top: i32,
        bottom: i32,
    }
    #[link(name = "dwmapi")]
    extern "system" {
        fn DwmSetWindowAttribute(hwnd: isize, attribute: u32, value: *const core::ffi::c_void, size: u32) -> i32;
        fn DwmExtendFrameIntoClientArea(hwnd: isize, margins: *const Margins) -> i32;
    }

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return;
    };
    let hwnd = handle.hwnd.get();
    // DWMWA_WINDOW_CORNER_PREFERENCE = 33，DWMWCP_ROUND = 2。Windows 10 不认识这个属性，失败了也无妨
    let round: u32 = 2;
    unsafe {
        DwmSetWindowAttribute(hwnd, 33, (&round as *const u32).cast(), 4);
        // 边框往客户区里伸 1 像素，DWM 才会给没有边框的窗口画阴影；这 1 像素会被界面盖住
        DwmExtendFrameIntoClientArea(hwnd, &Margins { left: 0, right: 0, top: 0, bottom: 1 });
    }
}

#[cfg(not(windows))]
pub fn polish(_window: &impl raw_window_handle::HasWindowHandle) {}
