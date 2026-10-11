//! 界面风格："星の · 水墨"的配色——宣纸底、墨色字、黛青主色、朱砂点缀，细线边框，不用阴影。

use std::sync::Arc;

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Pos2, Rect, Stroke, Vec2};

#[derive(Clone, Copy)]
pub struct Palette {
    pub bg: Color32,
    pub surface: Color32,
    pub soft: Color32,
    pub ink: Color32,
    pub muted: Color32,
    pub line: Color32,
    pub green: Color32,
    pub green_soft: Color32,
    pub seal: Color32,
}

const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8)
}

pub const LIGHT: Palette = Palette {
    bg: rgb(0xf6f3eb),
    surface: rgb(0xfcfbf7),
    soft: rgb(0xede9df),
    ink: rgb(0x1f1e1b),
    muted: rgb(0x6e6a62),
    line: rgb(0xdfd9cc),
    green: rgb(0x2e4a4e),
    green_soft: rgb(0xe3e6df),
    seal: rgb(0xa8342b),
};

pub const DARK: Palette = Palette {
    bg: rgb(0x141413),
    surface: rgb(0x1c1c1a),
    soft: rgb(0x262522),
    ink: rgb(0xe9e5db),
    muted: rgb(0xa49f94),
    line: rgb(0x34322e),
    green: rgb(0xa8c4be),
    green_soft: rgb(0x253030),
    seal: rgb(0xd5594c),
};

pub fn palette(dark: bool) -> Palette {
    if dark {
        DARK
    } else {
        LIGHT
    }
}

pub fn bold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("bold".into()))
}

/// 拉丁字母用 Segoe UI，中文用微软雅黑；找不到的字体跳过，由 egui 自带字体兜底。
pub fn install_fonts(ctx: &egui::Context) {
    let dir = std::path::PathBuf::from(std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string())).join("Fonts");
    let mut fonts = egui::FontDefinitions::default();
    let mut load = |key: &str, candidates: &[&str]| -> bool {
        match candidates.iter().find_map(|name| std::fs::read(dir.join(name)).ok()) {
            Some(bytes) => {
                fonts.font_data.insert(key.to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
                true
            }
            None => false,
        }
    };
    let have = [
        ("latin", load("latin", &["segoeui.ttf"])),
        ("cjk", load("cjk", &["msyh.ttc", "msyhl.ttc", "Deng.ttf", "simhei.ttf", "NotoSansSC-VF.ttf"])),
        ("latin_bold", load("latin_bold", &["seguisb.ttf", "segoeuib.ttf"])),
        ("cjk_bold", load("cjk_bold", &["msyhbd.ttc", "Dengb.ttf", "simhei.ttf"])),
    ];
    let present = |key: &str| have.iter().any(|(k, ok)| *k == key && *ok);
    let fallback = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();

    let regular: Vec<String> = ["latin", "cjk"].iter().filter(|k| present(k)).map(|k| k.to_string()).chain(fallback.clone()).collect();
    let strong: Vec<String> = ["latin_bold", "cjk_bold", "latin", "cjk"].iter().filter(|k| present(k)).map(|k| k.to_string()).chain(fallback).collect();
    fonts.families.insert(FontFamily::Proportional, regular);
    fonts.families.insert(FontFamily::Name("bold".into()), strong);
    if present("cjk") {
        fonts.families.entry(FontFamily::Monospace).or_default().push("cjk".to_owned());
    }
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context, dark: bool) {
    let p = palette(dark);
    // egui 给明暗两套主题各存一份样式：先固定用哪一套，后面的设置才会写到正在显示的那一套上
    ctx.set_theme(if dark { egui::Theme::Dark } else { egui::Theme::Light });
    let mut visuals = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    visuals.override_text_color = Some(p.ink);
    visuals.panel_fill = p.bg;
    visuals.window_fill = p.surface;
    visuals.extreme_bg_color = p.surface;
    visuals.faint_bg_color = p.soft;
    visuals.hyperlink_color = p.green;
    visuals.window_stroke = Stroke::new(1.0_f32, p.line);
    visuals.window_shadow = egui::Shadow::NONE;
    visuals.popup_shadow = egui::Shadow::NONE;
    visuals.selection.bg_fill = p.green_soft;
    visuals.selection.stroke = Stroke::new(1.0_f32, p.green);
    visuals.text_cursor.stroke = Stroke::new(1.5_f32, p.green);

    let radius = CornerRadius::same(6);
    let w = &mut visuals.widgets;
    w.noninteractive.bg_fill = p.surface;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, p.line);
    w.noninteractive.fg_stroke = Stroke::new(1.0_f32, p.muted);
    for (state, fill, border, text) in [
        (&mut w.inactive, p.soft, p.line, p.ink),
        (&mut w.hovered, p.green_soft, p.green, p.ink),
        (&mut w.active, p.green_soft, p.green, p.ink),
        (&mut w.open, p.soft, p.green, p.ink),
    ] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = Stroke::new(1.0_f32, border);
        state.fg_stroke = Stroke::new(1.2_f32, text);
        state.corner_radius = radius;
        state.expansion = 0.0;
    }
    w.noninteractive.corner_radius = radius;
    ctx.set_visuals(visuals);

    ctx.style_mut(|style| {
        use egui::TextStyle::*;
        style.text_styles.insert(Body, FontId::proportional(14.0));
        style.text_styles.insert(Button, FontId::proportional(13.5));
        style.text_styles.insert(Small, FontId::proportional(11.5));
        style.text_styles.insert(Heading, bold(26.0));
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(10.0, 5.0);
        style.spacing.interact_size.y = 26.0;
        style.spacing.slider_rail_height = 4.0;
    });
}

/// 四角星。形状是凹的，从中心做三角扇来填充。
pub fn spark(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    const TIPS: [(f32, f32); 4] = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(center, color);
    let steps = 8;
    for k in 0..4 {
        let (ax, ay) = TIPS[k];
        let (bx, by) = TIPS[(k + 1) % 4];
        // 相邻两个尖之间是一段向中心凹进去的二次曲线
        let (cx, cy) = ((ax + bx) * 0.105, (ay + by) * 0.105);
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let u = 1.0 - t;
            let x = u * u * ax + 2.0 * u * t * cx + t * t * bx;
            let y = u * u * ay + 2.0 * u * t * cy + t * t * by;
            mesh.colored_vertex(center + Vec2::new(x, y) * radius, color);
        }
    }
    let n = (4 * steps) as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % n);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// 语言按钮的图标：带支架的地球仪，球有一半涂实——中文涂上半球，英文涂下半球。
/// spin 是球绕地轴转过的角度（弧度），经线跟着横向扫过，支架不动。back 是按钮的底色，涂实的那一半里经线用它来画。
pub fn globe(painter: &egui::Painter, center: Pos2, radius: f32, spin: f32, upper: bool, color: Color32, back: Color32) {
    use std::f32::consts::{FRAC_PI_2, PI};
    // 支架弧的半径，以球的半径为 1
    const ARM: f32 = 1.38;
    let frame = Stroke::new(1.2_f32, color);
    // 地轴向左歪 20 度
    let (sin, cos) = (-20.0_f32).to_radians().sin_cos();
    let at = |x: f32, y: f32| center + Vec2::new(x * cos - y * sin, x * sin + y * cos) * radius;
    // 从 from 到 to 的一段弧，f 给出每个角度对应的点
    let arc = |from: f32, to: f32, f: &dyn Fn(f32) -> Pos2| -> Vec<Pos2> { (0..=16).map(|i| f(from + (to - from) * i as f32 / 16.0)).collect() };

    // 支架：绕过左侧和底部、连着两极的半圆弧，两极各一小段轴，下面是立柱和底座
    painter.add(egui::Shape::line(arc(FRAC_PI_2, FRAC_PI_2 + PI, &|u| at(ARM * u.cos(), ARM * u.sin())), frame));
    painter.line_segment([at(0.0, -ARM), at(0.0, -1.0)], frame);
    painter.line_segment([at(0.0, ARM), at(0.0, 1.0)], frame);
    let foot = center + Vec2::new(0.0, ARM + 0.34) * radius;
    painter.line_segment([center + Vec2::new(0.0, ARM) * radius, foot], frame);
    painter.line_segment([foot - Vec2::new(0.72 * radius, 0.0), foot + Vec2::new(0.72 * radius, 0.0)], frame);

    // 球：涂实的半球和轮廓
    let side = if upper { -1.0 } else { 1.0 };
    painter.add(egui::Shape::convex_polygon(arc(0.0, PI, &|u| at(u.cos(), side * u.sin())), color, Stroke::NONE));
    painter.circle_stroke(center, radius, frame);
    // 经线每隔 60 度一条，只画朝外的那一半；以赤道为界分两段，落在涂实半球里的那段用底色
    for k in 0..6 {
        let lon = PI / 6.0 + spin + k as f32 * PI / 3.0;
        if lon.cos() <= 0.02 {
            continue;
        }
        for half in [-1.0, 1.0] {
            let line = arc(0.0, FRAC_PI_2, &|t| at(lon.sin() * t.cos(), half * t.sin()));
            painter.add(egui::Shape::line(line, Stroke::new(1.0_f32, if half == side { back } else { color })));
        }
    }
}

/// 透明底的示意棋盘格，格子和贴图像素对齐。
pub fn checker(painter: &egui::Painter, rect: Rect, cells: usize, a: Color32, b: Color32) {
    painter.rect_filled(rect, CornerRadius::ZERO, a);
    let step = rect.width() / cells as f32;
    for y in 0..cells {
        for x in 0..cells {
            if (x + y) % 2 == 1 {
                let min = rect.min + Vec2::new(x as f32 * step, y as f32 * step);
                painter.rect_filled(Rect::from_min_size(min, Vec2::splat(step)), CornerRadius::ZERO, b);
            }
        }
    }
}

/// 画到一半的像素星星：# 是已经画好的，+ 是还没落笔的那一角。
const STAR: [&[u8; 9]; 9] = [
    b"....#....",
    b"....#....",
    b"...###...",
    b"#########",
    b".#######.",
    b"..#####..",
    b"..#####..",
    b".###.+++.",
    b".#.....+.",
];

/// 徽标上某一点的颜色，坐标归一化到 [-1, 1]；圆角方块之外返回 None。
fn logo_color(x: f32, y: f32) -> Option<[u8; 3]> {
    const TILE: [u8; 3] = [0x1f, 0x1e, 0x1b];
    const PAPER: [u8; 3] = [0xf4, 0xef, 0xe4];
    const SEAL: [u8; 3] = [0xa8, 0x34, 0x2b];
    const GHOST: [u8; 3] = [0xdd, 0xd4, 0xc2];
    const PENCIL: [u8; 3] = [0x8f, 0xb5, 0xad];
    const WOOD: [u8; 3] = [0xe6, 0xdf, 0xcf];
    const ERASER: [u8; 3] = [0xd5, 0x59, 0x4c];
    let rounded = |x: f32, y: f32, half: f32, r: f32| {
        let (dx, dy) = ((x.abs() - (half - r)).max(0.0), (y.abs() - (half - r)).max(0.0));
        x.abs() <= half && y.abs() <= half && dx * dx + dy * dy <= r * r
    };
    if !rounded(x, y, 1.0, 0.46) {
        return None;
    }
    let mut color = TILE;

    // 画板：一张纸，上面是像素星星
    let (px, py) = (x + 0.14, y + 0.14);
    if rounded(px, py, 0.58, 0.10) {
        color = PAPER;
        let cell = 0.11;
        let (gx, gy) = (((px + 4.5 * cell) / cell).floor(), ((py + 4.5 * cell) / cell).floor());
        if (0.0..9.0).contains(&gx) && (0.0..9.0).contains(&gy) {
            match STAR[gy as usize][gx as usize] {
                b'#' => color = SEAL,
                b'+' => color = GHOST,
                _ => {}
            }
        }
    }

    // 铅笔：笔尖落在没画完的那一角，笔身指向右下。u 沿笔身方向，v 垂直于笔身
    let (rx, ry) = (x - 0.24, y - 0.34);
    let (u, v) = ((rx + ry) * std::f32::consts::FRAC_1_SQRT_2, (ry - rx) * std::f32::consts::FRAC_1_SQRT_2);
    let (len, tip, half) = (0.72, 0.24, 0.085);
    let inside = |grow: f32| u >= -grow && u <= len + grow && v.abs() <= half * (u / tip).clamp(0.0, 1.0) + grow;
    if inside(0.0) {
        color = if u < tip * 0.30 {
            TILE
        } else if u < tip {
            WOOD
        } else if u > len - 0.12 {
            ERASER
        } else {
            PENCIL
        };
    } else if inside(0.04) {
        color = TILE; // 一圈深色描边，让铅笔压在纸上也分得清
    }
    Some(color)
}

/// 把徽标渲染成 n×n 的 RGBA（4x4 超采样）。界面顶栏和窗口图标用的是同一张图。
pub fn logo_rgba(n: usize) -> Vec<u8> {
    const SS: usize = 4;
    let mut rgba = Vec::with_capacity(n * n * 4);
    for py in 0..n {
        for px in 0..n {
            let (mut acc, mut cover) = ([0f32; 3], 0f32);
            for sy in 0..SS {
                for sx in 0..SS {
                    let x = (px as f32 + (sx as f32 + 0.5) / SS as f32) / n as f32 * 2.0 - 1.0;
                    let y = (py as f32 + (sy as f32 + 0.5) / SS as f32) / n as f32 * 2.0 - 1.0;
                    if let Some(color) = logo_color(x, y) {
                        for k in 0..3 {
                            acc[k] += color[k] as f32;
                        }
                        cover += 1.0;
                    }
                }
            }
            if cover == 0.0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let alpha = cover / (SS * SS) as f32 * 255.0;
                rgba.extend_from_slice(&[(acc[0] / cover) as u8, (acc[1] / cover) as u8, (acc[2] / cover) as u8, alpha as u8]);
            }
        }
    }
    rgba
}

pub fn icon() -> egui::IconData {
    egui::IconData { rgba: logo_rgba(128), width: 128, height: 128 }
}
