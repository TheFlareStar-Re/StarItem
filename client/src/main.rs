#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;
mod frame;
mod i18n;
mod models;
mod theme;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use eframe::egui::{self, Align, Color32, CornerRadius, Frame, Layout, Margin, Rect, RichText, Sense, Stroke, Vec2};
use engine::{Choice, Engine, Network, Reference, Request, Sprite};
use i18n::tr;
use theme::Palette;

const EXAMPLES_ZH: &[&str] = &["红宝石镐", "golden crown", "草莓蛋糕", "青铜头盔", "red potion", "紫色的剑"];
const EXAMPLES_EN: &[&str] = &["ruby pickaxe", "red potion", "strawberry cake", "bronze helmet", "amethyst", "purple sword"];
const SIDE: i8 = 28;

enum Event {
    Status(String),
    Ready { version: String, steps: usize, guidance: f32, trained_32: bool, has_style: bool, two_networks: bool },
    Progress(usize, usize),
    Done(engine::Output, u64),
    Failed(String),
}

/// 状态栏显示的内容。存成枚举而不是现成的文字，切换语言时才能跟着变。
enum Status {
    Preparing,
    Ready,
    Busy,
    Done { seed: u64, style: Option<f32>, expanded: Option<String> },
    Broken,
    /// 后台线程和保存操作给出的现成文字。
    Text(String),
    /// 参考图载入的结果。模型加载完成时不会被"就绪"盖掉。
    Notice(String),
}

impl Status {
    fn text(&self) -> String {
        match self {
            Status::Preparing => tr("正在准备模型…", "Preparing the model…").to_string(),
            Status::Ready => tr("就绪", "Ready").to_string(),
            Status::Busy => tr("正在生成…", "Generating…").to_string(),
            Status::Broken => tr("模型不可用", "Model unavailable").to_string(),
            Status::Text(text) | Status::Notice(text) => text.clone(),
            Status::Done { seed, style, expanded } => {
                let mut text = format!("{} {seed}", tr("完成 · 种子", "Done · seed"));
                if let Some(style) = style {
                    text += &format!(" · {} {style:.1}", tr("画风", "style"));
                }
                // 提示词被补全过的话，把实际用的写法告诉用户
                if let Some(used) = expanded {
                    text += &format!(" · {}{used}", tr("已补全为：", "expanded to: "));
                }
                text
            }
        }
    }
}

/// 每次生成几张。两个网络各出一半时正好一边两张。
const COUNT: usize = 4;

#[derive(PartialEq)]
enum State {
    Loading,
    Ready,
    Busy,
    Broken,
}

struct App {
    events: Receiver<Event>,
    jobs: Sender<Request>,
    state: State,
    status: Status,
    error: String,
    model_version: String,
    trained_32: bool,
    dark: bool,
    // 调试用：--dark 启动时固定用暗色，不跟随系统
    force_dark: bool,
    // 指针贴着窗口边缘时要显示的缩放光标
    resize: Option<egui::CursorIcon>,
    // 语言切换过几次，地球仪每次转半圈
    lang_turns: u32,
    // 徽标按当前屏幕缩放下的实际像素大小生成，(边长, 纹理)
    logo: Option<(usize, egui::TextureHandle)>,
    prompt: String,
    size: usize,
    steps: usize,
    guidance: f32,
    random_seed: bool,
    seed: u64,
    expand: bool,
    auto_style: bool,
    style: f32,
    has_style: bool,
    progress: f32,
    results: Vec<(Sprite, egui::TextureHandle)>,
    // 每张结果是哪个网络画的；模型只有一个网络时为空
    result_networks: Vec<Network>,
    // 模型包里有两个出图网络时才显示"网络"选项
    two_networks: bool,
    network: Choice,
    result_prompt: String,
    // 参考图：(文件名, 缩成 16x16 的底稿, 底稿的纹理)，以及照着画的两个力度
    reference: Option<(String, Sprite, egui::TextureHandle)>,
    ref_strength: f32,
    ref_colors: f32,
    // "许可"窗口是否打开
    show_licenses: bool,
    // 调试用：--prompt 启动后自动生成，--ref 带上一张参考图，--licenses 打开许可窗口，--screenshot 截一张界面图后退出
    auto_prompt: Option<String>,
    auto_ref: Option<PathBuf>,
    shot: Option<PathBuf>,
    shot_wait: u32,
}

/// 后台线程：加载模型，然后逐个处理生成请求。
fn worker(jobs: Receiver<Request>, events: Sender<Event>, ctx: egui::Context) {
    let send = |e: Event| {
        events.send(e).ok();
        ctx.request_repaint();
    };
    let loaded = models::locate(&mut |s| send(Event::Status(s))).and_then(|(source, manifest)| {
        send(Event::Status(tr("正在加载模型…", "Loading the model…").to_string()));
        let info = (
            manifest.model_version.clone(),
            manifest.defaults.steps,
            manifest.defaults.guidance,
            manifest.trained_sizes.contains(&32),
            manifest.has_source,
        );
        Engine::load(&source, manifest).map(|e| (e, info))
    });
    let mut engine = match loaded {
        Ok((engine, (version, steps, guidance, trained_32, has_style))) => {
            let two_networks = engine.two_networks();
            send(Event::Ready { version, steps, guidance, trained_32, has_style, two_networks });
            engine
        }
        Err(e) => {
            send(Event::Failed(format!("{e:#}")));
            return;
        }
    };
    for req in jobs {
        let seed = req.seed;
        match engine.generate(&req, |i, n| send(Event::Progress(i, n))) {
            Ok(output) => send(Event::Done(output, seed)),
            Err(e) => send(Event::Status(format!("{}{e:#}", tr("生成失败：", "Generation failed: ")))),
        }
    }
}

fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.0, color);
}

/// 主题里的文字链接：没有边框的强调色小字。
fn text_link(ui: &mut egui::Ui, p: &Palette, text: &str) -> egui::Response {
    ui.add(egui::Button::new(RichText::new(text).font(theme::bold(12.0)).color(p.green)).frame(false))
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// 顶栏里的窗口按钮：最小化是一条横线，关闭是一个叉，悬停时关闭按钮用朱砂底。
fn window_button(ui: &mut egui::Ui, p: &Palette, close: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
    let hovered = response.hovered();
    if hovered {
        ui.painter().rect_filled(rect, CornerRadius::same(7), if close { p.seal } else { p.soft });
    }
    let stroke = Stroke::new(1.3_f32, if !hovered { p.muted } else if close { p.bg } else { p.ink });
    let (c, r) = (rect.center(), 5.0);
    if close {
        ui.painter().line_segment([c + Vec2::new(-r, -r), c + Vec2::new(r, r)], stroke);
        ui.painter().line_segment([c + Vec2::new(-r, r), c + Vec2::new(r, -r)], stroke);
    } else {
        ui.painter().line_segment([c + Vec2::new(-r, 0.0), c + Vec2::new(r, 0.0)], stroke);
    }
    response
}

fn segment(ui: &mut egui::Ui, p: &Palette, selected: bool, text: &str) -> egui::Response {
    let (fill, color) = if selected { (p.green, p.bg) } else { (p.surface, p.ink) };
    let border = Stroke::new(1.0_f32, if selected { p.green } else { p.line });
    ui.add(egui::Button::new(RichText::new(text).size(12.5).color(color)).fill(fill).stroke(border).min_size(Vec2::new(46.0, 26.0)))
}

fn write_png(path: &std::path::Path, sprite: &Sprite) -> image::ImageResult<()> {
    let size = sprite.size as u32;
    image::save_buffer(path, &sprite.rgba, size, size, image::ExtendedColorType::Rgba8)
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let force_dark = std::env::args().any(|a| a == "--dark");
        let dark = force_dark || cc.egui_ctx.system_theme() == Some(egui::Theme::Dark);
        frame::polish(cc);
        theme::install_fonts(&cc.egui_ctx);
        theme::apply(&cc.egui_ctx, dark);
        let (job_tx, job_rx) = channel();
        let (event_tx, event_rx) = channel();
        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || worker(job_rx, event_tx, ctx));
        let mut app = Self {
            events: event_rx,
            jobs: job_tx,
            state: State::Loading,
            status: Status::Preparing,
            error: String::new(),
            model_version: String::new(),
            trained_32: false,
            dark,
            force_dark,
            resize: None,
            lang_turns: 0,
            logo: None,
            prompt: String::new(),
            size: 16,
            steps: 30,
            guidance: 5.5,
            random_seed: true,
            seed: 0,
            expand: true,
            auto_style: true,
            style: 0.0,
            has_style: false,
            progress: 0.0,
            results: Vec::new(),
            result_networks: Vec::new(),
            two_networks: false,
            network: Choice::Auto,
            result_prompt: String::new(),
            reference: None,
            ref_strength: 0.65,
            ref_colors: 0.7,
            show_licenses: std::env::args().any(|a| a == "--licenses"),
            auto_prompt: arg_value("--prompt"),
            auto_ref: None,
            shot: arg_value("--screenshot").map(PathBuf::from),
            shot_wait: 0,
        };
        app.auto_ref = arg_value("--ref").map(PathBuf::from);
        app
    }

    /// 读一张图片当参考图。不能用的话（分不出物体、读不了）把原因显示在状态栏，原来的参考图保持不变。
    fn load_reference(&mut self, ctx: &egui::Context, path: &Path) {
        match engine::reference_draft(path) {
            Ok((draft, by_model)) => {
                let image = egui::ColorImage::from_rgba_unmultiplied([draft.size, draft.size], &draft.rgba);
                let texture = ctx.load_texture("reference", image, egui::TextureOptions::NEAREST);
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.reference = Some((name, draft, texture));
                self.status = Status::Notice(
                    if by_model {
                        tr(
                            "已从照片里抠出物体（见参考图一栏的缩略图），写上物品名再点生成",
                            "Cut the object out of the photo (see the thumbnail in the Reference row). Type the item name and generate",
                        )
                    } else {
                        tr("已载入参考图，写上物品名再点生成", "Reference loaded. Type the item name and generate")
                    }
                    .to_string(),
                );
            }
            Err(e) => self.status = Status::Notice(format!("{e:#}")),
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Status(s) => {
                    self.status = Status::Text(s);
                    if self.state == State::Busy {
                        self.state = State::Ready;
                    }
                }
                Event::Ready { version, steps, guidance, trained_32, has_style, two_networks } => {
                    self.has_style = has_style;
                    self.two_networks = two_networks;
                    self.model_version = version;
                    self.steps = steps;
                    self.guidance = guidance;
                    self.trained_32 = trained_32;
                    self.state = State::Ready;
                    if !matches!(self.status, Status::Notice(_)) {
                        self.status = Status::Ready;
                    }
                }
                Event::Progress(i, n) => self.progress = i as f32 / n as f32,
                Event::Done(output, seed) => {
                    let (used, style) = (output.prompt, output.style);
                    self.result_networks = if self.two_networks { output.networks } else { Vec::new() };
                    self.results = output
                        .sprites
                        .into_iter()
                        .enumerate()
                        .map(|(i, sprite)| {
                            let image = egui::ColorImage::from_rgba_unmultiplied([sprite.size, sprite.size], &sprite.rgba);
                            let texture = ctx.load_texture(format!("result{i}"), image, egui::TextureOptions::NEAREST);
                            (sprite, texture)
                        })
                        .collect();
                    self.seed = seed;
                    self.state = State::Ready;
                    self.status = Status::Done {
                        seed,
                        style: self.has_style.then_some(style),
                        expanded: (used != self.result_prompt.to_lowercase()).then_some(used),
                    };
                    if self.auto_style {
                        self.style = style;
                    }
                }
                Event::Failed(message) => {
                    self.state = State::Broken;
                    self.status = Status::Broken;
                    self.error = message;
                }
            }
        }
    }

    fn demo_hooks(&mut self, ctx: &egui::Context) {
        if self.state == State::Ready {
            if let Some(path) = self.auto_ref.take() {
                self.load_reference(ctx, &path);
            }
            if let Some(prompt) = self.auto_prompt.take() {
                self.prompt = prompt;
                self.start();
            }
        }
        let Some(path) = self.shot.clone() else {
            return;
        };
        if matches!(self.state, State::Ready | State::Broken) && self.auto_prompt.is_none() {
            self.shot_wait += 1;
            if self.shot_wait == 8 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }
        }
        let image = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = image {
            image::save_buffer(&path, image.as_raw(), image.size[0] as u32, image.size[1] as u32, image::ExtendedColorType::Rgba8).ok();
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint();
    }

    /// 没有系统边框后，窗口边缘的缩放要自己接：指针贴边时换成缩放光标，按下就交给系统开始缩放。
    fn resize_edges(&mut self, ctx: &egui::Context) {
        use egui::{CursorIcon as Cursor, ResizeDirection as Dir};
        const EDGE: f32 = 5.0;
        self.resize = None;
        let (rect, maximized, pos) = ctx.input(|i| (i.screen_rect(), i.viewport().maximized.unwrap_or(false), i.pointer.hover_pos()));
        let Some(pos) = pos.filter(|_| !maximized) else {
            return;
        };
        let (west, east) = (pos.x < rect.left() + EDGE, pos.x > rect.right() - EDGE);
        let (north, south) = (pos.y < rect.top() + EDGE, pos.y > rect.bottom() - EDGE);
        let (direction, cursor) = match (north, south, west, east) {
            (true, _, true, _) => (Dir::NorthWest, Cursor::ResizeNwSe),
            (true, _, _, true) => (Dir::NorthEast, Cursor::ResizeNeSw),
            (_, true, true, _) => (Dir::SouthWest, Cursor::ResizeNeSw),
            (_, true, _, true) => (Dir::SouthEast, Cursor::ResizeNwSe),
            (true, ..) => (Dir::North, Cursor::ResizeVertical),
            (_, true, ..) => (Dir::South, Cursor::ResizeVertical),
            (_, _, true, _) => (Dir::West, Cursor::ResizeHorizontal),
            (_, _, _, true) => (Dir::East, Cursor::ResizeHorizontal),
            _ => return,
        };
        self.resize = Some(cursor);
        if ctx.input(|i| i.pointer.primary_pressed()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
        }
    }

    fn start(&mut self) {
        let prompt = self.prompt.trim().to_string();
        if prompt.is_empty() || self.state != State::Ready {
            return;
        }
        if self.random_seed {
            self.seed = rand::random::<u32>() as u64;
        }
        let req = Request {
            prompt: prompt.clone(),
            size: self.size,
            count: COUNT,
            steps: self.steps,
            guidance: self.guidance,
            seed: self.seed,
            expand: self.expand,
            style: if self.auto_style { None } else { Some(self.style) },
            network: self.network,
            reference: self.reference.as_ref().map(|(_, draft, _)| Reference {
                draft: draft.clone(),
                strength: self.ref_strength,
                colors: self.ref_colors,
            }),
        };
        if self.jobs.send(req).is_ok() {
            self.result_prompt = prompt;
            self.progress = 0.0;
            self.state = State::Busy;
            self.status = Status::Busy;
        }
    }

    fn file_stem(&self, index: usize) -> String {
        let name: String = self.result_prompt.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
        format!("{}_{}", name.trim_matches('_'), index + 1)
    }

    fn save(&mut self, index: usize) {
        let sprite = &self.results[index].0;
        let Some(path) = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name(format!("{}.png", self.file_stem(index))).save_file() else {
            return;
        };
        self.status = Status::Text(match write_png(&path, sprite) {
            Ok(()) => format!("{} {}", tr("已保存", "Saved"), path.display()),
            Err(e) => format!("{}{e}", tr("保存失败：", "Save failed: ")),
        });
    }

    fn save_all(&mut self) {
        let Some(dir) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        let failed = (0..self.results.len())
            .filter(|&i| write_png(&dir.join(format!("{}.png", self.file_stem(i))), &self.results[i].0).is_err())
            .count();
        self.status = Status::Text(if failed == 0 {
            format!("{} {}", tr("已保存到", "Saved to"), dir.display())
        } else if i18n::english() {
            format!("{failed} sprite(s) failed to save")
        } else {
            format!("有 {failed} 张保存失败")
        });
    }

    fn header(&mut self, ctx: &egui::Context, p: &Palette) {
        let frame = Frame::new().fill(p.bg).inner_margin(Margin::symmetric(SIDE, 0));
        egui::TopBottomPanel::top("header").exact_height(60.0).show_separator_line(false).frame(frame).show(ctx, |ui| {
            ui.painter().hline(ui.clip_rect().x_range(), ui.max_rect().bottom() - 0.5, Stroke::new(1.0_f32, p.line));
            // 没有系统标题栏：顶栏的空白处用来拖动窗口，双击最大化或还原
            let bar = ui.interact(ui.max_rect().expand2(Vec2::new(SIDE as f32, 0.0)), ui.id().with("drag"), Sense::click_and_drag());
            if bar.double_clicked() {
                let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
            } else if bar.drag_started_by(egui::PointerButton::Primary) && self.resize.is_none() {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            ui.horizontal_centered(|ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(34.0), Sense::hover());
                let pixels = (rect.width() * ctx.pixels_per_point()).round() as usize;
                if self.logo.as_ref().map(|(n, _)| *n) != Some(pixels) {
                    let image = egui::ColorImage::from_rgba_unmultiplied([pixels, pixels], &theme::logo_rgba(pixels));
                    self.logo = Some((pixels, ctx.load_texture("logo", image, egui::TextureOptions::LINEAR)));
                }
                if let Some((_, texture)) = &self.logo {
                    let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    ui.painter().image(texture.id(), rect, uv, Color32::WHITE);
                }
                ui.add_space(4.0);
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label(RichText::new("Star").font(theme::bold(21.0)).color(p.ink));
                ui.label(RichText::new("Item").font(theme::bold(21.0)).color(p.seal));

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if window_button(ui, p, true).on_hover_text(tr("关闭", "Close")).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if window_button(ui, p, false).on_hover_text(tr("最小化", "Minimize")).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    }
                    ui.add_space(8.0);
                    // 语言切换：地球仪图标，中文涂上半球、英文涂下半球；点一下换成另一种，球跟着转半圈
                    let (rect, response) = ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
                    let hovered = response.hovered();
                    if hovered {
                        ui.painter().rect_filled(rect, CornerRadius::same(7), p.soft);
                    }
                    let spin = ctx.animate_value_with_time(response.id, self.lang_turns as f32 * std::f32::consts::PI, 0.6);
                    let (color, back) = if hovered { (p.green, p.soft) } else { (p.muted, p.bg) };
                    // 图标连支架不是上下左右对称的，球心略偏右上，整体才落在按钮正中
                    theme::globe(ui.painter(), rect.center() + Vec2::new(1.0, -1.0), 5.0, spin, !i18n::english(), color, back);
                    if response.on_hover_text(tr("Switch to English", "切换到中文")).clicked() {
                        i18n::choose(!i18n::english());
                        self.lang_turns += 1;
                    }
                    ui.add_space(3.0);
                    if !self.model_version.is_empty() {
                        ui.add_space(7.0);
                        let text = format!("{} {}", tr("模型", "Model"), self.model_version);
                        let galley = ui.painter().layout_no_wrap(text, egui::FontId::proportional(11.0), p.muted);
                        let (rect, _) = ui.allocate_exact_size(galley.size() + Vec2::new(18.0, 9.0), Sense::hover());
                        ui.painter().rect(rect, CornerRadius::same(5), p.surface, Stroke::new(1.0_f32, p.line), egui::StrokeKind::Inside);
                        ui.painter().galley(rect.min + Vec2::new(9.0, 4.5), galley, p.muted);
                    }
                });
            });
        });
    }

    fn footer(&mut self, ctx: &egui::Context, p: &Palette) {
        let frame = Frame::new().fill(p.soft).inner_margin(Margin::symmetric(SIDE, 0));
        egui::TopBottomPanel::bottom("status").exact_height(34.0).show_separator_line(false).frame(frame).show(ctx, |ui| {
            ui.painter().hline(ui.clip_rect().x_range(), ui.max_rect().top() + 0.5, Stroke::new(1.0_f32, p.line));
            ui.horizontal_centered(|ui| {
                if self.state == State::Loading {
                    ui.add(egui::Spinner::new().size(11.0).color(p.muted));
                } else {
                    dot(ui, match self.state {
                        State::Ready => p.green,
                        _ => p.seal,
                    });
                }
                ui.label(RichText::new(self.status.text()).size(12.0).color(p.muted));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("FOLLOW THE STARS.").size(10.0).color(p.muted).extra_letter_spacing(1.6));
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                    theme::spark(ui.painter(), rect.center(), 5.0, p.seal);
                    ui.add_space(10.0);
                    if text_link(ui, p, tr("许可", "Licenses")).clicked() {
                        self.show_licenses = !self.show_licenses;
                    }
                });
            });
        });
    }

    /// "许可"窗口：本软件各部分的许可，以及随 exe 分发的第三方模型的许可证全文。
    fn licenses_window(&mut self, ctx: &egui::Context, p: &Palette) {
        let mut open = self.show_licenses;
        egui::Window::new(tr("许可", "Licenses"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_size([640.0, 440.0])
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .frame(Frame::window(&ctx.style()).fill(p.surface).stroke(Stroke::new(1.0_f32, p.line)).inner_margin(Margin::same(16)))
            .show(ctx, |ui| {
                ui.set_width(640.0);
                egui::ScrollArea::vertical().max_height(440.0).auto_shrink([false, false]).show(ui, |ui| {
                    let line = |ui: &mut egui::Ui, text: &str| {
                        ui.label(RichText::new(text).size(12.5).color(p.ink));
                    };
                    line(ui, tr("StarItem 客户端的代码：MIT 许可。", "The StarItem client's code: MIT License."));
                    line(
                        ui,
                        tr(
                            "SIC 模型权重：CC BY-NC-SA 4.0，只能用于非商业用途。这个程序内置了权重，所以整体也只能用于非商业用途。",
                            "The SIC model weights: CC BY-NC-SA 4.0, non-commercial use only. This program bundles the weights, so as a whole it is non-commercial too.",
                        ),
                    );
                    line(
                        ui,
                        tr(
                            "文本编码器 multilingual-e5-small（intfloat）：MIT 许可。",
                            "The text encoder, multilingual-e5-small (intfloat): MIT License.",
                        ),
                    );
                    line(
                        ui,
                        tr(
                            "抠图模型 U²-Net-p：Apache License 2.0，原样分发。出处和许可证全文如下。",
                            "The matting model, U²-Net-p: Apache License 2.0, distributed unmodified. Its origin and the full license text follow.",
                        ),
                    );
                    ui.add_space(10.0);
                    ui.label(RichText::new(models::MATTING_LICENSE).monospace().size(11.0).color(p.muted));
                });
            });
        self.show_licenses = open;
    }

    fn prompt_card(&mut self, ui: &mut egui::Ui, p: &Palette) {
        let ready = self.state == State::Ready;
        let id = ui.make_persistent_id("prompt");
        let focused = ui.memory(|m| m.has_focus(id));
        let border = if focused { Stroke::new(1.5_f32, p.green) } else { Stroke::new(1.0_f32, p.line) };
        let mut submit = false;
        Frame::new().fill(p.surface).stroke(border).corner_radius(10).inner_margin(Margin { left: 14, right: 8, top: 8, bottom: 8 }).show(ui, |ui| {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(18.0, 40.0), Sense::hover());
                theme::spark(ui.painter(), rect.center(), 7.0, if focused { p.seal } else { p.muted });
                let button_width = if i18n::english() { 118.0 } else { 104.0 };
                let edit = egui::TextEdit::singleline(&mut self.prompt)
                    .id(id)
                    .frame(false)
                    .font(egui::FontId::proportional(16.0))
                    .margin(Margin::symmetric(4, 9))
                    .hint_text(RichText::new(tr("描述一个物品，例如：红宝石镐 / golden crown", "Describe an item, e.g. ruby pickaxe / golden crown")).color(p.muted))
                    .desired_width(ui.available_width() - button_width - 10.0);
                let response = ui.add(edit);
                submit |= response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                ui.scope(|ui| {
                    let w = &mut ui.visuals_mut().widgets;
                    w.inactive.weak_bg_fill = p.green;
                    w.hovered.weak_bg_fill = p.ink;
                    w.active.weak_bg_fill = p.ink;
                    for state in [&mut w.inactive, &mut w.hovered, &mut w.active] {
                        state.bg_stroke = Stroke::NONE;
                        state.corner_radius = CornerRadius::same(7);
                    }
                    let label = RichText::new(tr("生成  ↗", "Generate  ↗")).font(theme::bold(14.0)).color(p.bg);
                    submit |= ui.add_enabled(ready, egui::Button::new(label).min_size(Vec2::new(button_width, 40.0))).clicked();
                });
            });
        });
        // 进度：输入框下面的一条细线
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 3.0), Sense::hover());
        if self.state == State::Busy {
            ui.painter().rect_filled(rect, CornerRadius::same(2), p.soft);
            let done = Rect::from_min_size(rect.min, Vec2::new(rect.width() * self.progress, rect.height()));
            ui.painter().rect_filled(done, CornerRadius::same(2), p.green);
        }
        if submit {
            self.start();
        }
    }

    fn options(&mut self, ui: &mut egui::Ui, p: &Palette) {
        let muted = |text: &str| RichText::new(text).size(12.0).color(p.muted);
        let row = Vec2::new(ui.available_width(), 30.0);
        ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            // 32x 暂不开放：只有模型用 32x 数据训练过，才显示分辨率选项
            if self.trained_32 {
                ui.label(muted(tr("分辨率", "Size")));
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if segment(ui, p, self.size == 16, "16x").clicked() {
                        self.size = 16;
                    }
                    if segment(ui, p, self.size == 32, "32x").clicked() {
                        self.size = 32;
                    }
                });
                ui.add_space(10.0);
            }
            ui.scope(|ui| {
                // 滑条已走过的部分用主色
                ui.visuals_mut().selection.bg_fill = p.green;
                // 英文标签更长，滑条收窄一点，最小窗口宽度下才排得下
                let narrow = if i18n::english() { 24.0 } else { 0.0 };
                ui.spacing_mut().slider_width = 120.0 - narrow;
                ui.label(muted(tr("引导强度", "Guidance")));
                ui.add(egui::Slider::new(&mut self.guidance, 1.0..=10.0).fixed_decimals(1).trailing_fill(true))
                    .on_hover_text(tr("越高越贴近描述，越低变化越多", "Higher sticks closer to the prompt, lower gives more variety"));
                ui.add_space(10.0);
                ui.spacing_mut().slider_width = 90.0 - narrow;
                ui.label(muted(tr("步数", "Steps")));
                ui.add(egui::Slider::new(&mut self.steps, 10..=50).trailing_fill(true));
            });
            ui.add_space(10.0);
            ui.checkbox(&mut self.random_seed, RichText::new(tr("随机种子", "Random seed")).size(12.5));
            ui.add_enabled(!self.random_seed, egui::DragValue::new(&mut self.seed).speed(1.0));
            ui.add_space(10.0);
            ui.checkbox(&mut self.expand, RichText::new(tr("名称补全", "Name completion")).size(12.5)).on_hover_text(tr(
                "认出常见物品的名字后，自动补上模型训练时用的中英文名称",
                "When a common item name is recognised, add the Chinese and English names the model was trained with",
            ));
        });
        if !self.has_style {
            return;
        }
        let row = Vec2::new(ui.available_width(), 30.0);
        ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.label(muted(tr("画风", "Style")));
            ui.checkbox(&mut self.auto_style, RichText::new(tr("自动", "Auto")).size(12.5)).on_hover_text(tr(
                "模组里常见的物品用手绘风格，其他物品混入合成风格，画得更像但没那么精致",
                "Items common in mods use the hand-drawn style; other items mix in the synthetic style, which is more recognisable but less polished",
            ));
            ui.label(muted(tr("手绘", "Hand-drawn")));
            ui.scope(|ui| {
                ui.visuals_mut().selection.bg_fill = p.green;
                ui.spacing_mut().slider_width = 140.0;
                ui.add_enabled(!self.auto_style, egui::Slider::new(&mut self.style, 0.0..=1.0).fixed_decimals(1).trailing_fill(true));
            });
            ui.label(muted(tr("合成", "Synthetic")));
            if self.two_networks {
                ui.add_space(14.0);
                ui.label(muted(tr("网络", "Network"))).on_hover_text(tr(
                    "模型里有两个出图网络。Jang 认得名字的物品画得准，Mo 更听颜色词、对没见过的组合更稳。自动：恰好是认得的名字用 Jang，其余两个各出一半",
                    "The model holds two networks. Jang draws items it knows by name accurately; Mo follows color words and new combinations more readily. Auto uses Jang for names it knows and splits everything else between both",
                ));
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for (choice, label) in [
                        (Choice::Auto, tr("自动", "Auto")),
                        (Choice::Only(Network::Mo), "Mo"),
                        (Choice::Only(Network::Jang), "Jang"),
                    ] {
                        if segment(ui, p, self.network == choice, label).clicked() {
                            self.network = choice;
                        }
                    }
                });
            }
        });
    }

    /// 参考图那一行：没选时是一个选择按钮，选了以后显示底稿的缩略图、文件名和两个力度滑条。
    fn reference_row(&mut self, ui: &mut egui::Ui, p: &Palette) {
        let muted = |text: &str| RichText::new(text).size(12.0).color(p.muted);
        let current = self.reference.as_ref().map(|(name, _, texture)| (name.clone(), texture.id()));
        let (mut pick, mut clear) = (false, false);
        let row = Vec2::new(ui.available_width(), 30.0);
        ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.label(muted(tr("参考图", "Reference")));
            let Some((name, texture)) = current else {
                ui.scope(|ui| {
                    ui.visuals_mut().widgets.inactive.weak_bg_fill = p.surface;
                    ui.spacing_mut().button_padding = Vec2::new(9.0, 3.0);
                    pick = ui.add(egui::Button::new(RichText::new(tr("选择图片…", "Choose an image…")).size(12.0))).clicked();
                });
                ui.label(muted(tr(
                    "可选。一个物体的图片或照片，也可以直接拖进窗口",
                    "Optional. A picture or photo of one object; you can also drop it onto the window",
                )));
                return;
            };
            // 底稿的缩略图：让用户看到模型实际照着画的是什么
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
            theme::checker(ui.painter(), rect, 4, p.soft, p.bg);
            let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(texture, rect, uv, Color32::WHITE);
            ui.painter().rect_stroke(rect, CornerRadius::same(2), Stroke::new(1.0_f32, p.line), egui::StrokeKind::Outside);
            let short: String = if name.chars().count() > 18 { name.chars().take(17).chain(['…']).collect() } else { name.clone() };
            ui.label(RichText::new(short).size(12.0).color(p.ink)).on_hover_text(name);
            clear = text_link(ui, p, tr("移除", "Remove")).clicked();
            pick = text_link(ui, p, tr("换一张", "Change")).clicked();
            ui.add_space(10.0);
            ui.scope(|ui| {
                ui.visuals_mut().selection.bg_fill = p.green;
                let narrow = if i18n::english() { 16.0 } else { 0.0 };
                ui.spacing_mut().slider_width = 96.0 - narrow;
                ui.label(muted(tr("变化幅度", "Freedom")));
                ui.add(egui::Slider::new(&mut self.ref_strength, 0.3..=0.9).fixed_decimals(2).trailing_fill(true)).on_hover_text(tr(
                    "越低越贴近参考图的形状，越高画得越干净、但可能走样",
                    "Lower stays closer to the reference's shape; higher is cleaner but may drift away from it",
                ));
                ui.add_space(6.0);
                ui.spacing_mut().slider_width = 80.0 - narrow;
                ui.label(muted(tr("跟随颜色", "Keep colors")));
                ui.add(egui::Slider::new(&mut self.ref_colors, 0.0..=1.0).fixed_decimals(1).trailing_fill(true)).on_hover_text(tr(
                    "把结果的颜色拉回参考图；调到 0 就让模型自己选颜色",
                    "Pulls the result's colors back to the reference; at 0 the model picks its own colors",
                ));
            });
        });
        if clear {
            self.reference = None;
        }
        if pick {
            let dialog = rfd::FileDialog::new().add_filter(tr("图片", "Images"), &["png", "jpg", "jpeg"]);
            if let Some(path) = dialog.pick_file() {
                let ctx = ui.ctx().clone();
                self.load_reference(&ctx, &path);
            }
        }
    }

    /// 返回这张卡片上是否点了保存。
    fn card(&self, ui: &mut egui::Ui, p: &Palette, index: usize, size: Vec2) -> bool {
        let pad = 12.0;
        let canvas_size = Vec2::new(size.x - 2.0 * pad, size.y - pad - 38.0);
        let canvas_side = canvas_size.min_elem();
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let result = self.results.get(index);
        let hovered = result.is_some() && ui.rect_contains_pointer(rect);
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(11), p.surface);
        painter.rect_stroke(rect, CornerRadius::same(11), Stroke::new(1.0_f32, if hovered { p.green } else { p.line }), egui::StrokeKind::Inside);
        let canvas = Rect::from_min_size(rect.min + Vec2::splat(pad), canvas_size);
        painter.rect_filled(canvas, CornerRadius::same(7), p.soft);
        match result {
            Some((sprite, texture)) => {
                // 每个贴图像素占整数个界面像素，边缘才不会糊
                let pixel = ((canvas_side - 20.0) / sprite.size as f32).floor().max(1.0);
                let area = Rect::from_center_size(canvas.center(), Vec2::splat(pixel * sprite.size as f32));
                theme::checker(painter, area, 8, p.soft, p.bg);
                let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                painter.image(texture.id(), area, uv, Color32::WHITE);
            }
            None => theme::spark(painter, canvas.center(), 14.0, p.line),
        }

        let foot = Rect::from_min_max(egui::pos2(canvas.left() + 2.0, canvas.bottom()), egui::pos2(canvas.right() - 2.0, rect.bottom()));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(foot).layout(Layout::left_to_right(Align::Center)));
        child.label(RichText::new(format!("0{}", index + 1)).size(11.0).color(p.muted).extra_letter_spacing(1.2));
        if let (Some(_), Some(network)) = (result, self.result_networks.get(index)) {
            let label = match network {
                Network::Mo => "Mo",
                Network::Jang => "Jang",
            };
            child.label(RichText::new(label).size(11.0).color(p.muted));
        }
        let mut save = false;
        child.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if result.is_some() {
                save = text_link(ui, p, tr("保存 PNG", "Save PNG")).clicked();
            } else {
                ui.label(RichText::new(tr("等待生成", "Waiting")).size(11.0).color(p.muted));
            }
        });
        save
    }

    fn body(&mut self, ui: &mut egui::Ui, p: &Palette) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            dot(ui, p.green);
            ui.label(RichText::new("TEXT TO SPRITE").font(theme::bold(10.5)).color(p.green).extra_letter_spacing(2.0));
        });
        ui.add_space(2.0);
        ui.label(RichText::new(tr("一句话，画一个物品。", "One line, one item.")).font(theme::bold(27.0)).color(p.ink));
        let intro = tr(
            "输入物品名或一句描述，每次生成四张像素贴图，挑喜欢的保存。中英文都可以。",
            "Type an item name or a short description. You get four pixel sprites each time; save the ones you like. Chinese and English both work.",
        );
        ui.label(RichText::new(intro).size(13.0).color(p.muted));
        ui.add_space(10.0);

        self.prompt_card(ui, p);

        let mut picked = None;
        let row = Vec2::new(ui.available_width(), 26.0);
        ui.allocate_ui_with_layout(row, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(RichText::new(tr("试试", "Try")).size(12.0).color(p.muted));
            ui.scope(|ui| {
                ui.visuals_mut().widgets.inactive.weak_bg_fill = p.surface;
                ui.spacing_mut().button_padding = Vec2::new(9.0, 3.0);
                for example in if i18n::english() { EXAMPLES_EN } else { EXAMPLES_ZH } {
                    if ui.add(egui::Button::new(RichText::new(*example).size(12.0))).clicked() {
                        picked = Some(example.to_string());
                    }
                }
            });
        });
        if let Some(example) = picked {
            self.prompt = example;
            self.start();
        }
        ui.add_space(6.0);
        self.options(ui, p);
        self.reference_row(ui, p);

        if self.state == State::Broken {
            ui.add_space(10.0);
            Frame::new().fill(p.surface).stroke(Stroke::new(1.0_f32, p.seal)).corner_radius(8).inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| {
                ui.label(RichText::new(&self.error).size(13.0).color(p.seal));
            });
        }

        ui.add_space(16.0);
        let mut save_all = false;
        ui.horizontal(|ui| {
            dot(ui, p.green);
            ui.label(RichText::new(tr("生成结果", "Results")).font(theme::bold(15.0)).color(p.ink));
            if !self.result_prompt.is_empty() && !self.results.is_empty() {
                let quoted = if i18n::english() { format!("“{}”", self.result_prompt) } else { format!("「{}」", self.result_prompt) };
                ui.label(RichText::new(quoted).size(12.0).color(p.muted));
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if !self.results.is_empty() {
                    save_all = text_link(ui, p, tr("全部保存到文件夹 →", "Save all to folder →")).clicked();
                }
            });
        });
        ui.add_space(4.0);

        let gap = 14.0;
        let width = ((ui.available_width() - (COUNT - 1) as f32 * gap) / COUNT as f32).max(150.0);
        // 画布最多是正方形；窗口矮的时候压扁卡片，保证几张都在一屏里
        let room = ui.clip_rect().bottom() - ui.cursor().top() - 6.0;
        let size = Vec2::new(width, room.clamp(170.0, width + 26.0));
        let mut save = None;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for index in 0..COUNT {
                if self.card(ui, p, index, size) {
                    save = Some(index);
                }
            }
        });
        if let Some(index) = save {
            self.save(index);
        }
        if save_all {
            self.save_all();
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
        self.demo_hooks(ctx);
        self.resize_edges(ctx);
        // 拖进窗口的图片当参考图用
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.load_reference(ctx, &path);
        }
        // 明暗跟随系统，系统设置变了就换一套配色
        let dark = self.force_dark || ctx.system_theme() == Some(egui::Theme::Dark);
        if dark != self.dark {
            self.dark = dark;
            theme::apply(ctx, dark);
        }
        let p = theme::palette(self.dark);
        self.header(ctx, &p);
        self.footer(ctx, &p);
        if self.show_licenses {
            self.licenses_window(ctx, &p);
        }
        let frame = Frame::new().fill(p.bg).inner_margin(Margin::symmetric(SIDE, 14));
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.body(ui, &p));
        });
        // 窗口自己的一圈细边框，以及贴边时的缩放光标（放在最后，盖过控件设的光标）
        let frame = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("window frame")));
        frame.rect_stroke(ctx.screen_rect(), CornerRadius::ZERO, Stroke::new(1.0_f32, p.line), egui::StrokeKind::Inside);
        if let Some(cursor) = self.resize {
            ctx.set_cursor_icon(cursor);
        }
    }
}

fn main() -> eframe::Result {
    // 调试用：把徽标导出成 PNG 后退出
    if let Some(path) = arg_value("--logo-png") {
        image::save_buffer(path, &theme::logo_rgba(256), 256, 256, image::ExtendedColorType::Rgba8).ok();
        return Ok(());
    }
    i18n::init(arg_value("--lang").as_deref());
    // 调试用：把 --ref 的参考图缩成底稿存成 PNG 后退出
    if let (Some(reference), Some(out)) = (arg_value("--ref"), arg_value("--draft-png")) {
        match engine::reference_draft(Path::new(&reference)) {
            Ok((draft, _)) => write_png(Path::new(&out), &draft).ok(),
            Err(e) => std::fs::write(&out, format!("{e:#}")).ok(),
        };
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // 不要系统标题栏，最小化和关闭按钮画在顶栏里
            .with_decorations(false)
            .with_inner_size([920.0, 740.0])
            .with_min_inner_size([800.0, 600.0])
            .with_icon(theme::icon()),
        ..Default::default()
    };
    eframe::run_native("StarItem", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
