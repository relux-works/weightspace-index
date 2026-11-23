//! Interactive terminal UI (`wsp` / `wsp tui`) for choosing an inference
//! engine, browsing locally installed bases and newest index patches,
//! downloading bases/patches with SHA-256 verification, applying replacements,
//! and either launching the engine or printing absolute artifact paths.

use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::download::{PullFormat, PulledPatch, download_base_profile, pull_patch};
use crate::engines::{
    DiscoveryRoots, EngineKind, EngineStatus, LaunchPlan, LocalModel, best_local_base,
    detect_engines, discover_local_models, matches_base, plan_launch,
};
use crate::error::ToolError;
use crate::ops::{ApplyStack, ConflictPolicy, Cx, PatchSpec, Progress, apply_stack};
use crate::registry::{
    AppPaths, CatalogPatch, MergedCatalog, RegistryConfig, load_merged_catalog, sync_registry,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    SelectEngine,
    SelectPatch,
    Prepare,
    Ready,
}

#[derive(Clone, Debug)]
pub struct PatchRow {
    pub patch_index: usize,
    pub local_base: Option<PathBuf>,
    pub local_base_single_st: bool,
}

#[derive(Clone, Debug)]
pub enum TuiExitAction {
    Quit,
    PrintPaths(LaunchPlan),
    LaunchEngine(LaunchPlan),
}

/// Headless state machine powering the TUI (unit-testable without a terminal).
pub struct TuiApp {
    pub screen: Screen,
    pub engines: Vec<EngineStatus>,
    pub engine_idx: usize,
    pub catalog: MergedCatalog,
    pub local_models: Vec<LocalModel>,
    pub rows: Vec<PatchRow>,
    pub patch_row_idx: usize,
    pub filter: String,
    pub filtering: bool,
    pub selected_patch_idx: Option<usize>,
    pub base_path: Option<PathBuf>,
    pub base_single_st: bool,
    pub pulled: Option<PulledPatch>,
    pub patched_model_path: Option<PathBuf>,
    pub launch_plan: Option<LaunchPlan>,
    pub ready_choice: usize,
    pub status_line: String,
    pub error_line: Option<String>,
}

impl TuiApp {
    pub fn new(
        engines: Vec<EngineStatus>,
        catalog: MergedCatalog,
        local_models: Vec<LocalModel>,
    ) -> Self {
        let mut app = Self {
            screen: Screen::SelectEngine,
            engines,
            engine_idx: 0,
            catalog,
            local_models,
            rows: Vec::new(),
            patch_row_idx: 0,
            filter: String::new(),
            filtering: false,
            selected_patch_idx: None,
            base_path: None,
            base_single_st: false,
            pulled: None,
            patched_model_path: None,
            launch_plan: None,
            ready_choice: 0,
            status_line: "Choose an inference engine (Enter to continue, q to quit).".into(),
            error_line: None,
        };
        app.rebuild_rows();
        app
    }

    pub fn selected_engine(&self) -> EngineKind {
        self.engines
            .get(self.engine_idx)
            .map(|e| e.kind)
            .unwrap_or(EngineKind::LlamaCpp)
    }

    pub fn selected_engine_status(&self) -> Option<&EngineStatus> {
        self.engines.get(self.engine_idx)
    }

    pub fn selected_patch(&self) -> Option<&CatalogPatch> {
        self.selected_patch_idx
            .and_then(|i| self.catalog.patches.get(i))
    }

    /// Rebuilds the visible patch list:
    /// 1) Patches whose base model is already downloaded locally come first.
    /// 2) Remaining index patches follow, sorted by newest added first, marked
    ///    `[base not downloaded]`.
    pub fn rebuild_rows(&mut self) {
        let engine = self.selected_engine();
        let needle = self.filter.trim().to_ascii_lowercase();

        let mut local_first = Vec::new();
        let mut remote_rest = Vec::new();

        for (idx, p) in self.catalog.patches.iter().enumerate() {
            if !needle.is_empty() {
                let hay = format!(
                    "{} {} {} {} {}",
                    p.base,
                    p.slug,
                    p.title,
                    p.summary,
                    p.tags.join(" ")
                )
                .to_ascii_lowercase();
                if !needle.split_whitespace().all(|tok| hay.contains(tok)) {
                    continue;
                }
            }
            let local = best_local_base(&self.local_models, &p.base, engine);
            let row = PatchRow {
                patch_index: idx,
                local_base: local.map(|m| m.path.clone()),
                local_base_single_st: local.is_some_and(|m| m.has_single_safetensors),
            };
            if row.local_base.is_some() {
                local_first.push(row);
            } else {
                remote_rest.push(row);
            }
        }

        // `catalog.patches` is already sorted newest-added first, so preserving
        // order keeps both groups sorted by newest added.
        local_first.extend(remote_rest);
        self.rows = local_first;
        if self.patch_row_idx >= self.rows.len() {
            self.patch_row_idx = self.rows.len().saturating_sub(1);
        }
    }

    pub fn confirm_engine(&mut self) {
        self.screen = Screen::SelectPatch;
        self.error_line = None;
        self.rebuild_rows();
        let local_count = self.rows.iter().filter(|r| r.local_base.is_some()).count();
        self.status_line = format!(
            "Engine: {} · {} local-base matches, {} total patches (/ to filter, r to sync, b back)",
            self.selected_engine().label(),
            local_count,
            self.rows.len()
        );
    }

    pub fn confirm_patch(&mut self) {
        let Some(row) = self.rows.get(self.patch_row_idx).cloned() else {
            return;
        };
        self.selected_patch_idx = Some(row.patch_index);
        self.base_path = row.local_base;
        self.base_single_st = row.local_base_single_st;
        self.pulled = None;
        self.patched_model_path = None;
        self.launch_plan = None;
        self.ready_choice = 0;
        self.error_line = None;
        self.screen = Screen::Prepare;
        self.status_line = if self.base_path.is_some() {
            "Local base found. Press Enter to download & apply patch (or i for --in-place).".into()
        } else {
            "Base model not downloaded. Press d to download base, or Enter to pull patch/LoRA now."
                .into()
        };
    }

    pub fn build_ready_plan(&mut self) {
        let Some(patch) = self.selected_patch().cloned() else {
            return;
        };
        let engine = self.selected_engine();
        let adapter = self.pulled.as_ref().and_then(|p| {
            if engine.prefers_gguf() {
                p.gguf_adapter.as_deref().or(p.peft_adapter.as_deref())
            } else {
                p.peft_adapter.as_deref().or(p.gguf_adapter.as_deref())
            }
        });
        let plan = plan_launch(
            engine,
            &patch,
            self.base_path.as_deref(),
            self.patched_model_path.as_deref(),
            adapter,
        );
        self.launch_plan = Some(plan);
        self.ready_choice = 0;
        self.screen = Screen::Ready;
        self.status_line =
            "Patch ready. Choose: [1] Launch engine or [2] Print absolute paths.".into();
    }

    pub fn local_count_for_engine(&self, engine: EngineKind) -> usize {
        let mut matched = 0usize;
        for base_id in self.catalog.bases.keys() {
            if self
                .local_models
                .iter()
                .any(|m| matches_base(m, base_id) && engine_compatible(engine, m))
            {
                matched += 1;
            }
        }
        matched
    }
}

fn engine_compatible(engine: EngineKind, m: &LocalModel) -> bool {
    if engine.prefers_gguf() {
        true
    } else {
        m.format == crate::engines::ModelWeightFormat::Safetensors
    }
}

/// Runs the interactive TUI until the user exits, prints paths, or launches an engine.
pub fn run_tui(
    cfg: &RegistryConfig,
    paths: &AppPaths,
    registry_filter: Option<&str>,
) -> Result<TuiExitAction, ToolError> {
    let engines = detect_engines();
    let catalog = load_merged_catalog(cfg, paths, registry_filter, true)?;
    let roots = DiscoveryRoots::from_env(paths);
    let local_models = discover_local_models(&roots);
    let mut app = TuiApp::new(engines, catalog, local_models);

    enable_raw_mode().map_err(|e| ToolError::io("terminal", e))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| ToolError::io("terminal", e))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| ToolError::io("terminal", e))?;

    let outcome = event_loop(&mut terminal, &mut app, cfg, paths, registry_filter);

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    outcome
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    cfg: &RegistryConfig,
    paths: &AppPaths,
    registry_filter: Option<&str>,
) -> Result<TuiExitAction, ToolError> {
    loop {
        terminal
            .draw(|f| render(f, app))
            .map_err(|e| ToolError::io("terminal", e))?;

        if !event::poll(Duration::from_millis(120)).map_err(|e| ToolError::io("terminal", e))? {
            continue;
        }
        let Event::Key(key) = event::read().map_err(|e| ToolError::io("terminal", e))? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if app.filtering {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    app.filtering = false;
                }
                KeyCode::Backspace => {
                    app.filter.pop();
                    app.rebuild_rows();
                }
                KeyCode::Char(c) => {
                    app.filter.push(c);
                    app.rebuild_rows();
                }
                _ => {}
            }
            continue;
        }

        match app.screen {
            Screen::SelectEngine => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(TuiExitAction::Quit),
                KeyCode::Up | KeyCode::Char('k') => {
                    app.engine_idx = app.engine_idx.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if app.engine_idx + 1 < app.engines.len() {
                        app.engine_idx += 1;
                    }
                }
                KeyCode::Enter => app.confirm_engine(),
                _ => {}
            },
            Screen::SelectPatch => match key.code {
                KeyCode::Char('q') => return Ok(TuiExitAction::Quit),
                KeyCode::Esc | KeyCode::Char('b') => {
                    app.screen = Screen::SelectEngine;
                    app.status_line = "Choose an inference engine.".into();
                }
                KeyCode::Char('/') => {
                    app.filtering = true;
                }
                KeyCode::Char('r') => {
                    app.status_line = "Syncing configured registries...".into();
                    let _ = terminal.draw(|f| render(f, app));
                    let mut errs = Vec::new();
                    for reg in cfg.ordered_enabled() {
                        if let Err(e) = sync_registry(reg, paths) {
                            errs.push(e.to_string());
                        }
                    }
                    match load_merged_catalog(cfg, paths, registry_filter, false) {
                        Ok(cat) => {
                            app.catalog = cat;
                            app.rebuild_rows();
                            app.error_line = if errs.is_empty() {
                                None
                            } else {
                                Some(errs.join("; "))
                            };
                            app.status_line = format!(
                                "Synced {} registries ({} patches).",
                                app.catalog.loaded_registries.len(),
                                app.catalog.patches.len()
                            );
                        }
                        Err(e) => app.error_line = Some(e.to_string()),
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    app.patch_row_idx = app.patch_row_idx.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if app.patch_row_idx + 1 < app.rows.len() {
                        app.patch_row_idx += 1;
                    }
                }
                KeyCode::Enter => app.confirm_patch(),
                _ => {}
            },
            Screen::Prepare => match key.code {
                KeyCode::Char('q') => return Ok(TuiExitAction::Quit),
                KeyCode::Esc | KeyCode::Char('b') => {
                    app.screen = Screen::SelectPatch;
                    app.error_line = None;
                }
                KeyCode::Char('d') => {
                    handle_download_base(terminal, app, paths);
                }
                KeyCode::Enter => {
                    handle_pull_and_apply(terminal, app, paths, false);
                }
                KeyCode::Char('i') => {
                    handle_pull_and_apply(terminal, app, paths, true);
                }
                _ => {}
            },
            Screen::Ready => match key.code {
                KeyCode::Char('q') => return Ok(TuiExitAction::Quit),
                KeyCode::Esc | KeyCode::Char('b') => {
                    app.screen = Screen::SelectPatch;
                }
                KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('1') => {
                    app.ready_choice = 0;
                    if key.code == KeyCode::Char('1')
                        && let Some(plan) = app.launch_plan.clone()
                    {
                        return Ok(TuiExitAction::LaunchEngine(plan));
                    }
                }
                KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('2') => {
                    app.ready_choice = 1;
                    if key.code == KeyCode::Char('2')
                        && let Some(plan) = app.launch_plan.clone()
                    {
                        return Ok(TuiExitAction::PrintPaths(plan));
                    }
                }
                KeyCode::Enter => {
                    if let Some(plan) = app.launch_plan.clone() {
                        return if app.ready_choice == 0 {
                            Ok(TuiExitAction::LaunchEngine(plan))
                        } else {
                            Ok(TuiExitAction::PrintPaths(plan))
                        };
                    }
                }
                _ => {}
            },
        }
    }
}

fn handle_download_base(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    paths: &AppPaths,
) {
    let Some(patch) = app.selected_patch().cloned() else {
        return;
    };
    let Some(base_entry) = app.catalog.bases.get(&patch.base).cloned() else {
        app.error_line = Some(format!("Base profile metadata missing for {}", patch.base));
        return;
    };
    let Some(profile) = base_entry
        .profiles
        .iter()
        .find(|p| p.id == patch.target_profile)
        .or_else(|| base_entry.profiles.first())
        .cloned()
    else {
        app.error_line = Some(format!(
            "No downloadable profile recorded for {}",
            patch.base
        ));
        return;
    };

    app.error_line = None;
    let res = download_base_profile(&patch.base, &profile, paths, &mut |name, done, total| {
        app.status_line = format!(
            "Downloading base {} · {} ({}/{})",
            patch.base,
            name,
            human_bytes(done),
            human_bytes(total)
        );
        let _ = terminal.draw(|f| render(f, app));
    });
    match res {
        Ok(dir) => {
            app.base_path = Some(dir.clone());
            app.base_single_st = profile.files.len() == 1;
            app.status_line = format!(
                "Base downloaded to {}. Press Enter to pull & apply patch.",
                dir.display()
            );
        }
        Err(e) => {
            app.error_line = Some(format!("Base download failed: {e}"));
        }
    }
}

fn handle_pull_and_apply(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    paths: &AppPaths,
    in_place: bool,
) {
    let Some(patch) = app.selected_patch().cloned() else {
        return;
    };
    let engine = app.selected_engine();
    app.error_line = None;
    app.status_line = format!("Pulling {} ...", patch.address());
    let _ = terminal.draw(|f| render(f, app));

    let pulled = match pull_patch(
        &patch,
        paths,
        None,
        PullFormat::Auto,
        engine.prefers_gguf(),
        &mut |name, done, total| {
            app.status_line = format!(
                "Downloading {} ({}/{})",
                name,
                human_bytes(done),
                human_bytes(total)
            );
            let _ = terminal.draw(|f| render(f, app));
        },
    ) {
        Ok(p) => p,
        Err(e) => {
            app.error_line = Some(format!("Patch download failed: {e}"));
            return;
        }
    };

    // If this is a bit-exact replacement patch and we have a local single-safetensors base, apply it!
    if let Some(ref rep_file) = pulled.replacement_file
        && let Some(ref base_dir) = app.base_path
        && app.base_single_st
        && base_dir.is_dir()
    {
        app.status_line = format!("Applying bit-exact replacement {} ...", patch.slug);
        let _ = terminal.draw(|f| render(f, app));

        let out_dir = if in_place {
            None
        } else {
            Some(
                paths
                    .patched_models_dir()
                    .join(&patch.base)
                    .join(format!("{}-v{}", patch.slug, patch.version)),
            )
        };

        // Reuse existing managed patched directory if already created.
        if let Some(ref existing_out) = out_dir
            && existing_out.is_dir()
        {
            app.patched_model_path = Some(existing_out.clone());
            app.pulled = Some(pulled);
            app.build_ready_plan();
            return;
        }

        let cancelled = AtomicBool::new(false);
        let mut sink = |_: Progress| {};
        let mut cx = Cx::new(&cancelled, &mut sink);
        let req = ApplyStack {
            patches: vec![PatchSpec {
                path: rep_file.clone(),
                weight: 1.0,
            }],
            model: base_dir.clone(),
            out: out_dir.clone(),
            in_place,
            on_conflict: ConflictPolicy::Error,
            expect_tensor_sha256: BTreeMap::new(),
        };
        match apply_stack(&req, &mut cx) {
            Ok(_) => {
                app.patched_model_path = out_dir.or_else(|| Some(base_dir.clone()));
            }
            Err(e) => {
                app.error_line = Some(format!("Patch downloaded, but auto-apply refused: {e}"));
                app.pulled = Some(pulled);
                return;
            }
        }
    }

    app.pulled = Some(pulled);
    app.build_ready_plan();
}

fn render(f: &mut Frame, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(4),
        ])
        .split(f.area());

    // Header bar
    let step_label = match app.screen {
        Screen::SelectEngine => "Step 1/4 · Choose Inference Engine",
        Screen::SelectPatch => "Step 2/4 · Select Model & Patch",
        Screen::Prepare => "Step 3/4 · Download & Apply",
        Screen::Ready => "Step 4/4 · Launch or Print Paths",
    };
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " wsp ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  Weightspace Index  ·  "),
        Span::styled(step_label, Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!(
            "  ·  registries: {}",
            app.catalog.loaded_registries.join(", ")
        )),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Main body
    match app.screen {
        Screen::SelectEngine => render_engines(f, chunks[1], app),
        Screen::SelectPatch => render_patches(f, chunks[1], app),
        Screen::Prepare => render_prepare(f, chunks[1], app),
        Screen::Ready => render_ready(f, chunks[1], app),
    }

    // Status + error footer
    let mut footer_lines = vec![Line::from(Span::styled(
        &app.status_line,
        Style::default().fg(Color::Cyan),
    ))];
    if let Some(ref err) = app.error_line {
        footer_lines.push(Line::from(Span::styled(
            format!("Error: {err}"),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
    } else if app.filtering {
        footer_lines.push(Line::from(Span::styled(
            format!("Filter: {}_  (Enter/Esc to finish)", app.filter),
            Style::default().fg(Color::Yellow),
        )));
    } else {
        footer_lines.push(Line::from(Span::styled(
            "Keys: ↑/↓ move · Enter select · / filter · b back · q quit",
            Style::default().fg(Color::DarkGray),
        )));
    }
    let footer = Paragraph::new(footer_lines)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(footer, chunks[2]);
}

fn render_engines(f: &mut Frame, area: ratatui::layout::Rect, app: &TuiApp) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    let items: Vec<ListItem> = app
        .engines
        .iter()
        .map(|e| {
            let badge = if e.installed {
                Span::styled("[installed]   ", Style::default().fg(Color::Green))
            } else {
                Span::styled("[not in PATH] ", Style::default().fg(Color::Yellow))
            };
            let local_n = app.local_count_for_engine(e.kind);
            ListItem::new(Line::from(vec![
                badge,
                Span::styled(
                    format!("{:<20}", e.label),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ({local_n} local bases)"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.engine_idx));
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Inference Engines"),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");
    f.render_stateful_widget(list, cols[0], &mut state);

    let detail_lines = if let Some(e) = app.selected_engine_status() {
        let bin_str = e
            .binary_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Not detected in PATH".into());
        vec![
            Line::from(Span::styled(
                e.label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(e.description),
            Line::from(""),
            Line::from(vec![
                Span::styled("Binary: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(bin_str),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Install / setup:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(e.install_hint),
            Line::from(""),
            Line::from(Span::styled(
                "Note: even if the engine is not installed yet, you can still download & patch models and copy the exact paths/command.",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else {
        vec![]
    };
    let detail = Paragraph::new(detail_lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Engine Details"),
        );
    f.render_widget(detail, cols[1]);
}

fn render_patches(f: &mut Frame, area: ratatui::layout::Rect, app: &TuiApp) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(area);

    let items: Vec<ListItem> = app
        .rows
        .iter()
        .map(|row| {
            let p = &app.catalog.patches[row.patch_index];
            let status_span = if row.local_base.is_some() {
                Span::styled("[local base]     ", Style::default().fg(Color::Green))
            } else {
                Span::styled("[not downloaded] ", Style::default().fg(Color::Yellow))
            };
            let badges = p.format_badges().join("+");
            ListItem::new(Line::from(vec![
                status_span,
                Span::styled(&p.base, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" / "),
                Span::styled(&p.slug, Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("  [{badges}]"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    if !app.rows.is_empty() {
        state.select(Some(app.patch_row_idx));
    }
    let title = if app.filter.is_empty() {
        "Patches (local matches first, then newest added)"
    } else {
        "Patches (filtered)"
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");
    f.render_stateful_widget(list, cols[0], &mut state);

    let detail_lines = if let Some(row) = app.rows.get(app.patch_row_idx) {
        let p = &app.catalog.patches[row.patch_index];
        let mut lines = vec![
            Line::from(Span::styled(
                &p.title,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("{}  (v{}, {})", p.address(), p.version, p.kind)),
            Line::from(format!("Registry: {} · License: {}", p.registry, p.license)),
            Line::from(""),
            Line::from(p.summary.as_str()),
            Line::from(""),
        ];
        if let Some(ref h) = p.highlight {
            lines.push(Line::from(Span::styled(
                format!("Reported: {h}"),
                Style::default().fg(Color::Green),
            )));
            lines.push(Line::from(""));
        }
        lines.push(Line::from(format!(
            "Formats: {} · Total files: {}",
            p.format_badges().join(", "),
            human_bytes(p.total_bytes())
        )));
        if let Some(ref local) = row.local_base {
            lines.push(Line::from(Span::styled(
                format!("Local base: {}", local.display()),
                Style::default().fg(Color::Green),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                "Base model not downloaded locally yet.",
                Style::default().fg(Color::Yellow),
            )));
        }
        lines
    } else {
        vec![Line::from(
            "No matching patches. Press r to sync registries.",
        )]
    };
    let detail = Paragraph::new(detail_lines)
        .wrap(Wrap { trim: true })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Patch Details"),
        );
    f.render_widget(detail, cols[1]);
}

fn render_prepare(f: &mut Frame, area: ratatui::layout::Rect, app: &TuiApp) {
    let Some(p) = app.selected_patch() else {
        return;
    };
    let base_bytes: u64 = app
        .catalog
        .bases
        .get(&p.base)
        .and_then(|b| {
            b.profiles
                .iter()
                .find(|pr| pr.id == p.target_profile)
                .or_else(|| b.profiles.first())
        })
        .map(|pr| pr.files.iter().map(|f| f.size_bytes).sum())
        .unwrap_or(0);

    let mut lines = vec![
        Line::from(Span::styled(
            format!("{} — {}", p.address(), p.title),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(format!(
            "Target engine: {} · Kind: {} · Version: v{}",
            app.selected_engine().label(),
            p.kind,
            p.version
        )),
        Line::from(""),
    ];

    if let Some(ref base) = app.base_path {
        lines.push(Line::from(Span::styled(
            format!("✓ Base model ready locally: {}", base.display()),
            Style::default().fg(Color::Green),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "! Base model not downloaded locally (pinned {} profile: {})",
                p.target_profile,
                human_bytes(base_bytes)
            ),
            Style::default().fg(Color::Yellow),
        )));
        lines.push(Line::from(
            "  Press [d] to download the pinned base model from Hugging Face (no auth required).",
        ));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Actions:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    if app.base_path.is_none() {
        lines.push(Line::from("  [d]     Download base model first"));
    }
    if p.kind == "replacement" && app.base_single_st {
        lines.push(Line::from(
            "  [Enter] Download patch & apply into a safe managed copy (original untouched)",
        ));
        lines.push(Line::from(
            "  [i]     Download patch & apply --in-place (creates verified .weightspace-backup/)",
        ));
    } else {
        lines.push(Line::from(
            "  [Enter] Download & verify patch/LoRA adapter and prepare launch command",
        ));
    }
    lines.push(Line::from("  [b]     Back to patch list"));

    let body = Paragraph::new(lines).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Prepare & Apply"),
    );
    f.render_widget(body, area);
}

fn render_ready(f: &mut Frame, area: ratatui::layout::Rect, app: &TuiApp) {
    let Some(plan) = &app.launch_plan else {
        return;
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(6)])
        .split(area);

    let launch_label = if plan.engine_installed {
        format!(
            "[1] Launch {} with recommended parameters",
            plan.engine.label()
        )
    } else {
        format!(
            "[1] Show {} install guide + launch command (not in PATH)",
            plan.engine.label()
        )
    };
    let choices = vec![
        ListItem::new(Line::from(launch_label)),
        ListItem::new(Line::from(
            "[2] Print absolute paths to patched model / LoRA adapter (for advanced users)",
        )),
    ];
    let mut state = ListState::default();
    state.select(Some(app.ready_choice));
    let menu = List::new(choices)
        .block(Block::default().borders(Borders::ALL).title("Next Step"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▸ ");
    f.render_stateful_widget(menu, rows[0], &mut state);

    let mut info = Vec::new();
    if let Some(ref b) = plan.base_path {
        info.push(Line::from(format!("Base model:    {}", b.display())));
    }
    if let Some(ref pm) = plan.patched_model_path {
        info.push(Line::from(Span::styled(
            format!("Patched model: {}", pm.display()),
            Style::default().fg(Color::Green),
        )));
    }
    if let Some(ref ad) = plan.adapter_path {
        info.push(Line::from(Span::styled(
            format!("LoRA adapter:  {}", ad.display()),
            Style::default().fg(Color::Green),
        )));
    }
    info.push(Line::from(""));
    info.push(Line::from(Span::styled(
        "Recommended command:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    info.push(Line::from(Span::styled(
        plan.display_command.as_str(),
        Style::default().fg(Color::Cyan),
    )));
    for note in &plan.notes {
        info.push(Line::from(""));
        info.push(Line::from(Span::styled(
            format!("Note: {note}"),
            Style::default().fg(Color::Yellow),
        )));
    }

    let summary = Paragraph::new(info).wrap(Wrap { trim: true }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Resolved Paths & Command"),
    );
    f.render_widget(summary, rows[1]);
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0usize;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}
