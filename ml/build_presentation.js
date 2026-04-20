// PH 551 Final Project Presentation — Ian Lapinski
// KF Parameter Prediction from Stability Curves
// 15-min talk, Mon 04/20 9am

const path = require("path");
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";           // 10" x 5.625"
pres.author = "Ian Lapinski";
pres.title  = "KF Parameter Prediction from Stability Curves";

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "presentation.pptx");

// --- Palette: Midnight Executive ---
const NAVY    = "1E2761";
const ICE     = "CADCFC";
const ACCENT  = "E94F64";
const INK     = "1A1A2E";
const MUTED   = "6B7280";
const WHITE   = "FFFFFF";
const LIGHT   = "F5F7FB";

const FONT_H = "Georgia";
const FONT_B = "Calibri";

function addFooter(slide, slideNum, totalSlides) {
  slide.addText("PH 551 Final Project  •  Ian Lapinski  •  Apr 2026", {
    x: 0.4, y: 5.30, w: 6, h: 0.25,
    fontSize: 9, fontFace: FONT_B, color: MUTED, margin: 0
  });
  slide.addText(`${slideNum} / ${totalSlides}`, {
    x: 8.8, y: 5.30, w: 0.8, h: 0.25,
    fontSize: 9, fontFace: FONT_B, color: MUTED, align: "right", margin: 0
  });
}

function addTitleBar(slide, titleText, kickerText) {
  slide.addText(titleText, {
    x: 0.4, y: 0.28, w: 9.2, h: 0.55,
    fontSize: 26, fontFace: FONT_H, bold: true, color: NAVY, margin: 0
  });
  if (kickerText) {
    slide.addText(kickerText, {
      x: 0.4, y: 0.80, w: 9.2, h: 0.30,
      fontSize: 13, fontFace: FONT_B, italic: true, color: ACCENT, margin: 0
    });
  }
}

const TOTAL = 15;

// ============================================================
// Slide 1 — Title
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: ACCENT }, line: { color: ACCENT }
  });

  s.addText("KF Parameter Prediction", {
    x: 0.8, y: 1.2, w: 8.5, h: 0.9,
    fontSize: 42, fontFace: FONT_H, bold: true, color: WHITE, margin: 0
  });
  s.addText("from Frequency-Stability Curves", {
    x: 0.8, y: 2.05, w: 8.5, h: 0.6,
    fontSize: 28, fontFace: FONT_H, italic: true, color: ICE, margin: 0
  });

  s.addText("Instant initial guesses for Kalman-filter tuning on atomic clock data.", {
    x: 0.8, y: 2.85, w: 8.5, h: 0.4,
    fontSize: 15, fontFace: FONT_B, color: ICE, margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 4.10, w: 0.04, h: 0.80, fill: { color: ACCENT }, line: { color: ACCENT }
  });
  s.addText([
    { text: "Ian Lapinski", options: { bold: true, fontSize: 16, color: WHITE, breakLine: true } },
    { text: "PH 551 Final Project  •  April 20, 2026", options: { fontSize: 12, color: ICE } }
  ], { x: 1.0, y: 4.08, w: 6, h: 0.90, fontFace: FONT_B, margin: 0 });
}

// ============================================================
// Slide 2 — Problem & Research Question
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "The problem", "Kalman filters need good q-values. Getting them is slow.");

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.35, w: 4.5, h: 3.65,
    fill: { color: LIGHT }, line: { color: ICE, width: 1 }
  });
  s.addText("Use case", {
    x: 0.6, y: 1.45, w: 4.2, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });
  s.addText([
    { text: "\u201CI have clock phase measurements.", options: { italic: true, breakLine: true } },
    { text: " I don't know the q values.", options: { italic: true, breakLine: true } },
    { text: " I want to optimize the Kalman filter", options: { italic: true, breakLine: true } },
    { text: " \u2014 give me a quick initial guess.\u201D", options: { italic: true } }
  ], { x: 0.6, y: 1.85, w: 4.2, h: 1.3, fontSize: 14, fontFace: FONT_B, color: INK, margin: 0 });

  s.addText("Current options:", {
    x: 0.6, y: 3.20, w: 4.2, h: 0.30,
    fontSize: 13, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });
  s.addText([
    { text: "NLL optimization \u2014 multi-minute, highly sensitive to warm start", options: { bullet: true, breakLine: true } },
    { text: "ALS \u2014 well-conditioned but iterative, needs seed", options: { bullet: true, breakLine: true } },
    { text: "Manual MHDEV fit \u2014 requires expert \u03C4-region selection", options: { bullet: true } }
  ], { x: 0.6, y: 3.55, w: 4.2, h: 1.4, fontSize: 11, fontFace: FONT_B, color: INK, paraSpaceAfter: 3, margin: 0 });

  s.addText("Research question", {
    x: 5.2, y: 1.45, w: 4.4, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });
  s.addText("Can supervised learning on frequency-stability features predict KF q-parameters accurately enough to seed or replace iterative optimizers?", {
    x: 5.2, y: 1.85, w: 4.4, h: 1.3, fontSize: 15, fontFace: FONT_B, color: INK, margin: 0
  });

  s.addText("Sub-question", {
    x: 5.2, y: 3.20, w: 4.4, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });
  s.addText("Which stability statistic is most informative \u2014 ADEV, MDEV, HDEV, or MHDEV?", {
    x: 5.2, y: 3.55, w: 4.4, h: 0.7, fontSize: 13, fontFace: FONT_B, italic: true, color: INK, margin: 0
  });

  s.addText("Targets: log\u2081\u2080(q_wpm), log\u2081\u2080(q_wfm), log\u2081\u2080(q_rwfm)", {
    x: 5.2, y: 4.35, w: 4.4, h: 0.4,
    fontSize: 12, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });

  addFooter(s, 2, TOTAL);
}

// ============================================================
// Slide 3 — Why this matters
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Why it matters", "Clocks are everywhere \u2014 but their noise models aren't automatic.");

  const cols = [
    {
      title: "Atomic clocks drift",
      body: "GPS, telecom, defense timing, scientific instrumentation. Operators get phase data; they want stability predictions.",
      icon: "\u23F1"
    },
    {
      title: "KF tuning is slow",
      body: "NLL optimization: 10\u201360 s per sample, sensitive to warm start. Lab workflow bottleneck.",
      icon: "\u2699"
    },
    {
      title: "ML can do it in ms",
      body: "Trained regressor: sub-millisecond inference on raw features. Quantified uncertainty included.",
      icon: "\u2726"
    }
  ];
  cols.forEach((c, i) => {
    const x = 0.4 + i * 3.12;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.45, w: 2.92, h: 3.50,
      fill: { color: LIGHT }, line: { color: ICE, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.45, w: 2.92, h: 0.08, fill: { color: ACCENT }, line: { color: ACCENT }
    });
    s.addText(c.icon, {
      x: x + 0.15, y: 1.70, w: 0.6, h: 0.6,
      fontSize: 30, fontFace: FONT_H, color: NAVY, margin: 0
    });
    s.addText(c.title, {
      x: x + 0.15, y: 2.35, w: 2.6, h: 0.45,
      fontSize: 16, fontFace: FONT_H, bold: true, color: NAVY, margin: 0
    });
    s.addText(c.body, {
      x: x + 0.15, y: 2.85, w: 2.6, h: 2.0,
      fontSize: 12, fontFace: FONT_B, color: INK, margin: 0
    });
  });

  addFooter(s, 3, TOTAL);
}

// ============================================================
// Slide 4 — Dataset pipeline
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Dataset pipeline", "10,000 synthetic samples anchored to real GMR6000 Rb data.");

  const BLOCK_W = 1.70;
  const BLOCK_GAP = 0.18;
  const blocks = [
    { label: "Real GMR6000\nRb phase",        sub: "35 days, 1 PPS",          color: MUTED },
    { label: "h-range\nsampling",             sub: "Kasdin 1992",             color: NAVY },
    { label: "Composite\nnoise gen.",         sub: "N = 524,288 (~6 days)",   color: NAVY },
    { label: "196-feature\nextraction",       sub: "ADEV, MDEV, HDEV, MHDEV", color: NAVY },
    { label: "ALS label\noptimizer",          sub: "noise-free vs truth",     color: ACCENT }
  ];
  const ROW_W = blocks.length * BLOCK_W + (blocks.length - 1) * BLOCK_GAP;
  const ROW_START = (10 - ROW_W) / 2;
  blocks.forEach((b, i) => {
    const bx = ROW_START + i * (BLOCK_W + BLOCK_GAP);
    const fillC = b.color === ACCENT ? ACCENT : (b.color === MUTED ? ICE : NAVY);
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 1.55, w: BLOCK_W, h: 1.1,
      fill: { color: fillC }, line: { color: fillC }
    });
    s.addText(b.label, {
      x: bx + 0.05, y: 1.60, w: BLOCK_W - 0.10, h: 0.60,
      fontSize: 10, fontFace: FONT_B, bold: true,
      color: b.color === MUTED ? NAVY : WHITE,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(b.sub, {
      x: bx + 0.05, y: 2.20, w: BLOCK_W - 0.10, h: 0.40,
      fontSize: 8, fontFace: FONT_B, italic: true,
      color: b.color === MUTED ? MUTED : ICE,
      align: "center", valign: "middle", margin: 0
    });
    if (i < blocks.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: bx + BLOCK_W, y: 2.10, w: BLOCK_GAP, h: 0,
        line: { color: NAVY, width: 2, endArrowType: "triangle" }
      });
    }
  });

  s.addText("Key design decisions", {
    x: 0.4, y: 3.05, w: 9.2, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });

  const decisions = [
    { title: "Physically grounded",     body: "h_\u03B1 ranges anchored to measured GMR6000 Rb (not arbitrary noise blends)." },
    { title: "4+1 noise types",         body: "WPM + WFM + FFM + RWFM always; FPM in ~30% of samples." },
    { title: "Targets are log\u2081\u2080(q)", body: "Log-space equalizes multi-decade dynamic range; KF-native." },
    { title: "ALS-optimized labels",    body: "Probed on 10 draws: ALS matches truth to \u226410\u207B\u00B3 dec; NLL drifted \u00B18 dec." }
  ];
  decisions.forEach((d, i) => {
    const x = 0.4 + i * 2.35;
    s.addText(d.title, {
      x, y: 3.50, w: 2.25, h: 0.35,
      fontSize: 12, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
    });
    s.addText(d.body, {
      x, y: 3.85, w: 2.25, h: 1.05,
      fontSize: 10, fontFace: FONT_B, color: INK, margin: 0
    });
  });

  addFooter(s, 4, TOTAL);
}

// ============================================================
// Slide 5 — Features
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Feature engineering", "196 features per sample from 20 log-spaced \u03C4 values.");

  s.addText("Feature decomposition", {
    x: 0.4, y: 1.40, w: 4.5, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });

  s.addTable([
    [
      { text: "Group",       options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
      { text: "Count",       options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
      { text: "Description", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } }
    ],
    ["Raw \u03C3 values",    "80",  "4 stats \u00D7 20 \u03C4: log\u2081\u2080 of ADEV, MDEV, HDEV, MHDEV"],
    ["Log-log slopes", "76",  "4 stats \u00D7 19 adjacent-\u03C4 slopes (noise-type signature)"],
    ["Variance ratios","40",  "MVAR/AVAR (20) + MHVAR/HVAR (20) \u2014 bias-type detectors"],
    [{ text: "Total", options: { bold: true, fill: { color: ICE } } },
     { text: "196",   options: { bold: true, fill: { color: ICE } } },
     { text: "Per-sample feature vector", options: { bold: true, fill: { color: ICE } } }]
  ], {
    x: 0.4, y: 1.78, w: 4.8, colW: [1.35, 0.70, 2.75],
    rowH: 0.42,
    border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 10, fontFace: FONT_B, color: INK
  });

  s.addText("Why these four statistics?", {
    x: 5.50, y: 1.40, w: 4.1, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });

  s.addText([
    { text: "ADEV ", options: { bold: true, color: NAVY } },
    { text: "\u2014 standard reference; baseline comparison\n", options: { breakLine: true } },
    { text: "MDEV ", options: { bold: true, color: NAVY } },
    { text: "\u2014 distinguishes WPM vs FPM (slope \u22123 vs \u22122)\n", options: { breakLine: true } },
    { text: "HDEV ", options: { bold: true, color: NAVY } },
    { text: "\u2014 Hadamard; rejects frequency drift\n", options: { breakLine: true } },
    { text: "MHDEV ", options: { bold: true, color: NAVY } },
    { text: "\u2014 modified Hadamard; best for long \u03C4 on drifting clocks", options: {} }
  ], { x: 5.50, y: 1.80, w: 4.1, h: 1.8, fontSize: 11, fontFace: FONT_B, color: INK, margin: 0 });

  s.addText("Secondary research hypothesis", {
    x: 5.50, y: 3.70, w: 4.1, h: 0.3,
    fontSize: 12, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });
  s.addText("Feature-importance analysis will reveal which statistic carries the most information about KF noise parameters.", {
    x: 5.50, y: 4.02, w: 4.1, h: 0.9,
    fontSize: 11, fontFace: FONT_B, italic: true, color: INK, margin: 0
  });

  addFooter(s, 5, TOTAL);
}

// ============================================================
// Slide 6 — EDA
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Data exploration", "Target distributions, example \u03C3(\u03C4) curves, feature\u2013target correlations.");

  s.addImage({
    path: path.join(FIG, "eda_distributions.png"),
    x: 0.35, y: 1.30, w: 6.0, h: 3.7
  });

  s.addText("What the data shows", {
    x: 6.55, y: 1.30, w: 3.1, h: 0.35,
    fontSize: 13, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });

  s.addText([
    { text: "Targets span ~6 log decades (q_rwfm widest; q_wpm tightest)", options: { bullet: true, breakLine: true } },
    { text: "Short-\u03C4 features (top row) correlate strongest with q_wpm", options: { bullet: true, breakLine: true } },
    { text: "Long-\u03C4 features correlate with q_rwfm (log\u2081\u2080 q ~ \u221230)", options: { bullet: true, breakLine: true } },
    { text: "FPM-present samples (30%) have steeper short-\u03C4 slope", options: { bullet: true, breakLine: true } },
    { text: "NaN rate 0%; no outliers at |z|>5 \u2014 clean dataset", options: { bullet: true } }
  ], { x: 6.55, y: 1.70, w: 3.1, h: 3.3, fontSize: 10, fontFace: FONT_B, color: INK, paraSpaceAfter: 6, margin: 0 });

  addFooter(s, 6, TOTAL);
}

// ============================================================
// Slide 7 — Statistical baselines
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Statistical baselines", "Setting the bar: what does 'good' look like?");

  s.addText("Baseline predictors", {
    x: 0.4, y: 1.40, w: 4.5, h: 0.35,
    fontSize: 14, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });

  s.addTable([
    [
      { text: "Method",                  options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
      { text: "RMSE (log\u2081\u2080 q)",options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
      { text: "Notes",                   options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } }
    ],
    ["Naive (train mean)",     "0.87 / 0.87 / 1.73", "Ignores features (R\u00B2 \u2248 0)"],
    ["Linear regression",      "(not run)",          "Monotonic but nonlinear signal"],
    ["MHDEV 3-region fit",     "see slide 13",       "Real-data only, requires expert \u03C4-regions"],
    [
      { text: "RF (ours)", options: { bold: true, fill: { color: ICE }, color: ACCENT } },
      { text: "0.07 / 0.09 / 0.78", options: { bold: true, fill: { color: ICE }, color: ACCENT } },
      { text: "~12\u00D7 better than naive", options: { bold: true, fill: { color: ICE } } }
    ]
  ], {
    x: 0.4, y: 1.80, w: 4.6, colW: [1.65, 1.45, 1.50],
    rowH: 0.42,
    border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 10, fontFace: FONT_B, color: INK
  });

  s.addImage({
    path: path.join(FIG, "eda_corr_feature_target.png"),
    x: 5.40, y: 1.40, w: 4.3, h: 1.9
  });

  s.addText([
    { text: "Correlation pattern above: ", options: { bold: true, color: NAVY } },
    { text: "short-\u03C4 features drive q_wpm; long-\u03C4 features drive q_rwfm. ", options: {} },
    { text: "Monotonic but nonlinear \u2014 justifies tree ensembles over linear regression.",
      options: { italic: true, color: MUTED } }
  ], { x: 5.40, y: 3.45, w: 4.3, h: 1.5, fontSize: 11, fontFace: FONT_B, color: INK, margin: 0 });

  addFooter(s, 7, TOTAL);
}

// ============================================================
// Slide 8 — Methodology
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Methodology", "Random Forest + XGBoost, multi-output regression.");

  const models = [
    {
      x: 0.4, title: "Random Forest", color: NAVY,
      pts: [
        "Bagged ensemble of decision trees",
        "Built-in UQ via tree-variance (forestci)",
        "Invariant to monotone feature transforms",
        "Robust baseline for tabular data"
      ]
    },
    {
      x: 5.05, title: "XGBoost (Gradient Boosted)", color: ACCENT,
      pts: [
        "Sequential boosting \u2014 corrects errors",
        "Typically ~1\u20133\u00D7 better RMSE than RF on tabular",
        "UQ via quantile regression (\u03B1=0.05, 0.95)",
        "Best-in-class for multi-output regression"
      ]
    }
  ];
  models.forEach(m => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: m.x, y: 1.40, w: 4.55, h: 2.70,
      fill: { color: LIGHT }, line: { color: m.color, width: 1.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: m.x, y: 1.40, w: 4.55, h: 0.10, fill: { color: m.color }, line: { color: m.color }
    });
    s.addText(m.title, {
      x: m.x + 0.20, y: 1.55, w: 4.15, h: 0.40,
      fontSize: 16, fontFace: FONT_H, bold: true, color: m.color, margin: 0
    });
    const bullets = m.pts.map((p, i) => ({
      text: p,
      options: { bullet: true, breakLine: i < m.pts.length - 1 }
    }));
    s.addText(bullets, {
      x: m.x + 0.20, y: 2.05, w: 4.15, h: 2.00,
      fontSize: 11, fontFace: FONT_B, color: INK, paraSpaceAfter: 4, margin: 0
    });
  });

  s.addText("Why both?", {
    x: 0.4, y: 4.25, w: 2.0, h: 0.4,
    fontSize: 13, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });
  s.addText("Head-to-head comparison on identical features. Different bias/variance tradeoffs \u2192 diagnostic of whether signal is smooth (boosting wins) or rough (RF wins). UQ mechanisms differ \u2014 cross-check coverage.", {
    x: 2.4, y: 4.25, w: 7.2, h: 0.75,
    fontSize: 11, fontFace: FONT_B, italic: true, color: INK, margin: 0
  });

  addFooter(s, 8, TOTAL);
}

// ============================================================
// Slide 9 — Hyperparameter Tuning
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Hyperparameter tuning", "5-fold GridSearchCV with explicit justification per parameter.");

  s.addText("Random Forest grid", {
    x: 0.4, y: 1.35, w: 4.5, h: 0.32,
    fontSize: 13, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });
  s.addTable([
    [{ text: "Parameter", options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 10 } },
     { text: "Grid",      options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 10 } },
     { text: "Best",      options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 10 } }],
    ["n_estimators",     "{200, 500, 1000}", "1000"],
    ["max_depth",        "{None, 20, 30}",   "20"],
    ["min_samples_leaf", "{3, 5, 10}",       "3"],
    ["max_features",     "{sqrt, 0.5}",      "0.5"]
  ], {
    x: 0.4, y: 1.72, w: 4.5, colW: [1.55, 1.70, 1.25],
    rowH: 0.38, border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 10, fontFace: FONT_B, color: INK
  });

  s.addText("XGBoost grid", {
    x: 5.10, y: 1.35, w: 4.5, h: 0.32,
    fontSize: 13, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });
  s.addTable([
    [{ text: "Parameter", options: { bold: true, fill: { color: ACCENT }, color: WHITE, fontSize: 10 } },
     { text: "Grid",      options: { bold: true, fill: { color: ACCENT }, color: WHITE, fontSize: 10 } },
     { text: "Best",      options: { bold: true, fill: { color: ACCENT }, color: WHITE, fontSize: 10 } }],
    ["n_estimators",  "{200, 500}",        "500"],
    ["learning_rate", "{0.01, 0.05, 0.1}", "0.01"],
    ["max_depth",     "{4, 6, 8}",         "6"],
    ["subsample",     "{0.8, 1.0}",        "0.8"]
  ], {
    x: 5.10, y: 1.72, w: 4.5, colW: [1.55, 1.70, 1.25],
    rowH: 0.38, border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 10, fontFace: FONT_B, color: INK
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.95, w: 9.2, h: 1.00,
    fill: { color: LIGHT }, line: { color: ICE, width: 1 }
  });
  s.addText("CV protocol", {
    x: 0.6, y: 4.05, w: 2.2, h: 0.35,
    fontSize: 12, fontFace: FONT_B, bold: true, color: NAVY, margin: 0
  });
  s.addText([
    { text: "5-fold CV  \u2022  270 RF fits + 180 XGB fits  \u2022  ", options: {} },
    { text: "scoring = ", options: {} },
    { text: "neg_mean_squared_error", options: { bold: true } },
    { text: " in log\u2081\u2080-q space  \u2022  ", options: {} },
    { text: "seed = 42 for reproducibility", options: {} }
  ], { x: 0.6, y: 4.35, w: 9.0, h: 0.55,
      fontSize: 11, fontFace: FONT_B, color: INK, margin: 0 });

  addFooter(s, 9, TOTAL);
}

// ============================================================
// Slide 10 — Results: RMSE/R²/MAE
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Results", "Per-target metrics + model comparison.");

  s.addTable([
    [
      { text: "Target", options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 11 } },
      { text: "Model",  options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 11 } },
      { text: "RMSE",   options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 11 } },
      { text: "R\u00B2", options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 11 } },
      { text: "MAE",    options: { bold: true, fill: { color: NAVY }, color: WHITE, fontSize: 11 } }
    ],
    ["q_wpm",  "Naive", "0.874", "0.000", "0.759"],
    ["q_wpm",  "RF",    "0.069", "0.994", "0.052"],
    ["q_wpm",  { text: "XGB", options: { bold: true, color: ACCENT } },
               { text: "0.007", options: { bold: true, color: ACCENT } },
               { text: "0.99993", options: { bold: true, color: ACCENT } },
               { text: "0.006",  options: { bold: true, color: ACCENT } }],
    ["q_wfm",  "Naive", "0.873", "0.000", "0.758"],
    ["q_wfm",  "RF",    "0.090", "0.989", "0.065"],
    ["q_wfm",  { text: "XGB", options: { bold: true, color: ACCENT } },
               { text: "0.053", options: { bold: true, color: ACCENT } },
               { text: "0.996", options: { bold: true, color: ACCENT } },
               { text: "0.030", options: { bold: true, color: ACCENT } }],
    ["q_rwfm", "Naive", "1.732", "0.000", "1.494"],
    ["q_rwfm", "RF",    "0.784", "0.795", "0.551"],
    ["q_rwfm", { text: "XGB", options: { bold: true, color: ACCENT } },
               { text: "0.736", options: { bold: true, color: ACCENT } },
               { text: "0.819", options: { bold: true, color: ACCENT } },
               { text: "0.510", options: { bold: true, color: ACCENT } }]
  ], {
    x: 0.4, y: 1.35, w: 4.9, colW: [1.0, 0.8, 1.1, 1.1, 0.9],
    rowH: 0.32, border: { pt: 0.5, color: "CCCCCC" },
    fontSize: 10, fontFace: FONT_B, color: INK
  });

  s.addImage({
    path: path.join(FIG, "model_comparison_rmse.png"),
    x: 5.50, y: 1.35, w: 4.1, h: 2.9
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.30, w: 9.2, h: 0.65,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText([
    { text: "Headline: ", options: { bold: true, color: ACCENT } },
    { text: "XGB \u2192 <0.05-dec RMSE on q_wpm & q_wfm (R\u00B2 \u2265 0.996); ", options: { color: WHITE } },
    { text: "q_rwfm R\u00B2 = 0.82 ", options: { bold: true, color: WHITE } },
    { text: "(vs 0.65 before switching labels from NLL \u2192 ALS).", options: { color: WHITE } }
  ], { x: 0.6, y: 4.38, w: 8.8, h: 0.5, fontSize: 12, fontFace: FONT_B, margin: 0 });

  addFooter(s, 10, TOTAL);
}

// ============================================================
// Slide 11 — Uncertainty Quantification
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Uncertainty quantification", "Grad-student requirement \u2014 the model knows when it doesn't know.");

  s.addImage({
    path: path.join(FIG, "rf_uq.png"),
    x: 0.3, y: 1.30, w: 5.8, h: 3.6
  });

  s.addText("Two complementary approaches", {
    x: 6.25, y: 1.30, w: 3.4, h: 0.35,
    fontSize: 13, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });

  s.addText([
    { text: "Random Forest \u2014 forestci", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Infinitesimal jackknife on tree predictions", options: { breakLine: true } },
    { text: "\u2192 analytic variance", options: { italic: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "XGBoost \u2014 quantile regression", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Two extra models at \u03B1=0.05, \u03B1=0.95", options: { breakLine: true } },
    { text: "\u2192 non-parametric prediction intervals", options: { italic: true } }
  ], { x: 6.25, y: 1.70, w: 3.4, h: 2.2, fontSize: 10, fontFace: FONT_B, color: INK, margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.25, y: 4.00, w: 3.4, h: 1.00,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });
  s.addText([
    { text: "Empirical 90% PI coverage (XGB):\n", options: { fontSize: 11, color: WHITE, bold: true, breakLine: true } },
    { text: "q_wpm 84%   q_wfm 81%   q_rwfm 81%", options: { fontSize: 13, color: WHITE, bold: true } }
  ], { x: 6.35, y: 4.08, w: 3.2, h: 0.85, fontFace: FONT_B, margin: 0, align: "center" });

  addFooter(s, 11, TOTAL);
}

// ============================================================
// Slide 12 — Feature Importance
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Which statistic carries the signal?", "Headline: feature-importance answer to the sub-question.");

  s.addImage({
    path: path.join(FIG, "importance_by_statistic.png"),
    x: 0.30, y: 1.30, w: 5.9, h: 3.6
  });

  s.addText("What the data says", {
    x: 6.35, y: 1.30, w: 3.3, h: 0.35,
    fontSize: 13, fontFace: FONT_B, bold: true, color: ACCENT, margin: 0
  });

  s.addText([
    { text: "MDEV dominates ", options: { bold: true, color: NAVY } },
    { text: "(\u03A3 importance \u2248 0.59)\n", options: { breakLine: true } },
    { text: "MHDEV 5\u00D7 less important (\u03A3 \u2248 0.12)\n", options: { breakLine: true } },
    { text: "ADEV / HDEV contribute the remaining ~0.29\n", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Slopes carry more signal than raw \u03C3 values \u2014", options: { breakLine: true } },
    { text: "consistent with noise-type diagnostics driving q.", options: { italic: true, color: MUTED } }
  ], { x: 6.35, y: 1.72, w: 3.3, h: 2.7, fontSize: 11, fontFace: FONT_B, color: INK, margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.35, y: 4.30, w: 3.3, h: 0.65,
    fill: { color: LIGHT }, line: { color: ACCENT, width: 1 }
  });
  s.addText([
    { text: "Contribution: ", options: { bold: true, color: ACCENT } },
    { text: "MDEV is the most informative stability statistic for KF tuning.", options: { color: INK } }
  ], { x: 6.50, y: 4.35, w: 3.0, h: 0.55, fontSize: 10, fontFace: FONT_B, italic: true, margin: 0 });

  addFooter(s, 12, TOTAL);
}

// ============================================================
// Slide 13 — Real-data validation
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Real-data validation", "GMR6000 Rb phase \u2192 ML prediction \u2192 ADEV overlay on 4 windows.");

  s.addImage({
    path: path.join(FIG, "real_data_method_comparison.png"),
    x: 0.15, y: 1.30, w: 9.7, h: 3.35
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.75, w: 9.2, h: 0.52,
    fill: { color: LIGHT }, line: { color: ACCENT, width: 1 }
  });
  s.addText([
    { text: "Headline: ", options: { bold: true, color: ACCENT } },
    { text: "RF & XGB \u03C3(\u03C4) predictions track measured ADEV across all 4 windows \u2014 ", options: { color: INK } },
    { text: "model generalizes from synthetic to real hardware.", options: { bold: true, color: NAVY } }
  ], { x: 0.6, y: 4.80, w: 9.0, h: 0.45, fontSize: 11, fontFace: FONT_B, margin: 0 });

  addFooter(s, 13, TOTAL);
}

// ============================================================
// Slide 14 — Takeaways + Limitations + Future Work
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitleBar(s, "Takeaways & limitations", "");

  const cols = [
    {
      x: 0.4, title: "What worked", color: NAVY,
      pts: [
        "ALS labels \u2192 noise-free targets (key finding; lifted q_rwfm R\u00B2 from 0.65 \u2192 0.82)",
        "XGB \u2192 sub-0.05-dec RMSE on q_wpm & q_wfm",
        "forestci + quantile UQ \u2192 80\u201384% empirical coverage of nominal 90% PIs",
        "Feature importance \u2192 MDEV quantitatively most informative (5\u00D7 MHDEV)"
      ]
    },
    {
      x: 3.53, title: "Limitations", color: ACCENT,
      pts: [
        "Trained on Rb GMR6000-class h-ranges only",
        "3-state KF model lacks flicker-FM \u2014 limits long-horizon holdover",
        "Single real-data record (6k27feb) for cross-check",
        "XGB 90% PI under-covers by 6\u201310 pp (conservative regularization)"
      ]
    },
    {
      x: 6.66, title: "Future work", color: NAVY,
      pts: [
        "Extend to OCXO, HSO, cesium training ranges",
        "4-state model with explicit FFM for sub-ns long-term holdover",
        "End-to-end NN on raw phase (current: hand-crafted features)",
        "Active learning on where the model is least confident"
      ]
    }
  ];
  cols.forEach(c => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 1.20, w: 3.00, h: 3.85,
      fill: { color: LIGHT }, line: { color: c.color, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: c.x, y: 1.20, w: 3.00, h: 0.08, fill: { color: c.color }, line: { color: c.color }
    });
    s.addText(c.title, {
      x: c.x + 0.18, y: 1.35, w: 2.7, h: 0.40,
      fontSize: 15, fontFace: FONT_H, bold: true, color: c.color, margin: 0
    });
    const bullets = c.pts.map((p, i) => ({
      text: p,
      options: { bullet: true, breakLine: i < c.pts.length - 1 }
    }));
    s.addText(bullets, {
      x: c.x + 0.18, y: 1.85, w: 2.7, h: 3.10,
      fontSize: 10, fontFace: FONT_B, color: INK, paraSpaceAfter: 6, margin: 0
    });
  });

  addFooter(s, 14, TOTAL);
}

// ============================================================
// Slide 15 — Thank you / Q&A
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: ACCENT }, line: { color: ACCENT }
  });

  s.addText("Thank you", {
    x: 0.8, y: 1.40, w: 8.5, h: 0.90,
    fontSize: 48, fontFace: FONT_H, bold: true, color: WHITE, margin: 0
  });
  s.addText("Questions?", {
    x: 0.8, y: 2.25, w: 8.5, h: 0.60,
    fontSize: 28, fontFace: FONT_H, italic: true, color: ICE, margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 3.70, w: 0.04, h: 1.20, fill: { color: ACCENT }, line: { color: ACCENT }
  });
  s.addText([
    { text: "Code", options: { bold: true, fontSize: 13, color: WHITE, breakLine: true } },
    { text: "github.com/ianlap/SigmaTau", options: { fontSize: 12, color: ICE, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Ian Lapinski  \u2022  PH 551 Final Project  \u2022  April 2026",
      options: { fontSize: 11, color: ICE, italic: true } }
  ], { x: 1.0, y: 3.68, w: 8.2, h: 1.25, fontFace: FONT_B, margin: 0 });
}

pres.writeFile({ fileName: OUT })
  .then(f => console.log("Wrote:", f));
