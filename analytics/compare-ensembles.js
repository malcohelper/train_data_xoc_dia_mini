#!/usr/bin/env node
/**
 * Compare multiple ensemble configurations side-by-side on the same rounds.
 *
 * Run:
 *   node analytics/compare-ensembles.js
 *   node analytics/compare-ensembles.js --rounds rounds --burn-in 800
 *   node analytics/compare-ensembles.js --burn-in 800 --out analytics/compare-report.json
 *
 * Reports for each config:
 *   - Ensemble dynamic + static type/exact %
 *   - Per-algo type % (so you can see which algos are pulling weight)
 *   - Selective betting curve (threshold -> type %, coverage)
 *
 * The default config set tests the "slim ensemble" hypothesis: that
 * dropping low-accuracy predictors lifts the dynamic ensemble closer to
 * the best individual member (markov ~62% in current data).
 */
"use strict";

const fs = require("fs");
const path = require("path");
const engine = require(path.join(__dirname, "prediction-engine.js"));

const DICE_TO_RED = {
  "4_red": 4, "3r_1w": 3, "2w_2r": 2, "3w_1r": 1, "4_white": 0,
};

function loadRounds(roundsDir) {
  const skipped = [];
  if (!fs.existsSync(roundsDir)) {
    throw new Error(`Không thấy thư mục: ${roundsDir}`);
  }
  const names = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json")).sort();
  const merged = [];
  for (const name of names) {
    let raw;
    try { raw = JSON.parse(fs.readFileSync(path.join(roundsDir, name), "utf8")); }
    catch (e) { skipped.push({ file: name, reason: e.message }); continue; }
    if (!raw || raw.dice_result === undefined) {
      skipped.push({ file: name, reason: "thiếu dice_result" }); continue;
    }
    const red = DICE_TO_RED[raw.dice_result];
    if (red === undefined) {
      skipped.push({ file: name, reason: `dice_result lạ: ${raw.dice_result}` }); continue;
    }
    let time = null;
    if (raw.started_at) { const d = new Date(raw.started_at); if (!isNaN(d.getTime())) time = d; }
    if (!time) {
      const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(raw.round_id || "");
      if (m) time = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    }
    if (!time) { skipped.push({ file: name, reason: "không xác định time" }); continue; }
    let finalisedAt = null;
    if (raw.finalised_at) { const fd = new Date(raw.finalised_at); if (!isNaN(fd.getTime())) finalisedAt = fd; }
    let durationSec = null;
    if (finalisedAt) { const ds = Math.round((finalisedAt.getTime() - time.getTime()) / 1000); if (ds >= 0 && ds < 7200) durationSec = ds; }
    merged.push({
      red, type: red % 2 === 0 ? "chan" : "le", time,
      round_id: raw.round_id || "", finalisedAt, durationSec,
      percent: raw.percent && typeof raw.percent === "object" ? { ...raw.percent } : null,
      bets: raw.bets && typeof raw.bets === "object" ? JSON.parse(JSON.stringify(raw.bets)) : null,
    });
  }
  return { merged, skipped, fileCount: names.length };
}

function runConfig(history, burnIn, cfg, historyWindow) {
  // Apply ensemble overrides
  const DE = engine.DYNAMIC_ENSEMBLE;
  const prev = {};
  for (const k of Object.keys(cfg.de || {})) {
    prev[k] = DE[k];
    DE[k] = cfg.de[k];
  }
  engine.setActivePredictors(cfg.predictors || engine.ALL_ALGO_IDS);

  const normalized = engine.normalizeHistory(history);
  const algoIds = engine.ALGO_IDS.slice();
  const perAlgo = {};
  algoIds.forEach((id) => { perAlgo[id] = { exact: 0, type: 0, n: 0 }; });
  let dynExact = 0, dynType = 0, statExact = 0, statType = 0, n = 0;
  const stepsLite = [];

  for (let t = burnIn; t < normalized.length; t++) {
    // historyWindow caps how many past rounds the predictors see for the
    // walk-forward step at index t. null/0 means "unbounded" (use full
    // history before t). With a positive cap, we slide a window of size N
    // ending at t-1 — this is the non-stationarity-aware mode.
    const startIdx = historyWindow > 0 ? Math.max(0, t - historyWindow) : 0;
    const past = normalized.slice(startIdx, t);
    const actual = normalized[t];
    const currentRound = { percent: actual.percent || null, bets: actual.bets || null };
    const ensDyn = engine.ensemblePredict(past, { currentRound });
    const ensStat = engine.ensemblePredict(past, { dynamic: false, currentRound });
    n++;
    if (ensDyn.predictedRed === actual.red) dynExact++;
    if (ensDyn.predictedRed % 2 === actual.red % 2) dynType++;
    if (ensStat.predictedRed === actual.red) statExact++;
    if (ensStat.predictedRed % 2 === actual.red % 2) statType++;
    ensDyn.algorithms.forEach((a) => {
      const st = perAlgo[a.id];
      if (!st) return;
      st.n++;
      if (a.predictedRed === actual.red) st.exact++;
      if (a.predictedRed % 2 === actual.red % 2) st.type++;
    });
    stepsLite.push({
      betConf: ensDyn.betConfidence || 0,
      parityOk: ensDyn.predictedRed % 2 === actual.red % 2,
      exactOk: ensDyn.predictedRed === actual.red,
    });
  }

  // Selective curve
  const thresholds = [0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60];
  const selective = thresholds.map((th) => {
    let bet = 0, ty = 0, ex = 0;
    for (const s of stepsLite) {
      if (s.betConf >= th) { bet++; if (s.parityOk) ty++; if (s.exactOk) ex++; }
    }
    return {
      threshold: th,
      bet, coverage: stepsLite.length ? Math.round((10000 * bet) / stepsLite.length) / 100 : 0,
      typePct: bet ? Math.round((10000 * ty) / bet) / 100 : 0,
      exactPct: bet ? Math.round((10000 * ex) / bet) / 100 : 0,
    };
  });

  // Restore globals
  for (const k of Object.keys(prev)) { DE[k] = prev[k]; }
  engine.resetActivePredictors();

  const pct = (x, d) => d > 0 ? Math.round((10000 * x) / d) / 100 : 0;
  const byAlgo = {};
  algoIds.forEach((id) => {
    const s = perAlgo[id];
    byAlgo[id] = { ...s, typePct: pct(s.type, s.n), exactPct: pct(s.exact, s.n) };
  });

  return {
    name: cfg.name,
    predictors: algoIds,
    de: cfg.de || {},
    n,
    dynamic: { type: dynType, exact: dynExact, typePct: pct(dynType, n), exactPct: pct(dynExact, n) },
    static: { type: statType, exact: statExact, typePct: pct(statType, n), exactPct: pct(statExact, n) },
    byAlgo,
    selective,
  };
}

function parseArgs(argv) {
  const out = {
    roundsDir: path.join(__dirname, "..", "rounds"),
    burnIn: 1,
    outPath: null,
    only: null,
    historyWindow: 0, // 0 = unbounded, >0 = sliding window of N rounds
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--rounds" && argv[i + 1]) out.roundsDir = path.resolve(argv[++i]);
    else if (a === "--burn-in" && argv[i + 1]) out.burnIn = Math.max(0, parseInt(argv[++i], 10) || 1);
    else if (a === "--out" && argv[i + 1]) out.outPath = path.resolve(argv[++i]);
    else if (a === "--only" && argv[i + 1]) out.only = argv[++i].split(",").map((s) => s.trim());
    else if (a === "--history-window" && argv[i + 1]) {
      const v = parseInt(argv[++i], 10);
      out.historyWindow = Number.isFinite(v) && v > 0 ? v : 0;
    }
  }
  return out;
}

function buildConfigs() {
  // ALL_ALGO_IDS = ['pattern','streak','markov','markov2','time','regression',
  //                 'cauPattern','balance','crowd','parityRepeat','bayesian']

  return [
    { name: "baseline-11algo (current main)",
      predictors: null /* all 11 */,
      de: {} },

    // Drop predictors below or near random (50%) on prior backtest
    { name: "slim-6: pattern + markov + markov2 + time + crowd + parityRepeat",
      predictors: ["pattern","markov","markov2","time","crowd","parityRepeat"],
      de: { TOP_K: 6 } },

    // Even slimmer: keep only top-3 individual algos
    { name: "lean-3: pattern + markov + markov2",
      predictors: ["pattern","markov","markov2"],
      de: { TOP_K: 3 } },

    // Pure markov solo (best individual on prior backtest)
    { name: "markov-solo",
      predictors: ["markov"],
      de: { TOP_K: 1 } },

    // Markov + pattern duo
    { name: "duo: markov + pattern",
      predictors: ["markov","pattern"],
      de: { TOP_K: 2 } },

    // Slim with HIT_BLEND_EXACT bumped up — favour algos that nail exact red
    { name: "slim-6 + blend=0.85",
      predictors: ["pattern","markov","markov2","time","crowd","parityRepeat"],
      de: { TOP_K: 6, HIT_BLEND_EXACT: 0.85 } },

    // Slim with stronger BETA — let high-confidence algos dominate more
    { name: "slim-6 + beta=2.5",
      predictors: ["pattern","markov","markov2","time","crowd","parityRepeat"],
      de: { TOP_K: 6, BETA: 2.5 } },

    // Slim with both above
    { name: "slim-6 + beta=2.5 + blend=0.85",
      predictors: ["pattern","markov","markov2","time","crowd","parityRepeat"],
      de: { TOP_K: 6, BETA: 2.5, HIT_BLEND_EXACT: 0.85 } },

    // === Markov-decay sweep ===
    // λ controls how fast old transitions are forgotten. λ=1.0 = no
    // decay (full history equally weighted, "stationary" assumption);
    // λ=0.98 = ~34-round half-life (default, mild decay); λ=0.95 = ~14-
    // round half-life (aggressive recency bias).
    //
    // Empirically: if game is non-stationary, lower λ should win. If
    // game is stationary, λ=1.0 should win. Use these to test which
    // regime the dataset lives in.
    { name: "duo + markov λ=1.00 (no decay)",
      predictors: ["markov","pattern"],
      de: { TOP_K: 2, MARKOV_DECAY: 1.0 } },
    { name: "duo + markov λ=0.98 (default)",
      predictors: ["markov","pattern"],
      de: { TOP_K: 2, MARKOV_DECAY: 0.98 } },
    { name: "duo + markov λ=0.95 (aggressive)",
      predictors: ["markov","pattern"],
      de: { TOP_K: 2, MARKOV_DECAY: 0.95 } },

    // NOTE: Tuned variants (duo-tuned, slim-6-tuned, lean-3-tuned) were
    // removed after verification showed they only deliver +1pp over their
    // default-hyperparam siblings when measured through engine.ensemblePredict.
    // The 3-7pp gap previously claimed came from analytics/grid-search.js
    // using a hand-rolled evalDynamic that diverges from the engine
    // (no regime boost, no parity weight boost, different hit-rate window
    // source). See analytics/grid-search.js preamble for details.
  ];
}

function fmtPct(x) { return `${x.toFixed(2).padStart(5)}%`; }

function main() {
  const opts = parseArgs(process.argv);
  const { merged, skipped, fileCount } = loadRounds(opts.roundsDir);
  console.log(`Loaded ${merged.length}/${fileCount} rounds (${skipped.length} skipped) from ${opts.roundsDir}`);
  if (merged.length <= opts.burnIn) {
    console.error(`Cần > burnIn (${opts.burnIn}) phiên hợp lệ; hiện ${merged.length}.`);
    process.exit(1);
  }
  const hwLabel = opts.historyWindow > 0 ? `${opts.historyWindow} rounds (sliding)` : "all (unbounded)";
  console.log(`Burn-in: ${opts.burnIn}  ·  Test rounds: ${merged.length - opts.burnIn}  ·  History window: ${hwLabel}\n`);

  const allCfgs = buildConfigs();
  const cfgs = opts.only ? allCfgs.filter((c) => opts.only.some((o) => c.name.includes(o))) : allCfgs;

  const results = [];
  for (const cfg of cfgs) {
    const t0 = Date.now();
    process.stdout.write(`[${results.length + 1}/${cfgs.length}] ${cfg.name} ...`);
    const r = runConfig(merged, opts.burnIn, cfg, opts.historyWindow);
    r.elapsedSec = ((Date.now() - t0) / 1000).toFixed(1);
    results.push(r);
    process.stdout.write(`  done in ${r.elapsedSec}s\n`);
  }

  console.log("\n=== ENSEMBLE COMPARISON ===\n");
  console.log("  " + "Config".padEnd(46) + "DynCL%   DynVi%   StatCL%  StatVi%   n");
  console.log("  " + "─".repeat(86));
  for (const r of results) {
    console.log(
      "  " + r.name.padEnd(46) +
      fmtPct(r.dynamic.typePct) + "  " +
      fmtPct(r.dynamic.exactPct) + "  " +
      fmtPct(r.static.typePct) + "  " +
      fmtPct(r.static.exactPct) + "   " + r.n
    );
  }

  console.log("\n=== SELECTIVE BETTING (Dynamic) — type % @ threshold ===\n");
  const thrs = results[0].selective.map((s) => s.threshold);
  console.log("  " + "Config".padEnd(46) + thrs.map((t) => `t${(t * 100).toFixed(0)}`.padStart(7)).join(""));
  console.log("  " + "─".repeat(46 + thrs.length * 7));
  for (const r of results) {
    console.log(
      "  " + r.name.padEnd(46) +
      r.selective.map((s) => `${s.typePct.toFixed(1)}%`.padStart(7)).join("")
    );
  }
  console.log("\n  (coverage at threshold 0.50 → " + results.map((r) => `${r.name.split(":")[0]}=${r.selective.find((s) => s.threshold === 0.5).coverage}%`).join(", ") + ")");

  console.log("\n=== PER-ALGO TYPE % (within each config) ===\n");
  for (const r of results) {
    const items = Object.entries(r.byAlgo).map(([id, s]) => `${id}:${s.typePct.toFixed(1)}%`);
    console.log(`  ${r.name}\n    ${items.join("  ")}`);
  }

  // Highlight best config (max ensemble dynamic type%, then max exact%)
  const best = results.slice().sort((a, b) =>
    b.dynamic.typePct - a.dynamic.typePct ||
    b.dynamic.exactPct - a.dynamic.exactPct
  )[0];
  console.log(`\n★ Best ensemble dynamic CL%: ${best.name} → ${best.dynamic.typePct}% type, ${best.dynamic.exactPct}% exact\n`);

  if (opts.outPath) {
    fs.writeFileSync(opts.outPath, JSON.stringify({
      generatedAt: new Date().toISOString(),
      roundsDir: opts.roundsDir,
      burnIn: opts.burnIn,
      historyWindow: opts.historyWindow || null,
      mergedCount: merged.length,
      results,
      bestByDynamicTypePct: best.name,
    }, null, 2));
    console.log(`Wrote ${opts.outPath}`);
  }
}

if (require.main === module) main();
