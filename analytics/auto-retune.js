#!/usr/bin/env node
/**
 * Auto-retune monitor: evaluates the production preset list against the
 * most-recent N rounds and recommends whether to switch the default
 * preset.
 *
 * Designed to be run periodically (e.g. nightly via cron, or every time
 * `rounds/` grows by ~500 entries) once total round volume gets large.
 *
 * Usage:
 *   node analytics/auto-retune.js
 *   node analytics/auto-retune.js --rounds rounds --recent 500
 *   node analytics/auto-retune.js --recent 500 --burn-in 100 --out analytics/auto-retune-report.json
 *
 * Defaults:
 *   --rounds   ../rounds
 *   --recent   500          (use last N rounds for the rolling window)
 *   --burn-in  100          (within those N, first --burn-in rounds train,
 *                            the rest are tested walk-forward)
 *   --margin   2.0          (min CL pp gap to recommend changing default)
 *   --current  duo          (current default preset; recommendation only
 *                            triggers if best preset beats this by --margin)
 *
 * Output:
 *   - Stdout: human-readable per-preset CL%, selective-bet metrics,
 *     coverage at threshold 0.50, and explicit recommendation line.
 *   - --out <path>: structured JSON snapshot, suitable for committing to
 *     git or feeding into a dashboard.
 *
 * Recommendation logic:
 *   - Find best preset by all-rounds CL% on the recent window.
 *   - If (best.clPct - current.clPct) >= margin, recommend switch.
 *   - Otherwise, "no change". Reports gap size either way so you can
 *     audit the call.
 *
 * This script does NOT mutate code or config files. It only emits
 * recommendations. Apply by editing app-with-prediction.js DEFAULT_PRESET
 * by hand (or via a separate apply step).
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
  const merged = [];
  if (!fs.existsSync(roundsDir)) {
    console.error(`Rounds dir not found: ${roundsDir}`);
    process.exit(2);
  }
  const names = fs.readdirSync(roundsDir).filter((n) => n.endsWith(".json"));
  for (const name of names) {
    const fp = path.join(roundsDir, name);
    try {
      const j = JSON.parse(fs.readFileSync(fp, "utf-8"));
      const finalised = j.finalised_at || j.finalisedAt;
      // Production rounds (realtime_capture.py) write `dice_result`; tolerate
      // legacy `dice_class` aliases too.
      const dice = j.dice_result || j.diceResult || j.dice_class || j.diceClass;
      if (!finalised || !DICE_TO_RED.hasOwnProperty(dice)) {
        skipped.push(name);
        continue;
      }
      merged.push({
        red: DICE_TO_RED[dice],
        finalised_at: finalised,
        durationSec: j.duration_sec,
        bets: j.bets || null,
        percent: j.percent || null,
      });
    } catch (e) {
      skipped.push(name);
    }
  }
  merged.sort((a, b) => a.finalised_at.localeCompare(b.finalised_at));
  return { merged, skipped, fileCount: names.length };
}

// Production preset list — must stay in sync with app-with-prediction.js
// PREDICTOR_PRESETS. Tuned variants intentionally excluded.
const PRESETS = [
  { id: "duo", predictors: ["markov", "pattern"], de: { TOP_K: 2 } },
  { id: "markov-solo", predictors: ["markov"], de: { TOP_K: 1 } },
  { id: "lean-3", predictors: ["pattern", "markov", "markov2"], de: { TOP_K: 3 } },
  { id: "slim-6", predictors: ["pattern", "markov", "markov2", "time", "crowd", "parityRepeat"], de: { TOP_K: 6 } },
  { id: "baseline-11", predictors: null, de: {} },
];

function runPreset(history, burnIn, preset) {
  const DE = engine.DYNAMIC_ENSEMBLE;
  const prev = {};
  for (const k of Object.keys(preset.de || {})) {
    prev[k] = DE[k];
    DE[k] = preset.de[k];
  }
  engine.setActivePredictors(preset.predictors || engine.ALL_ALGO_IDS);

  const normalized = engine.normalizeHistory(history);
  let dynExact = 0, dynType = 0, n = 0;
  const stepsLite = [];

  for (let t = burnIn; t < normalized.length; t++) {
    const past = normalized.slice(0, t);
    const actual = normalized[t];
    const currentRound = { percent: actual.percent || null, bets: actual.bets || null };
    const ens = engine.ensemblePredict(past, { currentRound });
    n++;
    if (ens.predictedRed === actual.red) dynExact++;
    if (ens.predictedRed % 2 === actual.red % 2) dynType++;
    stepsLite.push({
      betConf: ens.betConfidence || 0,
      parityOk: ens.predictedRed % 2 === actual.red % 2,
      exactOk: ens.predictedRed === actual.red,
    });
  }

  // Selective at t=0.50 — the operational threshold the UI uses by default.
  let bet = 0, ty = 0;
  for (const s of stepsLite) {
    if (s.betConf >= 0.5) { bet++; if (s.parityOk) ty++; }
  }
  const sel50TypePct = bet ? (ty / bet) * 100 : 0;
  const sel50Coverage = stepsLite.length ? (bet / stepsLite.length) * 100 : 0;

  for (const k of Object.keys(prev)) DE[k] = prev[k];
  engine.resetActivePredictors();

  return {
    id: preset.id,
    n,
    clPct: n ? (dynType / n) * 100 : 0,
    viPct: n ? (dynExact / n) * 100 : 0,
    sel50TypePct,
    sel50Coverage,
  };
}

function parseArgs(argv) {
  const out = {
    roundsDir: path.join(__dirname, "..", "rounds"),
    recent: 500,
    burnIn: 100,
    margin: 2.0,
    current: "duo",
    outPath: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--rounds" && argv[i + 1]) out.roundsDir = path.resolve(argv[++i]);
    else if (a === "--recent" && argv[i + 1]) out.recent = Math.max(50, parseInt(argv[++i], 10) || 500);
    else if (a === "--burn-in" && argv[i + 1]) out.burnIn = Math.max(10, parseInt(argv[++i], 10) || 100);
    else if (a === "--margin" && argv[i + 1]) out.margin = Math.max(0, parseFloat(argv[++i]) || 2.0);
    else if (a === "--current" && argv[i + 1]) out.current = argv[++i];
    else if (a === "--out" && argv[i + 1]) out.outPath = path.resolve(argv[++i]);
  }
  return out;
}

function fmtPct(x) { return `${x.toFixed(2).padStart(5)}%`; }

function main() {
  const opts = parseArgs(process.argv);
  const { merged, skipped, fileCount } = loadRounds(opts.roundsDir);
  console.log(`Loaded ${merged.length}/${fileCount} rounds (${skipped.length} skipped) from ${opts.roundsDir}`);

  if (merged.length < opts.recent + 10) {
    console.error(`Need at least --recent + 10 rounds; got ${merged.length}, recent=${opts.recent}.`);
    process.exit(1);
  }
  if (opts.burnIn >= opts.recent) {
    console.error(`--burn-in (${opts.burnIn}) must be < --recent (${opts.recent}).`);
    process.exit(1);
  }

  const window = merged.slice(-opts.recent);
  const testN = window.length - opts.burnIn;
  console.log(`Recent window: ${window.length} rounds  ·  Burn-in: ${opts.burnIn}  ·  Test rounds: ${testN}\n`);

  console.log("=== AUTO-RETUNE EVALUATION ===\n");
  console.log("  Preset           CL%      Vi%     Sel@0.50  Cov@0.50  n");
  console.log("  ──────────────────────────────────────────────────────────");
  const results = [];
  for (const preset of PRESETS) {
    const r = runPreset(window, opts.burnIn, preset);
    results.push(r);
    console.log(
      `  ${preset.id.padEnd(15)} ${fmtPct(r.clPct)} ${fmtPct(r.viPct)} ` +
      ` ${fmtPct(r.sel50TypePct)}  ${fmtPct(r.sel50Coverage)}  ${r.n}`,
    );
  }

  const best = results.reduce((a, b) => (b.clPct > a.clPct ? b : a), results[0]);
  const current = results.find((r) => r.id === opts.current);

  console.log("\n=== RECOMMENDATION ===\n");
  console.log(`  Current default preset : ${opts.current} → ${current ? fmtPct(current.clPct) : "n/a"}`);
  console.log(`  Best preset (by CL%)   : ${best.id} → ${fmtPct(best.clPct)}`);

  let recommendation;
  let gap;
  if (!current) {
    recommendation = "current_unknown";
    gap = null;
    console.log(`  ⚠ Current preset id "${opts.current}" not in PRESETS list. No comparison.`);
  } else {
    gap = best.clPct - current.clPct;
    if (best.id === opts.current) {
      recommendation = "no_change_already_best";
      console.log(`  ✓ Current preset is already best. No change recommended.`);
    } else if (gap < opts.margin) {
      recommendation = "no_change_within_margin";
      console.log(`  ◯ Best preset beats current by only ${gap.toFixed(2)}pp (< margin ${opts.margin}pp). Within noise; no change.`);
    } else {
      recommendation = "switch";
      console.log(`  ★ Best preset beats current by ${gap.toFixed(2)}pp (>= margin ${opts.margin}pp). RECOMMEND SWITCH to "${best.id}".`);
    }
  }

  if (opts.outPath) {
    const report = {
      generated_at: new Date().toISOString(),
      rounds_total: merged.length,
      window_size: window.length,
      burn_in: opts.burnIn,
      test_n: testN,
      margin_pp: opts.margin,
      current_preset: opts.current,
      best_preset: best.id,
      gap_pp: gap,
      recommendation,
      results,
    };
    fs.writeFileSync(opts.outPath, JSON.stringify(report, null, 2));
    console.log(`\n  Report written to ${opts.outPath}`);
  }
}

main();
