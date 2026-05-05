#!/usr/bin/env node
/**
 * Fast grid search: pre-computes all per-algo predictions, then searches
 * over ensemble weighting parameters without re-running predictors.
 *
 * ⚠️  IMPORTANT — divergence from production engine
 *
 * `evalDynamic` below is a hand-rolled re-implementation of ensemble
 * weighting; it does NOT call engine.ensemblePredict. It diverges from
 * the production engine in several ways:
 *
 *   1. Hit-rate window: this script accumulates hit-rates from the test
 *      set as it walks forward. The engine recomputes hit-rates by
 *      re-running predictors over the last N rounds of FULL HISTORY
 *      (training + already-tested).
 *   2. No regime boost: engine applies REGIME_BOOST (parityRepeat ×
 *      1.15 in streaky regime, etc.); this script does not.
 *   3. No parity weight boost: engine applies
 *      `min(1.25, max(0.8, parityConf/conf))` to per-algo weights when
 *      computing parity tally; this script uses raw weights.
 *   4. Single hit-window vs multi-timeframe: engine blends short + long
 *      windows (`mergeMultiTimeframeHitRates`); this script uses a
 *      single window.
 *
 * Empirically this script OVERSTATES dynamic CL% by 3-7pp vs
 * compare-ensembles.js (which uses engine.ensemblePredict). USE THIS
 * SCRIPT FOR DIRECTION ONLY (which subset / window / α-β region looks
 * promising) — verify absolute numbers with compare-ensembles.js.
 *
 * Usage:
 *   node analytics/grid-search.js
 *   node analytics/grid-search.js --rounds path/to/rounds
 *   node analytics/grid-search.js --burn-in 800 --predictors markov,pattern
 */
"use strict";

const fs = require("fs");
const path = require("path");
const engine = require(path.join(__dirname, "prediction-engine.js"));

const DICE_TO_RED = {
  "4_red": 4, "3r_1w": 3, "2w_2r": 2, "3w_1r": 1, "4_white": 0,
};

function loadRounds(roundsDir) {
  const names = fs.readdirSync(roundsDir).filter((f) => f.endsWith(".json")).sort();
  const merged = [];
  for (const name of names) {
    try {
      const raw = JSON.parse(fs.readFileSync(path.join(roundsDir, name), "utf8"));
      if (!raw || !raw.dice_result) continue;
      const red = DICE_TO_RED[raw.dice_result];
      if (red === undefined) continue;
      let time = null;
      if (raw.started_at) { const d = new Date(raw.started_at); if (!isNaN(d.getTime())) time = d; }
      if (!time) {
        const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(raw.round_id || "");
        if (m) time = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
      }
      if (!time) continue;
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
    } catch (_) { /* skip */ }
  }
  return merged;
}

/**
 * Pre-compute all predictor outputs for every walk-forward step.
 * Returns: predictions[t] = { algos: [{id, predictedRed, confidence, parityConfidence?}], actual: {red, type} }
 */
function precomputePredictions(history, burnIn) {
  const h = engine.normalizeHistory(history);
  const predictions = [];
  for (let t = burnIn; t < h.length; t++) {
    const past = h.slice(0, t);
    const actual = h[t];
    const currentRound = { percent: actual.percent || null, bets: actual.bets || null };
    const algos = engine.PREDICTORS.map((fn) => fn(past, currentRound));
    predictions.push({ algos, actual: { red: actual.red, type: actual.type } });
    if (predictions.length % 50 === 0) {
      process.stdout.write(`\r  Pre-computing predictions... ${predictions.length}/${h.length - burnIn}`);
    }
  }
  process.stdout.write("\r" + " ".repeat(60) + "\r");
  return predictions;
}

/**
 * Evaluate a set of ensemble parameters against pre-computed predictions.
 * Uses static mode (confidence-only) which is much faster and often better.
 */
function evalStatic(predictions, beta, topK) {
  let typeOk = 0, exactOk = 0;
  const n = predictions.length;
  const algoIds = engine.ALGO_IDS;

  for (let i = 0; i < n; i++) {
    const { algos, actual } = predictions[i];

    const weights = algos.map((a) => {
      const c = Math.max(0.05, Math.min(1, a.confidence));
      return Math.pow(c, beta);
    });

    let finalWeights = weights;
    if (topK > 0 && topK < weights.length) {
      finalWeights = weights.slice();
      const ranked = finalWeights.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w);
      for (let j = topK; j < ranked.length; j++) finalWeights[ranked[j].j] = 0;
    }

    const tally = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    algos.forEach((a, j) => { tally[a.predictedRed] += finalWeights[j]; });

    let predictedRed = 2, maxW = -1;
    for (let r = 0; r <= 4; r++) {
      if (tally[r] > maxW) { maxW = tally[r]; predictedRed = r; }
    }

    let wChan = 0, wLe = 0;
    algos.forEach((a, j) => {
      const pc = a.parityConfidence != null ? a.parityConfidence : a.confidence;
      const pw = Math.pow(Math.max(0.05, Math.min(1, pc)), beta);
      const fw = topK > 0 && topK < weights.length ? (finalWeights[j] > 0 ? pw : 0) : pw;
      if (a.predictedRed % 2 === 0) wChan += fw;
      else wLe += fw;
    });
    const predictedParity = wChan >= wLe ? "chan" : "le";
    const targetMod = predictedParity === "chan" ? 0 : 1;
    if (predictedRed % 2 !== targetMod) {
      let bestR = targetMod === 0 ? 2 : 3, bestW = -1;
      for (let r = 0; r <= 4; r++) {
        if (r % 2 === targetMod && tally[r] > bestW) { bestW = tally[r]; bestR = r; }
      }
      predictedRed = bestR;
    }

    if (predictedRed === actual.red) exactOk++;
    if (predictedRed % 2 === actual.red % 2) typeOk++;
  }

  return { clPct: (typeOk / n) * 100, viPct: (exactOk / n) * 100, n };
}

/**
 * Evaluate with dynamic mode (alpha blending of confidence + hit rates).
 */
function evalDynamic(predictions, alpha, beta, topK, hitWindow, blendExact) {
  let typeOk = 0, exactOk = 0;
  const n = predictions.length;
  const algoCount = engine.ALGO_IDS.length;
  const baseline = 0.15;
  const shrink = 2;

  const hitStats = {};
  engine.ALGO_IDS.forEach((id) => { hitStats[id] = { exactHits: 0, parityHits: 0, history: [] }; });

  for (let i = 0; i < n; i++) {
    const { algos, actual } = predictions[i];

    const hitRates = {};
    engine.ALGO_IDS.forEach((id, idx) => {
      const st = hitStats[id];
      const window = st.history.slice(-hitWindow);
      const total = window.length;
      let eHits = 0, pHits = 0;
      for (const h of window) { if (h.exact) eHits++; if (h.parity) pHits++; }
      hitRates[id] = {
        exactRate: total > 0 ? (eHits + shrink * baseline) / (total + shrink) : baseline,
        parityRate: total > 0 ? (pHits + shrink * baseline) / (total + shrink) : baseline,
        total,
      };
    });

    const weights = algos.map((a) => {
      const c = Math.max(0.05, Math.min(1, a.confidence));
      const hi = hitRates[a.id];
      const hExact = hi ? hi.exactRate : baseline;
      const hParity = hi ? hi.parityRate : baseline;
      const Hblend = Math.max(0.05, Math.min(1, blendExact * hExact + (1 - blendExact) * hParity));
      const dynamicScore = Math.max(0.05, Math.min(1, alpha * c + (1 - alpha) * Hblend));
      return Math.pow(dynamicScore, beta);
    });

    let finalWeights = weights.slice();
    if (topK > 0 && topK < finalWeights.length) {
      const ranked = finalWeights.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w);
      for (let j = topK; j < ranked.length; j++) finalWeights[ranked[j].j] = 0;
    }

    const tally = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    algos.forEach((a, j) => { tally[a.predictedRed] += finalWeights[j]; });

    let predictedRed = 2, maxW = -1;
    for (let r = 0; r <= 4; r++) {
      if (tally[r] > maxW) { maxW = tally[r]; predictedRed = r; }
    }

    let wChan = 0, wLe = 0;
    algos.forEach((a, j) => {
      if (a.predictedRed % 2 === 0) wChan += finalWeights[j];
      else wLe += finalWeights[j];
    });
    const predictedParity = wChan >= wLe ? "chan" : "le";
    const targetMod = predictedParity === "chan" ? 0 : 1;
    if (predictedRed % 2 !== targetMod) {
      let bestR = targetMod === 0 ? 2 : 3, bestW = -1;
      for (let r = 0; r <= 4; r++) {
        if (r % 2 === targetMod && tally[r] > bestW) { bestW = tally[r]; bestR = r; }
      }
      predictedRed = bestR;
    }

    if (predictedRed === actual.red) exactOk++;
    if (predictedRed % 2 === actual.red % 2) typeOk++;

    algos.forEach((a) => {
      hitStats[a.id].history.push({
        exact: a.predictedRed === actual.red,
        parity: a.predictedRed % 2 === actual.red % 2,
      });
    });
  }

  return { clPct: (typeOk / n) * 100, viPct: (exactOk / n) * 100, n };
}

function main() {
  let roundsDir = path.join(__dirname, "..", "rounds");
  let topN = 15;
  let predictors = null;
  let burnIn = 1;
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === "--rounds" && process.argv[i + 1]) roundsDir = path.resolve(process.argv[++i]);
    if (process.argv[i] === "--top" && process.argv[i + 1]) topN = parseInt(process.argv[++i], 10);
    if (process.argv[i] === "--burn-in" && process.argv[i + 1]) {
      const v = parseInt(process.argv[++i], 10);
      if (Number.isFinite(v) && v >= 1) burnIn = v;
    }
    if (process.argv[i] === "--predictors" && process.argv[i + 1]) {
      predictors = process.argv[++i].split(",").map((s) => s.trim()).filter(Boolean);
    }
  }

  if (predictors && predictors.length) {
    try { engine.setActivePredictors(predictors); }
    catch (e) { console.error(`--predictors error: ${e.message}`); process.exit(2); }
  }

  const history = loadRounds(roundsDir);
  console.log(`Loaded ${history.length} rounds from ${roundsDir}`);
  console.log(`Active predictors (${engine.ALGO_IDS.length}): ${engine.ALGO_IDS.join(", ")}`);
  if (burnIn >= history.length) {
    console.error(`--burn-in ${burnIn} >= ${history.length} rounds, nothing to test`);
    process.exit(2);
  }
  const testN = history.length - burnIn;
  console.log(`Burn-in: ${burnIn}  ·  Test rounds: ${testN}`);
  console.log("Pre-computing predictions...");
  const predictions = precomputePredictions(history, burnIn);
  console.log(`Pre-computed ${predictions.length} prediction steps\n`);

  console.log("=== STATIC MODE SEARCH (no hit rates) ===\n");
  const staticBetas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6];
  const staticTopKs = [5, 6, 7, 8, 9, 10, 11];
  const staticResults = [];

  for (const beta of staticBetas) {
    for (const topK of staticTopKs) {
      const r = evalStatic(predictions, beta, topK);
      staticResults.push({ mode: "static", beta, topK, ...r });
    }
  }
  staticResults.sort((a, b) => b.clPct - a.clPct || b.viPct - a.viPct);

  console.log("  #  BETA  TOP_K   CL%     Vi%");
  console.log("  " + "─".repeat(35));
  for (let i = 0; i < Math.min(topN, staticResults.length); i++) {
    const r = staticResults[i];
    console.log(`  ${String(i + 1).padStart(2)}.  ${r.beta.toFixed(1)}   ${String(r.topK).padStart(2)}    ${r.clPct.toFixed(2)}%  ${r.viPct.toFixed(2)}%`);
  }

  console.log("\n=== DYNAMIC MODE SEARCH ===\n");
  const dynAlphas = [0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0];
  const dynBetas = [1.5, 2, 2.5, 3, 4, 5];
  const dynTopKs = [6, 7, 8, 9, 11];
  const dynHitWindows = [15, 20, 30, 40];
  const dynBlends = [0.3, 0.55, 0.7, 0.85];
  const dynTotal = dynAlphas.length * dynBetas.length * dynTopKs.length * dynHitWindows.length * dynBlends.length;
  console.log(`  Searching ${dynTotal} dynamic combinations...`);

  const dynResults = [];
  let count = 0;
  for (const alpha of dynAlphas) {
    for (const beta of dynBetas) {
      for (const topK of dynTopKs) {
        for (const hitW of dynHitWindows) {
          for (const blend of dynBlends) {
            count++;
            if (count % 200 === 0) process.stdout.write(`\r  ${count}/${dynTotal} (${Math.round(100 * count / dynTotal)}%)`);
            const r = evalDynamic(predictions, alpha, beta, topK, hitW, blend);
            dynResults.push({ mode: "dynamic", alpha, beta, topK, hitW, blend, ...r });
          }
        }
      }
    }
  }
  process.stdout.write("\r" + " ".repeat(50) + "\r");
  dynResults.sort((a, b) => b.clPct - a.clPct || b.viPct - a.viPct);

  console.log("\n  #  ALPHA  BETA  TOP_K  HIT_W  BLEND   CL%     Vi%");
  console.log("  " + "─".repeat(55));
  for (let i = 0; i < Math.min(topN, dynResults.length); i++) {
    const r = dynResults[i];
    console.log(
      `  ${String(i + 1).padStart(2)}.  ${r.alpha.toFixed(2)}  ${r.beta.toFixed(1)}   ${String(r.topK).padStart(2)}     ${String(r.hitW).padStart(2)}    ${r.blend.toFixed(2)}   ${r.clPct.toFixed(2)}%  ${r.viPct.toFixed(2)}%`
    );
  }

  const bestStatic = staticResults[0];
  const bestDynamic = dynResults[0];
  const overall = bestStatic.clPct >= bestDynamic.clPct ? bestStatic : bestDynamic;

  console.log(`\n=== OVERALL BEST ===`);
  console.log(`  Mode: ${overall.mode}`);
  if (overall.mode === "static") {
    console.log(`  BETA: ${overall.beta}, TOP_K: ${overall.topK}`);
  } else {
    console.log(`  ALPHA: ${overall.alpha}, BETA: ${overall.beta}, TOP_K: ${overall.topK}, HIT_WINDOW: ${overall.hitW}, BLEND: ${overall.blend}`);
  }
  console.log(`  CL: ${overall.clPct.toFixed(2)}% | Vi: ${overall.viPct.toFixed(2)}%\n`);

  const outPath = path.join(__dirname, "grid-search-results.json");
  fs.writeFileSync(outPath, JSON.stringify({
    bestStatic, bestDynamic, overall,
    topStatic: staticResults.slice(0, 20),
    topDynamic: dynResults.slice(0, 20),
  }, null, 2));
  console.log(`Results saved to ${outPath}`);
}

main();
