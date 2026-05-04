#!/usr/bin/env node
/**
 * Walk-forward backtest trên toàn bộ file rounds/*.json (Node local).
 *
 * Cách hoạt động (walk-forward):
 *   Round 1: không dự đoán (chưa có dữ liệu)
 *   Round 2: dùng [1] để dự đoán → so kết quả round 2
 *   Round 3: dùng [1,2] để dự đoán → so kết quả round 3
 *   ...
 *   Round N: dùng [1..N-1] để dự đoán → so kết quả round N
 *
 * Chạy từ repo:
 *   node analytics/run-backtest-rounds.js
 *   node analytics/run-backtest-rounds.js --burn-in 1
 *   node analytics/run-backtest-rounds.js --burn-in 1 --detail
 *   node analytics/run-backtest-rounds.js --out analytics/backtest-report.json --burn-in 1
 *   node analytics/run-backtest-rounds.js --beta 4
 *   node analytics/run-backtest-rounds.js --hit-blend-exact 0.55
 */
"use strict";

const fs = require("fs");
const path = require("path");

const engine = require(path.join(__dirname, "prediction-engine.js"));

const DICE_TO_RED = {
  "4_red": 4,
  "3r_1w": 3,
  "2w_2r": 2,
  "3w_1r": 1,
  "4_white": 0,
};

const RED_LABELS = ["4T", "3T1Đ", "2T2Đ", "3Đ1T", "4Đ"];

function roundIdToDate(rid) {
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(rid || "");
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
}

/** @param {any} round raw JSON một phiên */
function roundJsonToRoundItem(round) {
  if (!round || typeof round !== "object") return null;
  const red = DICE_TO_RED[round.dice_result];
  if (red === undefined) return null;
  let time = null;
  if (round.started_at) {
    const d = new Date(round.started_at);
    if (!isNaN(d.getTime())) time = d;
  }
  if (!time) time = roundIdToDate(round.round_id);
  if (!time) return null;
  let finalisedAt = null;
  if (round.finalised_at) {
    const fd = new Date(round.finalised_at);
    if (!isNaN(fd.getTime())) finalisedAt = fd;
  }
  let durationSec = null;
  if (finalisedAt && !isNaN(finalisedAt.getTime())) {
    const ds = Math.round((finalisedAt.getTime() - time.getTime()) / 1000);
    if (ds >= 0 && ds < 7200) durationSec = ds;
  }
  return {
    red,
    type: red % 2 === 0 ? "chan" : "le",
    time,
    round_id: round.round_id || "",
    finalisedAt,
    durationSec,
    percent:
      round.percent && typeof round.percent === "object"
        ? { ...round.percent }
        : null,
    bets:
      round.bets && typeof round.bets === "object"
        ? JSON.parse(JSON.stringify(round.bets))
        : null,
  };
}

function parseArgs(argv) {
  const DE = engine.DYNAMIC_ENSEMBLE;
  const out = {
    outPath: path.join(__dirname, "backtest-report.json"),
    burnIn: 1,
    roundsDir: path.join(__dirname, "..", "rounds"),
    hitBlendExact: DE.HIT_BLEND_EXACT,
    beta: DE.BETA,
    detail: false,
    predictors: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out" && argv[i + 1]) {
      out.outPath = path.resolve(argv[++i]);
    } else if (a === "--burn-in" && argv[i + 1]) {
      out.burnIn = Math.max(0, parseInt(argv[++i], 10) || 1);
    } else if (a === "--rounds" && argv[i + 1]) {
      out.roundsDir = path.resolve(argv[++i]);
    } else if (a === "--beta" && argv[i + 1]) {
      const b = parseFloat(argv[++i]);
      out.beta = Number.isFinite(b) ? Math.max(0.5, Math.min(6, b)) : 2;
    } else if (a === "--hit-blend-exact" && argv[i + 1]) {
      const g = parseFloat(argv[++i]);
      out.hitBlendExact = Number.isFinite(g)
        ? Math.max(0, Math.min(1, g))
        : 0;
    } else if (a === "--detail") {
      out.detail = true;
    } else if (a === "--predictors" && argv[i + 1]) {
      out.predictors = argv[++i]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    }
  }
  return out;
}

function loadMergedHistory(roundsDir) {
  const skipped = [];
  if (!fs.existsSync(roundsDir)) {
    throw new Error(`Không thấy thư mục: ${roundsDir}`);
  }
  const names = fs
    .readdirSync(roundsDir)
    .filter((f) => f.endsWith(".json"))
    .sort();
  const merged = [];
  for (const name of names) {
    const fp = path.join(roundsDir, name);
    let raw;
    try {
      raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    } catch (e) {
      skipped.push({ file: name, reason: `parse error: ${e.message}` });
      continue;
    }
    const item = roundJsonToRoundItem(raw);
    if (!item) {
      skipped.push({
        file: name,
        reason:
          !raw || !raw.dice_result
            ? "thiếu dice_result"
            : `dice_result không hợp lệ: ${raw.dice_result}`,
      });
      continue;
    }
    merged.push(item);
  }
  return { merged, skipped, fileNames: names };
}

/**
 * Walk-forward chi tiết: tại mỗi round t, dùng [0..t-1] dự đoán round t.
 * Trả về mảng chi tiết từng bước + thống kê tổng hợp.
 */
function walkForwardDetail(h, burnIn) {
  const normalized = engine.normalizeHistory(h);
  const steps = [];
  const agg = {
    dyn: { exact: 0, type: 0, n: 0 },
    stat: { exact: 0, type: 0, n: 0 },
  };
  const perAlgo = {};
  engine.ALGO_IDS.forEach((id) => {
    perAlgo[id] = { exact: 0, type: 0, n: 0 };
  });

  let streakExact = 0;
  let streakParity = 0;
  let maxStreakExact = 0;
  let maxStreakParity = 0;

  for (let t = burnIn; t < normalized.length; t++) {
    const past = normalized.slice(0, t);
    const actual = normalized[t];
    const currentRound = {
      percent: actual.percent || null,
      bets: actual.bets || null,
    };

    const ensDyn = engine.ensemblePredict(past, { currentRound });
    const ensStat = engine.ensemblePredict(past, { dynamic: false, currentRound });

    const exactOk = ensDyn.predictedRed === actual.red;
    const parityOk = ensDyn.predictedRed % 2 === actual.red % 2;
    const staticExactOk = ensStat.predictedRed === actual.red;
    const staticParityOk = ensStat.predictedRed % 2 === actual.red % 2;

    agg.dyn.n++;
    if (exactOk) agg.dyn.exact++;
    if (parityOk) agg.dyn.type++;
    agg.stat.n++;
    if (staticExactOk) agg.stat.exact++;
    if (staticParityOk) agg.stat.type++;

    if (exactOk) {
      streakExact++;
      if (streakExact > maxStreakExact) maxStreakExact = streakExact;
    } else {
      streakExact = 0;
    }
    if (parityOk) {
      streakParity++;
      if (streakParity > maxStreakParity) maxStreakParity = streakParity;
    } else {
      streakParity = 0;
    }

    const algoDetail = {};
    ensDyn.algorithms.forEach((a) => {
      const st = perAlgo[a.id];
      st.n++;
      const aExact = a.predictedRed === actual.red;
      const aParity = a.predictedRed % 2 === actual.red % 2;
      if (aExact) st.exact++;
      if (aParity) st.type++;
      algoDetail[a.id] = {
        pred: a.predictedRed,
        conf: Math.round(a.confidence * 100),
        exactOk: aExact,
        parityOk: aParity,
      };
    });

    steps.push({
      step: t - burnIn + 1,
      round_id: actual.round_id,
      historySize: t,
      predDyn: ensDyn.predictedRed,
      predDynParity: ensDyn.predictedParity,
      predDynConf: Math.round(ensDyn.confidence * 100),
      predDynConsensus: Math.round(ensDyn.consensus * 100),
      predStat: ensStat.predictedRed,
      actual: actual.red,
      actualParity: actual.type,
      exactOk,
      parityOk,
      staticExactOk,
      staticParityOk,
      shouldBet: ensDyn.shouldBet,
      betConfidence: Math.round((ensDyn.betConfidence || 0) * 100),
      regime: ensDyn.regime || "unknown",
      algoDetail,
    });

    if (steps.length % 100 === 0) {
      const pct = Math.round((100 * steps.length) / (normalized.length - burnIn));
      process.stdout.write(`\r  Đang chạy... ${steps.length}/${normalized.length - burnIn} (${pct}%)`);
    }
  }
  if (steps.length >= 100) process.stdout.write("\r" + " ".repeat(60) + "\r");

  const pct = (x, n) => (n > 0 ? Math.round((10000 * x) / n) / 100 : 0);

  const algoSummary = {};
  engine.ALGO_IDS.forEach((id) => {
    const s = perAlgo[id];
    algoSummary[id] = {
      exact: s.exact,
      type: s.type,
      n: s.n,
      exactPct: pct(s.exact, s.n),
      typePct: pct(s.type, s.n),
    };
  });

  return {
    totalSteps: steps.length,
    burnIn,
    summary: {
      dynamic: {
        ...agg.dyn,
        exactPct: pct(agg.dyn.exact, agg.dyn.n),
        typePct: pct(agg.dyn.type, agg.dyn.n),
      },
      static: {
        ...agg.stat,
        exactPct: pct(agg.stat.exact, agg.stat.n),
        typePct: pct(agg.stat.type, agg.stat.n),
      },
      maxStreakExact,
      maxStreakParity,
    },
    byAlgo: algoSummary,
    steps,
  };
}

function main() {
  const opts = parseArgs(process.argv);
  const { merged, skipped, fileNames } = loadMergedHistory(opts.roundsDir);

  if (opts.predictors && opts.predictors.length) {
    try {
      engine.setActivePredictors(opts.predictors);
    } catch (e) {
      console.error(`--predictors error: ${e.message}`);
      process.exit(2);
    }
  }

  console.log(`=== Backtest walk-forward (${engine.ALGO_IDS.length} thuật toán) ===\n`);
  console.log(`Thư mục:    ${opts.roundsDir}`);
  console.log(`File .json: ${fileNames.length} · Hợp lệ: ${merged.length} · Bỏ qua: ${skipped.length}`);
  console.log(`Burn-in:    ${opts.burnIn} (bỏ qua ${opts.burnIn} phiên đầu)`);
  console.log(`BETA=${opts.beta} · HIT_BLEND_EXACT=${opts.hitBlendExact}`);
  console.log(`Active:     ${engine.ALGO_IDS.join(", ")}`);
  if (skipped.length && skipped.length <= 15) {
    skipped.forEach((s) => console.log(`  - ${s.file}: ${s.reason}`));
  } else if (skipped.length) {
    console.log(`  (bỏ ${skipped.length} file; xem report.skipped)`);
  }
  console.log("");

  if (merged.length <= opts.burnIn) {
    console.error(
      `Cần > burnIn (${opts.burnIn}) phiên hợp lệ; hiện chỉ ${merged.length}.`,
    );
    process.exit(1);
  }

  const DE = engine.DYNAMIC_ENSEMBLE;
  const prevBlend = DE.HIT_BLEND_EXACT;
  const prevBeta = DE.BETA;

  DE.HIT_BLEND_EXACT = opts.hitBlendExact;
  DE.BETA = opts.beta;

  let wf;
  let baselines;
  const startMs = Date.now();
  try {
    console.log("Chạy walk-forward...");
    wf = walkForwardDetail(merged, opts.burnIn);
    baselines = engine.runBaselines(merged, { burnIn: opts.burnIn });
  } finally {
    DE.HIT_BLEND_EXACT = prevBlend;
    DE.BETA = prevBeta;
  }
  const elapsedSec = ((Date.now() - startMs) / 1000).toFixed(1);

  const d = wf.summary.dynamic;
  const s = wf.summary.static;
  console.log(`\nHoàn thành ${wf.totalSteps} bước trong ${elapsedSec}s\n`);

  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║           TỔNG THỂ (walk-forward)               ║");
  console.log("╠══════════════════════════════════════════════════╣");
  console.log(`║  Ensemble Động:  CL ${String(d.typePct).padStart(5)}%  (${d.type}/${d.n})`.padEnd(51) + "║");
  console.log(`║                  Vị ${String(d.exactPct).padStart(5)}%  (${d.exact}/${d.n})`.padEnd(51) + "║");
  console.log(`║  Ensemble Tĩnh:  CL ${String(s.typePct).padStart(5)}%  (${s.type}/${s.n})`.padEnd(51) + "║");
  console.log(`║                  Vị ${String(s.exactPct).padStart(5)}%  (${s.exact}/${s.n})`.padEnd(51) + "║");
  console.log("║" + "─".repeat(50) + "║");
  console.log(`║  Δ CL (Động − Tĩnh):  ${d.typePct >= s.typePct ? "+" : ""}${(d.typePct - s.typePct).toFixed(2)} pp`.padEnd(51) + "║");
  console.log(`║  Δ Vị (Động − Tĩnh):  ${d.exactPct >= s.exactPct ? "+" : ""}${(d.exactPct - s.exactPct).toFixed(2)} pp`.padEnd(51) + "║");
  console.log("║" + "─".repeat(50) + "║");
  console.log(`║  Chuỗi đúng CL dài nhất:  ${wf.summary.maxStreakParity} phiên`.padEnd(51) + "║");
  console.log(`║  Chuỗi đúng Vị dài nhất:  ${wf.summary.maxStreakExact} phiên`.padEnd(51) + "║");
  console.log("║" + "─".repeat(50) + "║");
  console.log(`║  Baselines: random CL ${baselines.random.typePct}% · repeat CL ${baselines.lastRepeats.typePct}%`.padEnd(51) + "║");
  console.log("╚══════════════════════════════════════════════════╝");

  console.log("\n--- Theo từng thuật toán ---");
  console.log("  " + "Thuật toán".padEnd(16) + "CL %".padStart(7) + "  Vị %".padStart(7) + "    n");
  console.log("  " + "─".repeat(42));
  for (const id of engine.ALGO_IDS) {
    const row = wf.byAlgo[id];
    console.log(
      `  ${id.padEnd(16)}${String(row.typePct).padStart(6)}% ${String(row.exactPct).padStart(6)}%  ${row.n}`,
    );
  }

  console.log("\n--- Selective Betting (chỉ bet khi shouldBet=true) ---");
  const thresholds = [0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60];
  console.log("  Threshold   CL %    Vi %    Bet    Coverage");
  console.log("  " + "─".repeat(50));
  const selectiveResults = [];
  for (const th of thresholds) {
    let selType = 0, selExact = 0, selN = 0;
    for (const st of wf.steps) {
      const bc = (st.betConfidence || 0) / 100;
      if (bc >= th) {
        selN++;
        if (st.parityOk) selType++;
        if (st.exactOk) selExact++;
      }
    }
    const clPct = selN > 0 ? Math.round((10000 * selType) / selN) / 100 : 0;
    const viPct = selN > 0 ? Math.round((10000 * selExact) / selN) / 100 : 0;
    const covPct = wf.totalSteps > 0 ? Math.round((10000 * selN) / wf.totalSteps) / 100 : 0;
    selectiveResults.push({ threshold: th, clPct, viPct, bet: selN, coverage: covPct });
    const marker = clPct >= 60 ? " ★" : "";
    console.log(
      `  ${(th * 100).toFixed(0).padStart(5)}%    ${String(clPct).padStart(6)}%  ${String(viPct).padStart(6)}%  ${String(selN).padStart(5)}   ${String(covPct).padStart(6)}%${marker}`
    );
  }

  const best60 = selectiveResults.find((r) => r.clPct >= 60 && r.coverage >= 25);
  if (best60) {
    console.log(`\n  ★ Threshold ${(best60.threshold * 100).toFixed(0)}%: CL ${best60.clPct}% với coverage ${best60.coverage}% — ĐẠT MỤC TIÊU 60%+`);
  } else {
    const bestCL = selectiveResults.reduce((a, b) => a.clPct > b.clPct ? a : b);
    console.log(`\n  Tốt nhất: threshold ${(bestCL.threshold * 100).toFixed(0)}% → CL ${bestCL.clPct}% (coverage ${bestCL.coverage}%)`);
  }

  const betSteps = wf.steps.filter((st) => st.shouldBet);
  const skipSteps = wf.steps.filter((st) => !st.shouldBet);
  const betCL = betSteps.length > 0 ? Math.round((10000 * betSteps.filter((st) => st.parityOk).length) / betSteps.length) / 100 : 0;
  const betVi = betSteps.length > 0 ? Math.round((10000 * betSteps.filter((st) => st.exactOk).length) / betSteps.length) / 100 : 0;
  const skipCL = skipSteps.length > 0 ? Math.round((10000 * skipSteps.filter((st) => st.parityOk).length) / skipSteps.length) / 100 : 0;
  const betCov = wf.totalSteps > 0 ? Math.round((10000 * betSteps.length) / wf.totalSteps) / 100 : 0;

  console.log("\n╔══════════════════════════════════════════════════╗");
  console.log("║         SELECTIVE BETTING (shouldBet)            ║");
  console.log("╠══════════════════════════════════════════════════╣");
  console.log(`║  Bet (shouldBet=true):  CL ${String(betCL).padStart(5)}%  (${betSteps.filter((st) => st.parityOk).length}/${betSteps.length})`.padEnd(51) + "║");
  console.log(`║                        Vị ${String(betVi).padStart(5)}%  (${betSteps.filter((st) => st.exactOk).length}/${betSteps.length})`.padEnd(51) + "║");
  console.log(`║  Skip (shouldBet=false): CL ${String(skipCL).padStart(5)}%  (${skipSteps.filter((st) => st.parityOk).length}/${skipSteps.length})`.padEnd(51) + "║");
  console.log(`║  Coverage: ${betCov}% (${betSteps.length}/${wf.totalSteps} rounds)`.padEnd(51) + "║");
  console.log(`║  ${betCL >= 60 ? "★ ĐẠT MỤC TIÊU 60%+" : "○ Chưa đạt 60%"}`.padEnd(51) + "║");
  console.log("╚══════════════════════════════════════════════════╝");

  if (opts.detail) {
    console.log("\n--- Chi tiết 20 bước cuối ---");
    const tail = wf.steps.slice(-20);
    tail.forEach((st) => {
      const predLbl = RED_LABELS[st.predDyn] || st.predDyn;
      const actLbl = RED_LABELS[st.actual] || st.actual;
      const clMark = st.parityOk ? "✓" : "✗";
      const viMark = st.exactOk ? "✓" : "✗";
      console.log(
        `  #${String(st.step).padStart(4)} ${st.round_id}  [${st.historySize} phiên]  ` +
        `dự: ${predLbl} (${st.predDynParity === "chan" ? "C" : "L"})  ` +
        `thực: ${actLbl} (${st.actualParity === "chan" ? "C" : "L"})  ` +
        `CL:${clMark} Vị:${viMark}  conf:${st.predDynConf}%`,
      );
    });
  }

  const report = {
    generatedAt: new Date().toISOString(),
    elapsedSec: parseFloat(elapsedSec),
    roundsDir: opts.roundsDir,
    jsonFileCount: fileNames.length,
    mergedCount: merged.length,
    skippedCount: skipped.length,
    skipped: skipped.length <= 30 ? skipped : skipped.slice(0, 10),
    config: {
      burnIn: opts.burnIn,
      HIT_BLEND_EXACT: opts.hitBlendExact,
      BETA: opts.beta,
      algorithms: engine.ALGO_IDS,
      algorithmCount: engine.ALGO_IDS.length,
    },
    summary: wf.summary,
    byAlgo: wf.byAlgo,
    baselines,
    selectiveBetting: selectiveResults,
    steps: wf.steps,
  };

  console.log(`\nĐã ghi: ${opts.outPath}`);
  fs.writeFileSync(opts.outPath, JSON.stringify(report, null, 2), "utf8");

  const stepsSizeMB = (Buffer.byteLength(JSON.stringify(report.steps)) / 1e6).toFixed(1);
  console.log(`Kích thước report: ~${stepsSizeMB} MB (${wf.steps.length} bước chi tiết)\n`);
}

main();
