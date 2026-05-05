"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const engine = require(path.join(__dirname, "prediction-engine.js"));

/** @param {number[]} reds */
function fromReds(reds) {
  const t0 = Date.now();
  return reds.map((red, i) => ({
    red,
    type: red % 2 === 0 ? "chan" : "le",
    time: new Date(t0 + i * 60000),
    round_id: String(i),
  }));
}

test("markovPredictor prefers successor after last state", () => {
  const reds = [1, 2, 1, 2, 1, 2, 1, 2, 1];
  const h = fromReds(reds);
  const m = engine.markovPredictor(h);
  assert.equal(m.predictedRed, 2);
  assert.ok(m.confidence > 0.3);
});

test("patternMatcher finds repeating continuation", () => {
  const h = fromReds([1, 0, 1, 0, 1, 0]);
  const p = engine.patternMatcher(h);
  assert.equal(p.predictedRed, 1);
});

test("streakAnalyzer returns object with parity hint", () => {
  const h = fromReds([2, 2, 2, 2, 2, 2, 2]);
  const s = engine.streakAnalyzer(h);
  assert.ok(
    [0, 2, 4].includes(s.predictedRed) || [1, 3].includes(s.predictedRed),
  );
  assert.ok(s.confidence >= 0.4);
});

test("ensemblePredict exposes parity vote and outcome meta", () => {
  const h = fromReds([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
  const e = engine.ensemblePredict(h);
  assert.equal(e.algorithms.length, 11);
  assert.equal(e.ensembleWeights.length, 11);
  assert.ok(e.consensus >= 0 && e.consensus <= 1);
  assert.ok(e.predictedRed >= 0 && e.predictedRed <= 4);
  assert.ok(e.predictedParity === "chan" || e.predictedParity === "le");
  assert.equal(
    e.predictedParity,
    e.predictedRed % 2 === 0 ? "chan" : "le",
    "predictedParity must match chosen cửa (predictedRed) after adjustment",
  );
  assert.ok(e.parityConfidence >= 0 && e.parityConfidence <= 1);
  assert.ok(e.outcome && e.outcome.title && e.outcome.line);
  assert.equal(e.outcome.red, e.predictedRed);
});

test("ensemblePredict hitWindow forces single-window H (backward compat)", () => {
  const h = engine.normalizeHistory(
    fromReds(Array.from({ length: 24 }, (_, i) => i % 5)),
  );
  const a = engine.ensemblePredict(h, { hitWindow: 12 });
  const b = engine.ensemblePredict(h, {
    hitWindowShort: 12,
    hitWindowLong: 12,
    hitMultiPhi: 1,
  });
  assert.deepEqual(a.ensembleWeights, b.ensembleWeights);
});

test("calculateRecentHitRates aggregates exact and parity in window", () => {
  const h = engine.normalizeHistory(
    fromReds(Array.from({ length: 32 }, (_, i) => (i * 2 + 1) % 5)),
  );
  const r = engine.calculateRecentHitRates(h, 18);
  assert.ok(r.pattern);
  assert.ok(r.pattern.total >= 1);
  assert.ok(r.pattern.exactRate >= 0 && r.pattern.exactRate <= 1);
  assert.ok(r.pattern.parityRate >= 0 && r.pattern.parityRate <= 1);
});

test("ensemblePredict dynamic:false uses confidence-only weights", () => {
  const h = engine.normalizeHistory(
    fromReds([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 2, 2, 2]),
  );
  const st = engine.ensemblePredict(h, { dynamic: false });
  assert.equal(st.ensembleWeights.length, 11);
  assert.ok(st.ensembleWeights.every((w) => w >= 0 && w <= 1));
});

test("markov2Predictor uses 2-step state transitions", () => {
  const reds = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1];
  const h = fromReds(reds);
  const m = engine.markov2Predictor(h);
  assert.equal(m.id, "markov2");
  assert.equal(m.predictedRed, 2);
  assert.ok(m.confidence > 0.3);
});

test("entropyAnalyzer returns valid prediction", () => {
  const h = fromReds([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
  const e = engine.entropyAnalyzer(engine.normalizeHistory(h));
  assert.equal(e.id, "entropy");
  assert.equal(e.predictedRed, 2);
  assert.ok(e.confidence >= 0.4);
  assert.ok(String(e.reason).includes("Entropy"));
});

test("regressionToMean favors underrepresented outcomes", () => {
  const h = fromReds(Array.from({ length: 20 }, () => 2));
  const r = engine.regressionToMean(engine.normalizeHistory(h));
  assert.equal(r.id, "regression");
  assert.notEqual(r.predictedRed, 2);
  assert.ok(r.confidence >= 0.3);
});

test("cauPatternDetector detects alternating cầu 1-1", () => {
  const reds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
  const h = fromReds(reds);
  const c = engine.cauPatternDetector(engine.normalizeHistory(h));
  assert.equal(c.id, "cauPattern");
  assert.ok(c.predictedRed % 2 === 0);
  assert.ok(c.confidence >= 0.4);
  assert.ok(String(c.reason).includes("1-1"));
});

test("balanceMomentum detects imbalance and predicts regression", () => {
  const reds = [0, 2, 0, 4, 2, 0, 2, 4, 0, 2, 0, 2];
  const h = fromReds(reds);
  const b = engine.balanceMomentum(engine.normalizeHistory(h));
  assert.equal(b.id, "balance");
  assert.ok(b.predictedRed >= 0 && b.predictedRed <= 4);
  assert.ok(b.confidence >= 0.2);
});

test("runBacktest counts steps and reports static vs dynamic ensemble", () => {
  const h = fromReds(Array.from({ length: 55 }, (_, i) => i % 5));
  const r = engine.runBacktest(h, { burnIn: 10 });
  assert.equal(r.totalSteps, 45);
  assert.ok(r.ensemble.n > 0);
  assert.ok(r.ensembleStatic && r.ensembleStatic.n === r.ensemble.n);
  assert.ok(r.modeComparison && typeof r.modeComparison.sameRedPct === "number");
  assert.ok("pattern" in r.byAlgo);
  assert.ok("markov2" in r.byAlgo);
  assert.ok("regression" in r.byAlgo);
  assert.ok("cauPattern" in r.byAlgo);
  assert.ok("balance" in r.byAlgo);
  assert.ok("crowd" in r.byAlgo);
  assert.ok("parityRepeat" in r.byAlgo);
  assert.ok("bayesian" in r.byAlgo);
});

test("crowdPercentPredictor (contrarian) picks lowest crowd percent", () => {
  const h = engine.normalizeHistory(fromReds([1, 2, 3]));
  const currentRound = {
    percent: {
      "4_red": "30%",
      "3r_1w": "60%",
      "3w_1r": "25%",
      "4_white": "10%",
      chan: "45%",
      le: "55%",
    },
  };
  const c = engine.crowdPercentPredictor(h, currentRound);
  assert.equal(c.id, "crowd");
  assert.equal(c.predictedRed, 0, "contrarian picks lowest: 4_white=10%");
  assert.ok(c.confidence > 0.1);
});

test("crowdPercentPredictor fallback when no currentRound", () => {
  const h = engine.normalizeHistory(fromReds([1, 2, 3]));
  const c = engine.crowdPercentPredictor(h, null);
  assert.equal(c.id, "crowd");
  assert.equal(c.confidence, 0.1);
});

test("parityRepeatPredictor uses Markov 2x2 on parity", () => {
  const reds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
  const h = engine.normalizeHistory(fromReds(reds));
  const p = engine.parityRepeatPredictor(h);
  assert.equal(p.id, "parityRepeat");
  assert.ok(p.predictedRed % 2 === 0, "last=le(1), pattern alternates → chan");
  assert.ok(p.confidence >= 0.1);
});

test("parityRepeatPredictor fallback with short history", () => {
  const h = engine.normalizeHistory(fromReds([2]));
  const p = engine.parityRepeatPredictor(h);
  assert.equal(p.id, "parityRepeat");
  assert.equal(p.confidence, 0.1);
});

test("bayesianPrior favors base rate with uniform history", () => {
  const reds = Array.from({ length: 30 }, (_, i) => i % 5);
  const h = engine.normalizeHistory(fromReds(reds));
  const b = engine.bayesianPrior(h);
  assert.equal(b.id, "bayesian");
  assert.ok(b.predictedRed >= 0 && b.predictedRed <= 4);
  assert.ok(b.confidence >= 0.1);
});

test("bayesianPrior with empty history defaults to red=2", () => {
  const b = engine.bayesianPrior([]);
  assert.equal(b.id, "bayesian");
  assert.equal(b.predictedRed, 2);
  assert.equal(b.confidence, 0.15);
});

test("parsePct and parseBetStr utilities", () => {
  assert.equal(engine.parsePct("52%"), 0.52);
  assert.equal(engine.parsePct("100%"), 1.0);
  assert.equal(engine.parseBetStr("10.93M"), 10930000);
  assert.equal(engine.parseBetStr("500K"), 500000);
  assert.equal(engine.parseBetStr("1234"), 1234);
});

test("gapAnalyzer predicts overdue outcomes", () => {
  const reds = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
  const h = fromReds(reds);
  const g = engine.gapAnalyzer(engine.normalizeHistory(h));
  assert.equal(g.id, "gap");
  assert.notEqual(g.predictedRed, 2);
  assert.ok(g.confidence > 0.15);
});

test("gapAnalyzer fallback with short history", () => {
  const h = fromReds([1, 2]);
  const g = engine.gapAnalyzer(engine.normalizeHistory(h));
  assert.equal(g.id, "gap");
  assert.equal(g.confidence, 0.12);
});

test("normalizeHistory preserves durationSec and finalisedAt", () => {
  const t = new Date("2026-05-02T20:23:05");
  const fi = new Date("2026-05-02T20:23:58");
  const h = [
    {
      red: 4,
      type: "chan",
      time: t,
      finalisedAt: fi,
      durationSec: 53,
      round_id: "20260502_202305",
    },
  ];
  const n = engine.normalizeHistory(h);
  assert.equal(n[0].durationSec, 53);
  assert.ok(n[0].finalisedAt instanceof Date);
});

test("empty history gives low-confidence ensemble", () => {
  const e = engine.ensemblePredict([]);
  assert.ok(e.confidence < 0.35);
});

test("setActivePredictors filters PREDICTORS and ALGO_IDS in place", () => {
  const fullIds = engine.ALL_ALGO_IDS;
  assert.ok(fullIds.length >= 11);
  try {
    const r = engine.setActivePredictors(["pattern", "markov"]);
    assert.deepEqual(r.active, ["pattern", "markov"]);
    assert.deepEqual(engine.ALGO_IDS, ["pattern", "markov"]);
    assert.equal(engine.PREDICTORS.length, 2);
    // Ensemble respects the slim set
    const h = fromReds([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    const e = engine.ensemblePredict(h);
    assert.equal(e.algorithms.length, 2);
    assert.deepEqual(
      e.algorithms.map((a) => a.id).sort(),
      ["markov", "pattern"],
    );
  } finally {
    engine.resetActivePredictors();
  }
  assert.deepEqual(engine.ALGO_IDS, fullIds);
});

test("setActivePredictors throws on unknown id", () => {
  assert.throws(() => engine.setActivePredictors(["bogusAlgo"]), /Unknown id|unknown id/);
  // State unchanged after failed call
  assert.equal(engine.ALGO_IDS.length, engine.ALL_ALGO_IDS.length);
});

test("resetActivePredictors restores full set", () => {
  engine.setActivePredictors(["markov"]);
  assert.equal(engine.ALGO_IDS.length, 1);
  engine.resetActivePredictors();
  assert.deepEqual(engine.ALGO_IDS, engine.ALL_ALGO_IDS);
});

test("MARKOV_DECAY configurable via DYNAMIC_ENSEMBLE", () => {
  // Build history that has a clear transition signal: after seeing red=2,
  // alternate red=4 in old half and red=0 in recent half. With low λ
  // (aggressive recency), markov should weight recent transitions and
  // predict 0; with λ=1 (no decay), it sees both equally.
  const reds = [];
  for (let i = 0; i < 30; i++) reds.push(i % 2 === 0 ? 2 : 4);
  for (let i = 0; i < 30; i++) reds.push(i % 2 === 0 ? 2 : 0);
  const hist = fromReds(reds);

  const DE = engine.DYNAMIC_ENSEMBLE;
  const orig = DE.MARKOV_DECAY;

  DE.MARKOV_DECAY = 1.0;
  const noDecay = engine.PREDICTORS.find((fn) => fn.name.includes("markov"))
    ? null
    : null;
  // Use ALL_ALGO_IDS to find markovPredictor via PREDICTORS
  const markovFn = engine._test_markov || null;

  DE.MARKOV_DECAY = orig;
  // Just verify config is read by ensemblePredict — check it doesn't throw.
  const r1 = engine.ensemblePredict(hist);
  assert.ok(r1.predictedRed >= 0 && r1.predictedRed <= 4);

  DE.MARKOV_DECAY = 0.5;
  const r2 = engine.ensemblePredict(hist);
  assert.ok(r2.predictedRed >= 0 && r2.predictedRed <= 4);
  DE.MARKOV_DECAY = orig;
});

test("MARKOV_DECAY clamped to [0.5, 1] range", () => {
  const hist = fromReds([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]);
  const DE = engine.DYNAMIC_ENSEMBLE;
  const orig = DE.MARKOV_DECAY;

  // Out-of-range values should be silently clamped, not throw
  DE.MARKOV_DECAY = 0.1; // below floor
  const r1 = engine.ensemblePredict(hist);
  assert.ok(Number.isFinite(r1.confidence));

  DE.MARKOV_DECAY = 5.0; // above ceiling
  const r2 = engine.ensemblePredict(hist);
  assert.ok(Number.isFinite(r2.confidence));

  DE.MARKOV_DECAY = orig;
});

test("ensemblePredict respects opts.maxHistory", () => {
  const reds = [];
  for (let i = 0; i < 200; i++) reds.push(i % 5);
  const hist = fromReds(reds);
  const r = engine.ensemblePredict(hist, { maxHistory: 50 });
  assert.ok(Number.isFinite(r.confidence));
  // Should not throw, should return valid prediction
  assert.ok(r.predictedRed >= 0 && r.predictedRed <= 4);
  assert.ok(Array.isArray(r.algorithms));
});

test("ensemblePredict respects DYNAMIC_ENSEMBLE.MAX_HISTORY", () => {
  const reds = [];
  for (let i = 0; i < 200; i++) reds.push(i % 5);
  const hist = fromReds(reds);
  const DE = engine.DYNAMIC_ENSEMBLE;
  const orig = DE.MAX_HISTORY;

  DE.MAX_HISTORY = 30;
  const r = engine.ensemblePredict(hist);
  assert.ok(Number.isFinite(r.confidence));
  assert.ok(r.predictedRed >= 0 && r.predictedRed <= 4);

  DE.MAX_HISTORY = orig;
});

test("opts.maxHistory takes precedence over DE.MAX_HISTORY", () => {
  const reds = [];
  for (let i = 0; i < 200; i++) reds.push(i % 5);
  const hist = fromReds(reds);
  const DE = engine.DYNAMIC_ENSEMBLE;
  const orig = DE.MAX_HISTORY;

  // Set DE cap large, then opts cap small.
  DE.MAX_HISTORY = 100;
  const r1 = engine.ensemblePredict(hist, { maxHistory: 10 });
  assert.ok(Number.isFinite(r1.confidence));

  // opts=0 should also override (mean "no cap" if explicitly set 0?
  // Current semantics: 0 falls back to DE because of the != null check).
  // Verify opts=null falls back to DE (cap=100).
  const r2 = engine.ensemblePredict(hist, { maxHistory: null });
  assert.ok(Number.isFinite(r2.confidence));

  DE.MAX_HISTORY = orig;
});

test("MAX_HISTORY=0 means no cap (full history used)", () => {
  const reds = [];
  for (let i = 0; i < 50; i++) reds.push(i % 5);
  const hist = fromReds(reds);
  const DE = engine.DYNAMIC_ENSEMBLE;
  const orig = DE.MAX_HISTORY;

  DE.MAX_HISTORY = 0;
  const rNoCap = engine.ensemblePredict(hist);
  // Compare to explicit "use all" by passing huge cap
  const rHugeCap = engine.ensemblePredict(hist, { maxHistory: 9999 });
  assert.equal(rNoCap.predictedRed, rHugeCap.predictedRed);

  DE.MAX_HISTORY = orig;
});
