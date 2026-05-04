/**
 * Xóc đĩa — heuristic prediction core (browser + Node).
 *
 * Input: RoundItem[] chronological, last = newest observed outcome.
 * Predicts next red count 0..4 for the following round.
 */
(function (global) {
  "use strict";

  /**
   * RoundItem — nguồn JSON rounds/*.json:
   * - dice_result → red/type
   * - started_at | round_id → time
   * - finalised_at + started_at → durationSec (ảnh hướng mẫu thời gian)
   * - percent, bets → crowd (%, total_count, total_bet)
   */
  /** @typedef {{ red: number, type: "chan"|"le", time: Date, round_id?: string, finalisedAt?: Date|null, durationSec?: number|null, percent?: Record<string,string>|null, bets?: Record<string,{total_bet?:string,total_count?:string}>|null }} RoundItem */

  const ALGO_IDS = [
    "pattern",
    "streak",
    "markov",
    "markov2",
    "time",
    "regression",
    "cauPattern",
    "balance",
    "crowd",
    "parityRepeat",
    "bayesian",
  ];

  function ensureDate(t) {
    if (t instanceof Date && !isNaN(t)) return t;
    const d = new Date(t);
    return isNaN(d) ? new Date(0) : d;
  }

  function normalizeHistory(history) {
    if (!history || !history.length) return [];
    return history.map((h) => ({
      red: h.red,
      type: h.type,
      time: ensureDate(h.time),
      round_id: h.round_id,
      finalisedAt: h.finalisedAt ? ensureDate(h.finalisedAt) : null,
      durationSec:
        h.durationSec != null &&
        Number.isFinite(h.durationSec) &&
        h.durationSec >= 0
          ? h.durationSec
          : null,
      percent: h.percent && typeof h.percent === "object" ? h.percent : null,
      bets: h.bets && typeof h.bets === "object" ? h.bets : null,
    }));
  }


  /**
   * Năm kết cục xúc xắc (số đỏ 0–4) — tiêu đề + mô tả theo luật chơi.
   */
  const OUTCOME_META_VI = [
    {
      red: 0,
      short: "4 Trắng",
      title: "4 TRẮNG (Tứ tử trắng)",
      line: "Toàn bộ 4 quân vị đều hiển thị mặt trắng.",
      parity: "chan",
    },
    {
      red: 1,
      short: "3T1Đ",
      title: "3 TRẮNG — 1 ĐỎ",
      line: "Có 3 quân mặt trắng và 1 quân mặt đỏ.",
      parity: "le",
    },
    {
      red: 2,
      short: "Sấp đôi",
      title: "2 ĐỎ — 2 TRẮNG (Sấp đôi)",
      line: "Có 2 quân mặt đỏ và 2 quân mặt trắng. Đây là trường hợp phổ biến nhất của cửa Chẵn.",
      parity: "chan",
    },
    {
      red: 3,
      short: "3Đ1T",
      title: "3 ĐỎ — 1 TRẮNG",
      line: "Có 3 quân mặt đỏ và 1 quân mặt trắng.",
      parity: "le",
    },
    {
      red: 4,
      short: "4 Đỏ",
      title: "4 ĐỎ (Tứ tử đỏ)",
      line: "Toàn bộ 4 quân vị đều hiển thị mặt đỏ.",
      parity: "chan",
    },
  ];

  function getOutcomeMeta(red) {
    const r = Math.max(0, Math.min(4, red | 0));
    return OUTCOME_META_VI[r];
  }


  function pickRedLeastFrequentInWindow(history, wantLe) {
    const win = history.slice(-20);
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    win.forEach((h) => counts[h.red]++);
    const parity = wantLe ? 1 : 0;
    const candidates = [0, 1, 2, 3, 4].filter((r) => r % 2 === parity);
    candidates.sort((a, b) => counts[a] - counts[b] || a - b);
    return candidates[0];
  }

  function pickRedModeInTail(history, tailLen) {
    const tail = history.slice(-tailLen);
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    tail.forEach((h) => counts[h.red]++);
    let best = 0;
    let bestC = -1;
    for (let r = 0; r <= 4; r++) {
      if (counts[r] > bestC) {
        bestC = counts[r];
        best = r;
      }
    }
    return best;
  }

  function patternMatcher(history) {
    const name = "Pattern Matching";
    const reds = history.map((h) => h.red);
    const n = reds.length;
    if (n < 2) {
      return {
        id: "pattern",
        name,
        predictedRed: 2,
        confidence: 0.12,
        parityConfidence: 0.12,
        reason: "Chưa đủ phiên để nhận mẫu.",
      };
    }

    let bestMeta = null;
    let bestScore = 0;

    const maxPrefix = Math.min(6, n - 1);
    for (let prefixLen = maxPrefix; prefixLen >= 1; prefixLen--) {
      const suffix = reds.slice(n - prefixLen);
      const nextCounts = {};
      for (let i = 0; i <= n - prefixLen - 1; i++) {
        let ok = true;
        for (let j = 0; j < prefixLen; j++) {
          if (reds[i + j] !== suffix[j]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          const nx = reds[i + prefixLen];
          nextCounts[nx] = (nextCounts[nx] || 0) + 1;
        }
      }
      const total = Object.values(nextCounts).reduce((a, b) => a + b, 0);
      if (total === 0) continue;
      let bestR = 0;
      let bestC = 0;
      for (const r of Object.keys(nextCounts)) {
        const c = nextCounts[r];
        if (c > bestC) {
          bestC = c;
          bestR = +r;
        }
      }
      const purity = bestC / total;
      const score = prefixLen * (1 + Math.log1p(total)) * (0.5 + purity);
      if (score > bestScore) {
        bestScore = score;
        bestMeta = { red: bestR, prefixLen, total, purity, nextCounts };
      }
    }

    const parities = reds.map((r) => r % 2);
    let parityMeta = null;
    let parityBestScore = 0;
    const maxParityPrefix = Math.min(8, n - 1);
    for (let prefixLen = maxParityPrefix; prefixLen >= 2; prefixLen--) {
      const suffix = parities.slice(n - prefixLen);
      let chanNext = 0, leNext = 0;
      for (let i = 0; i <= n - prefixLen - 1; i++) {
        let ok = true;
        for (let j = 0; j < prefixLen; j++) {
          if (parities[i + j] !== suffix[j]) { ok = false; break; }
        }
        if (ok) {
          if (parities[i + prefixLen] === 0) chanNext++;
          else leNext++;
        }
      }
      const total = chanNext + leNext;
      if (total < 2) continue;
      const dominant = Math.max(chanNext, leNext);
      const purity = dominant / total;
      const score = prefixLen * (1 + Math.log1p(total)) * (0.3 + purity);
      if (score > parityBestScore) {
        parityBestScore = score;
        parityMeta = { parity: chanNext >= leNext ? 0 : 1, prefixLen, total, purity };
      }
    }

    if (!bestMeta && !parityMeta) {
      return {
        id: "pattern",
        name,
        predictedRed: 2,
        confidence: 0.18,
        parityConfidence: 0.18,
        reason: "Không thấy mẫu tiền tố trùng trong lịch sử.",
      };
    }

    let predictedRed, confidence, parityConfidence, reason;

    if (bestMeta) {
      confidence = Math.min(0.92, 0.32 + bestMeta.purity * 0.38 + bestMeta.prefixLen * 0.04);
      predictedRed = bestMeta.red;

      let parTotal = 0;
      const targetParity = bestMeta.red % 2;
      const nc = bestMeta.nextCounts;
      const allTotal = Object.values(nc).reduce((a, b) => a + b, 0);
      for (const r of Object.keys(nc)) {
        if (+r % 2 === targetParity) parTotal += nc[r];
      }
      const redParityPurity = allTotal > 0 ? parTotal / allTotal : 0.5;
      parityConfidence = Math.min(0.92, 0.3 + redParityPurity * 0.5 + bestMeta.prefixLen * 0.04);
      reason = `Tiền tố ${bestMeta.prefixLen} mẫu, ${bestMeta.total} lần khớp; hay tiếp ${bestMeta.red} (~${Math.round(bestMeta.purity * 100)}%).`;
    } else {
      predictedRed = parityMeta.parity === 0 ? 2 : 3;
      confidence = Math.min(0.7, 0.2 + parityMeta.purity * 0.3);
      parityConfidence = Math.min(0.88, 0.35 + parityMeta.purity * 0.4 + parityMeta.prefixLen * 0.03);
      reason = `Parity pattern ${parityMeta.prefixLen} mẫu, ${parityMeta.total} khớp; ${parityMeta.parity === 0 ? "chan" : "le"} (~${Math.round(parityMeta.purity * 100)}%).`;
    }

    if (parityMeta) {
      const parPC = Math.min(0.92, 0.35 + parityMeta.purity * 0.45 + parityMeta.prefixLen * 0.03);
      if (parPC > parityConfidence) {
        parityConfidence = parPC;
      }
    }

    return { id: "pattern", name, predictedRed, confidence, parityConfidence, reason };
  }

  function streakAnalyzer(history) {
    const name = "Streak Analysis";
    const n = history.length;
    if (n < 2) {
      return {
        id: "streak",
        name,
        predictedRed: 2,
        confidence: 0.15,
        parityConfidence: 0.15,
        reason: "Cần thêm dữ liệu cho streak.",
      };
    }

    const lastType = history[n - 1].type;
    let streak = 1;
    for (let i = n - 2; i >= 0; i--) {
      if (history[i].type === lastType) streak++;
      else break;
    }

    const runs = [];
    let runLen = 1;
    for (let i = 1; i < n; i++) {
      if (history[i].type === history[i - 1].type) runLen++;
      else {
        runs.push({ type: history[i - 1].type, len: runLen });
        runLen = 1;
      }
    }
    runs.push({ type: history[n - 1].type, len: runLen });

    const lensForType = runs
      .filter((r) => r.type === lastType)
      .map((r) => r.len);
    const avg =
      lensForType.length > 0
        ? lensForType.reduce((a, b) => a + b, 0) / lensForType.length
        : 2;

    const trendContinue = streak < Math.max(2, avg * 1.2);

    const tail = history.slice(-streak);
    const freq = {};
    tail.forEach((h) => {
      freq[h.red] = (freq[h.red] || 0) + 1;
    });
    let contRed = 2;
    let maxF = -1;
    for (let r = 0; r <= 4; r++) {
      const f = freq[r] || 0;
      if (f > maxF) {
        maxF = f;
        contRed = r;
      }
    }

    let predictedRed;
    let reason;
    if (trendContinue) {
      predictedRed = contRed;
      reason = `Chuỗi ${lastType === "chan" ? "Chẵn" : "Lẻ"} dài ${streak} (< TB ~${avg.toFixed(1)}); nghiêng tiếp diễn → ${predictedRed}.`;
    } else {
      const wantLe = lastType === "chan";
      predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
      reason = `Chuỗi ${lastType === "chan" ? "Chẵn" : "Lẻ"} ${streak} phiên (≥ ngưỡng đảo); nghiêng ${wantLe ? "Lẻ" : "Chẵn"} → ${predictedRed}.`;
    }

    const confidence = Math.min(0.9, 0.42 + Math.min(streak, 7) * 0.065);
    const parityConfidence = Math.min(0.92, confidence + 0.08);
    return { id: "streak", name, predictedRed, confidence, parityConfidence, reason };
  }

  function frequencyBalancer(history) {
    const name = "Frequency Balance";
    const win = history.slice(-30);
    if (win.length < 8) {
      return {
        id: "frequency",
        name,
        predictedRed: 2,
        confidence: 0.2,
        reason: "Cần ≥8 phiên trong cửa sổ 30 để cân bằng.",
      };
    }
    const total = win.length;
    let chan = 0;
    const v = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    win.forEach((h) => {
      if (h.type === "chan") chan++;
      v[h.red]++;
    });
    const chanRatio = chan / total;

    const target = 0.2;
    const deficits = [];
    for (let r = 0; r <= 4; r++) {
      const p = v[r] / total;
      deficits.push({ r, gap: target - p });
    }
    deficits.sort((a, b) => b.gap - a.gap);

    let predictedRed = deficits[0].r;
    let reason = `Cửa sổ 30: Chẵn ${Math.round(chanRatio * 100)}%; ưu tiên vị thiếu so 20% → ${predictedRed}.`;

    if (chanRatio < 0.45) {
      const wantLe = false;
      predictedRed = pickRedLeastFrequentInWindow(win, wantLe);
      reason = `Chẵn chỉ ${Math.round(chanRatio * 100)}% (<45%); cân bằng nghiêng Chẵn → ${predictedRed}.`;
    } else if (chanRatio > 0.55) {
      const wantLe = true;
      predictedRed = pickRedLeastFrequentInWindow(win, wantLe);
      reason = `Chẵn ${Math.round(chanRatio * 100)}% (>55%); cân bằng nghiêng Lẻ → ${predictedRed}.`;
    }

    const confidence = Math.min(0.85, 0.4 + Math.abs(0.5 - chanRatio));
    return { id: "frequency", name, predictedRed, confidence, reason };
  }

  const MARKOV_DECAY = 0.98;

  function markovPredictor(history) {
    const name = "Markov Chain";
    const reds = history.map((h) => h.red);
    const n = reds.length;
    if (n < 2) {
      return {
        id: "markov",
        name,
        predictedRed: 2,
        confidence: 0.18,
        parityConfidence: 0.18,
        reason: "Cần ít nhất 2 phiên cho Markov bậc 1.",
      };
    }
    const last = reds[n - 1];
    const counts = {};
    for (let r = 0; r <= 4; r++) counts[r] = 0.2;
    for (let i = 0; i < n - 1; i++) {
      if (reds[i] === last) {
        const w = Math.pow(MARKOV_DECAY, n - 2 - i);
        counts[reds[i + 1]] += w;
      }
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    let best = 0;
    let bestP = 0;
    for (let r = 0; r <= 4; r++) {
      const p = counts[r] / total;
      if (p > bestP) {
        bestP = p;
        best = r;
      }
    }
    const confidence = Math.min(0.95, 0.25 + bestP * 1.1);
    const parityConfidence = Math.min(0.85, confidence * 0.9);
    const reason = `Sau mặt ${last}, Markov decay (λ=${MARKOV_DECAY}) chọn ${best} (P≈${bestP.toFixed(2)}).`;
    return { id: "markov", name, predictedRed: best, confidence, parityConfidence, reason };
  }

  function hotColdAnalyzer(history) {
    const name = "Hot/Cold Numbers";
    const W = 35;
    const win = history.slice(-W);
    if (win.length < 10) {
      return {
        id: "hotcold",
        name,
        predictedRed: 2,
        confidence: 0.22,
        reason: "Cần ≥10 phiên trong cửa sổ nóng/lạnh.",
      };
    }
    const n = history.length;
    const lastIdx = {};
    for (let i = 0; i < n; i++) lastIdx[history[i].red] = i;

    const hotCount = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    win.forEach((h) => hotCount[h.red]++);

    const scores = {};
    for (let r = 0; r <= 4; r++) {
      const hot = hotCount[r] / win.length;
      const roundsSince = n - 1 - (lastIdx[r] ?? -999);
      const cold = Math.min(1, roundsSince / 24);
      scores[r] = 0.55 * hot + 0.45 * cold;
    }
    let best = 0;
    let bestS = -1;
    for (let r = 0; r <= 4; r++) {
      if (scores[r] > bestS) {
        bestS = scores[r];
        best = r;
      }
    }
    const confidence = Math.min(0.88, 0.38 + bestS * 0.45);
    const rs = n - 1 - (lastIdx[best] ?? 0);
    const reason = `Điểm nóng/lạnh cao nhất: ${best} (gần đây ${hotCount[best]}/${win.length}, lần cuối cách ${rs} phiên).`;
    return { id: "hotcold", name, predictedRed: best, confidence, reason };
  }

  /** Bin độ dài phiên (giây) để lọc mẫu thời gian; null = không có finalised_at */
  function durationBin(sec) {
    if (sec == null || !Number.isFinite(sec) || sec < 0) return null;
    return Math.min(45, Math.floor(sec / 5));
  }

  function timeAndDurationMatch(refItem, hi, refDay, refHour) {
    if (hi.time.getDay() !== refDay || hi.time.getHours() !== refHour)
      return false;
    const br = durationBin(refItem.durationSec);
    const bi = durationBin(hi.durationSec);
    if (br == null || bi == null) return true;
    return Math.abs(br - bi) <= 1;
  }

  function timePatternAnalyzer(history) {
    const name = "Time Pattern";
    const n = history.length;
    if (n < 12) {
      return {
        id: "time",
        name,
        predictedRed: pickRedModeInTail(history, Math.min(12, n)),
        confidence: 0.2,
        parityConfidence: 0.2,
        reason: "Ít phiên — dùng mode ngắn hạn, độ tin cậy thấp.",
      };
    }

    const refItem = history[n - 1];
    const ref = refItem.time;
    const day = ref.getDay();
    const hour = ref.getHours();
    const bucketCounts = { 0: 1, 1: 1, 2: 1, 3: 1, 4: 1 };
    let bucketN = 5;
    for (let i = 0; i < n - 1; i++) {
      const hi = history[i];
      if (!timeAndDurationMatch(refItem, hi, day, hour)) continue;
      bucketCounts[history[i + 1].red]++;
      bucketN++;
    }
    const total = Object.values(bucketCounts).reduce((a, b) => a + b, 0);
    let best = 0;
    let bestP = 0;
    for (let r = 0; r <= 4; r++) {
      const p = bucketCounts[r] / total;
      if (p > bestP) {
        bestP = p;
        best = r;
      }
    }
    const dataStrength = Math.min(1, (bucketN - 5) / 25);
    const confidence = Math.min(
      0.85,
      0.22 + bestP * 0.55 + dataStrength * 0.15,
    );
    const durHint =
      refItem.durationSec != null && durationBin(refItem.durationSec) != null
        ? `, ~${refItem.durationSec}s/phiên`
        : ", không lọc độ dài (thiếu finalised_at)";
    const parityConfidence = Math.min(0.8, confidence * 0.85);
    const reason = `Bucket thứ ${day}, giờ ${hour}h${durHint}: sau các phiên tương tự hay ra ${best} (P≈${bestP.toFixed(2)}, n=${bucketN}).`;
    return { id: "time", name, predictedRed: best, confidence, parityConfidence, reason };
  }

  function matchExact(pred, actual) {
    return pred === actual;
  }

  function matchType(pred, actual) {
    const pt = pred % 2 === 0 ? "chan" : "le";
    const at = actual % 2 === 0 ? "chan" : "le";
    return pt === at;
  }

  /**
   * Markov bậc 2: trạng thái = (red[n-2], red[n-1]) → đếm chuyển tiếp.
   * Bắt được pattern 2 bước mà Markov bậc 1 bỏ lỡ.
   */
  function markov2Predictor(history) {
    const name = "Markov Chain Bậc 2";
    const reds = history.map((h) => h.red);
    const n = reds.length;
    if (n < 3) {
      return {
        id: "markov2",
        name,
        predictedRed: 2,
        confidence: 0.15,
        parityConfidence: 0.15,
        reason: "Cần ít nhất 3 phiên cho Markov bậc 2.",
      };
    }
    const last2 = `${reds[n - 2]},${reds[n - 1]}`;
    const counts = {};
    for (let r = 0; r <= 4; r++) counts[r] = 0.2;
    let stateMatches = 0;
    for (let i = 0; i < n - 2; i++) {
      const state = `${reds[i]},${reds[i + 1]}`;
      if (state === last2) {
        const w = Math.pow(MARKOV_DECAY, n - 3 - i);
        counts[reds[i + 2]] += w;
        stateMatches++;
      }
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    let best = 0;
    let bestP = 0;
    for (let r = 0; r <= 4; r++) {
      const p = counts[r] / total;
      if (p > bestP) {
        bestP = p;
        best = r;
      }
    }
    const confidence = Math.min(
      0.92,
      0.2 + bestP * 0.9 + Math.min(stateMatches, 15) * 0.02,
    );
    const parityConfidence = Math.min(0.85, confidence * 0.9);
    const reason = `Sau cặp [${reds[n - 2]},${reds[n - 1]}], Markov bậc 2 decay chọn ${best} (P≈${bestP.toFixed(2)}, ${stateMatches} mẫu trùng).`;
    return { id: "markov2", name, predictedRed: best, confidence, parityConfidence, reason };
  }

  /**
   * Entropy Analysis: đo Shannon entropy cửa sổ gần đây.
   * Entropy thấp → phân bố lệch → tiếp tục vị phổ biến.
   * Entropy cao → phân bố đều → bù vị ít xuất hiện nhất.
   */
  function entropyAnalyzer(history) {
    const name = "Entropy Analysis";
    const W = 20;
    const win = history.slice(-W);
    if (win.length < 8) {
      return {
        id: "entropy",
        name,
        predictedRed: 2,
        confidence: 0.18,
        reason: "Cần ≥8 phiên để phân tích entropy.",
      };
    }
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    win.forEach((h) => counts[h.red]++);
    const len = win.length;

    let entropy = 0;
    for (let r = 0; r <= 4; r++) {
      const p = counts[r] / len;
      if (p > 0) entropy -= p * Math.log2(p);
    }
    const maxEntropy = Math.log2(5);
    const normE = entropy / maxEntropy;

    let predictedRed;
    let confidence;
    let reason;

    if (normE < 0.7) {
      let best = 0;
      let bestC = -1;
      for (let r = 0; r <= 4; r++) {
        if (counts[r] > bestC) {
          bestC = counts[r];
          best = r;
        }
      }
      predictedRed = best;
      confidence = Math.min(0.9, 0.45 + (1 - normE) * 0.5);
      reason = `Entropy thấp (${entropy.toFixed(2)}/${maxEntropy.toFixed(2)} = ${Math.round(normE * 100)}%); phân bố lệch → tiếp tục ${best} (${bestC}/${len}).`;
    } else {
      let best = 0;
      let bestC = Infinity;
      for (let r = 0; r <= 4; r++) {
        if (counts[r] < bestC) {
          bestC = counts[r];
          best = r;
        }
      }
      predictedRed = best;
      confidence = Math.min(0.78, 0.3 + normE * 0.35);
      reason = `Entropy cao (${entropy.toFixed(2)}/${maxEntropy.toFixed(2)} = ${Math.round(normE * 100)}%); phân bố đều → bù vị ít nhất ${best} (${bestC}/${len}).`;
    }

    return { id: "entropy", name, predictedRed, confidence, reason };
  }

  /**
   * Regression to Mean: so sánh tần suất thực tế với phân bố lý thuyết
   * (nhị thức 4 đồng xu, p=0.5). Ưu tiên vị đang thiếu so kỳ vọng.
   */
  function regressionToMean(history) {
    const name = "Regression to Mean";
    const W = 25;
    const win = history.slice(-W);
    if (win.length < 10) {
      return {
        id: "regression",
        name,
        predictedRed: 2,
        confidence: 0.22,
        parityConfidence: 0.22,
        reason: "Cần ≥10 phiên cho regression to mean.",
      };
    }
    const theoretical = {
      0: 1 / 16,
      1: 4 / 16,
      2: 6 / 16,
      3: 4 / 16,
      4: 1 / 16,
    };
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    win.forEach((h) => counts[h.red]++);
    const len = win.length;

    const scores = {};
    let maxDev = 0;
    for (let r = 0; r <= 4; r++) {
      const observed = counts[r] / len;
      const expected = theoretical[r];
      const dev = expected - observed;
      scores[r] = dev + expected;
      maxDev = Math.max(maxDev, Math.abs(dev));
    }

    let best = 2;
    let bestS = -Infinity;
    for (let r = 0; r <= 4; r++) {
      if (scores[r] > bestS) {
        bestS = scores[r];
        best = r;
      }
    }

    const confidence = Math.min(0.85, 0.35 + maxDev * 2.5);
    const obsPct = Math.round((counts[best] / len) * 100);
    const expPct = Math.round(theoretical[best] * 100);
    const parityConfidence = Math.min(0.82, confidence * 0.9);
    const reason = `Vị ${best}: thực tế ${obsPct}% vs lý thuyết ${expPct}% trong ${len} phiên; regression → ưu tiên vị thiếu.`;
    return { id: "regression", name, predictedRed: best, confidence, parityConfidence, reason };
  }

  /**
   * Cầu Pattern Detector (Martingale + Paroli insight):
   * Phân tích phân bố độ dài cầu (run-length) chẵn/lẻ gần đây.
   * - Cầu ngắn (median ≤ 1.5) → "cầu 1-1" alternating → predict đảo chiều.
   * - Cầu dài (median ≥ 3) → "cầu bệt" → predict tiếp diễn.
   * - Giữa → tín hiệu trung tính, nghiêng nhẹ theo xu hướng gần nhất.
   */
  function cauPatternDetector(history) {
    const name = "Cầu Pattern";
    const n = history.length;
    if (n < 6) {
      return {
        id: "cauPattern",
        name,
        predictedRed: 2,
        confidence: 0.15,
        parityConfidence: 0.15,
        reason: "Cần ≥6 phiên để phân tích cầu.",
      };
    }

    const W = Math.min(40, n);
    const win = history.slice(-W);

    const runs = [];
    let runLen = 1;
    for (let i = 1; i < win.length; i++) {
      if (win[i].type === win[i - 1].type) {
        runLen++;
      } else {
        runs.push(runLen);
        runLen = 1;
      }
    }
    runs.push(runLen);

    if (runs.length < 3) {
      return {
        id: "cauPattern",
        name,
        predictedRed: pickRedModeInTail(history, Math.min(10, n)),
        confidence: 0.2,
        parityConfidence: 0.2,
        reason: `Chỉ ${runs.length} cầu — chưa đủ dữ liệu run-length.`,
      };
    }

    const sorted = runs.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median =
      sorted.length % 2 === 1
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
    const mean = runs.reduce((a, b) => a + b, 0) / runs.length;

    const lastType = win[win.length - 1].type;
    const currentRunLen = runs[runs.length - 1];
    let predictedRed;
    let reason;
    let confidence;

    if (median <= 1.5) {
      const wantLe = lastType === "chan";
      predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
      confidence = Math.min(
        0.88,
        0.45 + (1.5 - median) * 0.2 + Math.min(runs.length, 15) * 0.015,
      );
      reason = `Cầu 1-1 (median=${median.toFixed(1)}, TB=${mean.toFixed(1)}, ${runs.length} cầu); đảo chiều ${wantLe ? "Lẻ" : "Chẵn"} → ${predictedRed}.`;
    } else if (median >= 3) {
      if (currentRunLen < median * 0.8) {
        predictedRed = pickRedModeInTail(
          history.slice(-currentRunLen),
          currentRunLen,
        );
        confidence = Math.min(
          0.85,
          0.42 + (median - 3) * 0.08 + Math.min(runs.length, 15) * 0.015,
        );
        reason = `Cầu bệt (median=${median.toFixed(1)}, cầu hiện tại=${currentRunLen} < median); tiếp diễn ${lastType === "chan" ? "Chẵn" : "Lẻ"} → ${predictedRed}.`;
      } else {
        const wantLe = lastType === "chan";
        predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
        confidence = Math.min(0.75, 0.35 + Math.min(runs.length, 15) * 0.015);
        reason = `Cầu bệt nhưng cầu hiện tại=${currentRunLen} ≥ median=${median.toFixed(1)}; có thể sắp đảo → ${predictedRed}.`;
      }
    } else {
      const recentRuns = runs.slice(-5);
      const recentAlternating =
        recentRuns.length >= 3 && recentRuns.every((r) => r <= 2);
      if (recentAlternating) {
        const wantLe = lastType === "chan";
        predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
        confidence = 0.4;
        reason = `Cầu trung tính (median=${median.toFixed(1)}), nhưng 5 cầu gần đây ngắn ≤2 → nghiêng đảo → ${predictedRed}.`;
      } else {
        predictedRed = pickRedModeInTail(history, Math.min(10, n));
        confidence = 0.32;
        reason = `Cầu trung tính (median=${median.toFixed(1)}, TB=${mean.toFixed(1)}); không rõ xu hướng → mode gần đây → ${predictedRed}.`;
      }
    }

    const parityConfidence = Math.min(0.92, confidence + 0.1);
    return { id: "cauPattern", name, predictedRed, confidence, parityConfidence, reason };
  }

  /**
   * Balance Momentum (D'Alembert + Fibonacci insight):
   * Theo dõi độ lệch tích lũy chẵn/lẻ với EMA (exponential moving average).
   * Lệch càng sâu → tín hiệu regression càng mạnh (D'Alembert).
   * Fibonacci-inspired: áp lực cân bằng tăng theo cấp số (không tuyến tính).
   */
  function balanceMomentum(history) {
    const name = "Balance Momentum";
    const n = history.length;
    if (n < 8) {
      return {
        id: "balance",
        name,
        predictedRed: 2,
        confidence: 0.18,
        parityConfidence: 0.18,
        reason: "Cần ≥8 phiên cho balance momentum.",
      };
    }

    const W = Math.min(50, n);
    const win = history.slice(-W);
    const alpha = 2 / (W + 1);

    let ema = 0;
    for (let i = 0; i < win.length; i++) {
      const signal = win[i].type === "chan" ? 1 : -1;
      ema = alpha * signal + (1 - alpha) * ema;
    }

    const fibSteps = [1, 1, 2, 3, 5, 8, 13, 21];
    let currentStreak = 1;
    const lastType = win[win.length - 1].type;
    for (let i = win.length - 2; i >= 0; i--) {
      if (win[i].type === lastType) currentStreak++;
      else break;
    }
    const fibIdx = Math.min(currentStreak - 1, fibSteps.length - 1);
    const fibPressure = fibSteps[fibIdx] / fibSteps[fibSteps.length - 1];

    const absEma = Math.abs(ema);
    let predictedRed;
    let reason;
    let confidence;

    if (absEma > 0.15) {
      const wantLe = ema > 0;
      predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
      const pressure = Math.min(1, absEma * 3 + fibPressure * 0.3);
      confidence = Math.min(0.88, 0.38 + pressure * 0.45);
      const dir = ema > 0 ? "Chẵn" : "Lẻ";
      reason = `EMA lệch ${dir} (${ema.toFixed(3)}); D'Alembert regression + Fibonacci áp lực cầu ${currentStreak} → nghiêng ${wantLe ? "Lẻ" : "Chẵn"} → ${predictedRed}.`;
    } else if (absEma > 0.05) {
      const wantLe = ema > 0;
      predictedRed = pickRedLeastFrequentInWindow(history, wantLe);
      confidence = Math.min(0.7, 0.3 + absEma * 2.5);
      const dir = ema > 0 ? "Chẵn" : "Lẻ";
      reason = `EMA nghiêng nhẹ ${dir} (${ema.toFixed(3)}); regression nhẹ → ${predictedRed}.`;
    } else {
      const tail = history.slice(-5);
      const freq = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
      tail.forEach((h) => (freq[h.red] = (freq[h.red] || 0) + 1));
      let best = 2;
      let bestF = -1;
      for (let r = 0; r <= 4; r++) {
        if (freq[r] > bestF) {
          bestF = freq[r];
          best = r;
        }
      }
      predictedRed = best;
      confidence = 0.28;
      reason = `EMA cân bằng (${ema.toFixed(3)}); D'Alembert trung tính → mode ngắn hạn → ${predictedRed}.`;
    }

    const parityConfidence = Math.min(0.92, confidence + 0.1);
    return { id: "balance", name, predictedRed, confidence, parityConfidence, reason };
  }

  const PCT_KEY_TO_RED = { "4_red": 4, "3r_1w": 3, "3w_1r": 1, "4_white": 0 };

  function parsePct(s) {
    if (typeof s === "number") return s;
    if (typeof s !== "string") return NaN;
    return parseFloat(s.replace("%", "")) / 100;
  }

  function parseBetStr(s) {
    if (typeof s === "number") return s;
    if (typeof s !== "string") return 0;
    const cleaned = s.trim().toUpperCase();
    if (cleaned.endsWith("M"))
      return parseFloat(cleaned) * 1e6 || 0;
    if (cleaned.endsWith("K"))
      return parseFloat(cleaned) * 1e3 || 0;
    return parseFloat(cleaned) || 0;
  }

  /**
   * Contrarian Crowd: đi NGƯỢC crowd prediction. Crowd thường sai,
   * chọn outcome có crowd% THẤP nhất trong các outcome cụ thể.
   */
  function crowdPercentPredictor(history, currentRound) {
    const name = "Contrarian Crowd";
    const pct = currentRound && currentRound.percent ? currentRound.percent : null;
    if (!pct) {
      return {
        id: "crowd",
        name,
        predictedRed: 2,
        confidence: 0.1,
        parityConfidence: 0.1,
        reason: "Không có dữ liệu percent cho round hiện tại.",
      };
    }

    const parsed = [];
    for (const [key, red] of Object.entries(PCT_KEY_TO_RED)) {
      const p = parsePct(pct[key]);
      if (Number.isFinite(p)) parsed.push({ key, red, p });
    }
    if (parsed.length === 0) {
      return {
        id: "crowd",
        name,
        predictedRed: 2,
        confidence: 0.12,
        parityConfidence: 0.12,
        reason: "Không parse được percent.",
      };
    }

    parsed.sort((a, b) => a.p - b.p);
    let bestRed = parsed[0].red;

    const chanPct = parsePct(pct.chan);
    const lePct = parsePct(pct.le);
    let parityLean = null;
    if (Number.isFinite(chanPct) && Number.isFinite(lePct)) {
      parityLean = chanPct < lePct ? "chan" : "le";
    }

    if (parityLean) {
      const targetParity = parityLean === "chan" ? 0 : 1;
      if (bestRed % 2 !== targetParity) {
        const sameParity = parsed.filter(({ red }) => red % 2 === targetParity);
        if (sameParity.length > 0) bestRed = sameParity[0].red;
      }
    }

    const spread = Number.isFinite(chanPct) && Number.isFinite(lePct)
      ? Math.abs(chanPct - lePct)
      : 0;
    const confidence = Math.min(0.75, 0.25 + spread * 1.2);

    const parityConfidence = Math.min(0.72, confidence + spread * 0.8);
    const reason = `Contrarian: crowd thấp nhất → ${bestRed}. ${parsed.map(({ key, p }) => `${key}=${Math.round(p * 100)}%`).join(", ")}`;
    return { id: "crowd", name, predictedRed: bestRed, confidence, parityConfidence, reason };
  }

  /**
   * BetFlowAnalysis: phân tích bet volume/count để tìm smart money signal.
   * Smart money = high avg bet size (ít người nhưng đặt lớn).
   */
  function betFlowAnalyzer(history, currentRound) {
    const name = "Bet Flow";
    const bets = currentRound && currentRound.bets ? currentRound.bets : null;
    if (!bets) {
      return {
        id: "betflow",
        name,
        predictedRed: 2,
        confidence: 0.1,
        reason: "Không có dữ liệu bets cho round hiện tại.",
      };
    }

    const outcomes = {};
    for (const [key, red] of Object.entries(PCT_KEY_TO_RED)) {
      const b = bets[key];
      if (!b) continue;
      const amount = parseBetStr(b.total_bet);
      const count = parseInt(b.total_count, 10) || 0;
      if (count > 0) {
        outcomes[red] = {
          amount,
          count,
          avgBet: amount / count,
        };
      }
    }

    if (Object.keys(outcomes).length === 0) {
      return {
        id: "betflow",
        name,
        predictedRed: 2,
        confidence: 0.12,
        reason: "Không parse được bets.",
      };
    }

    let bestRed = 2;
    let bestAvg = -1;
    for (const [red, info] of Object.entries(outcomes)) {
      if (info.avgBet > bestAvg) {
        bestAvg = info.avgBet;
        bestRed = Number(red);
      }
    }

    const chanBet = bets.chan ? parseBetStr(bets.chan.total_bet) : 0;
    const leBet = bets.le ? parseBetStr(bets.le.total_bet) : 0;
    const totalParityBet = chanBet + leBet;
    if (totalParityBet > 0) {
      const smartParity = chanBet > leBet ? "chan" : "le";
      const targetParity = smartParity === "chan" ? 0 : 1;
      if (bestRed % 2 !== targetParity) {
        const alt = Object.entries(outcomes)
          .filter(([r]) => Number(r) % 2 === targetParity)
          .sort((a, b) => b[1].avgBet - a[1].avgBet);
        if (alt.length > 0) bestRed = Number(alt[0][0]);
      }
    }

    const allAvgs = Object.values(outcomes).map((o) => o.avgBet);
    const maxAvg = Math.max(...allAvgs);
    const minAvg = Math.min(...allAvgs);
    const skew = maxAvg > 0 ? (maxAvg - minAvg) / maxAvg : 0;
    const confidence = Math.min(0.8, 0.2 + skew * 0.6);

    const detail = Object.entries(outcomes)
      .map(([r, o]) => `${r}r:avg=${Math.round(o.avgBet / 1000)}K`)
      .join(", ");
    const reason = `Smart money (avg bet): ${detail}. Chọn ${bestRed} (skew=${skew.toFixed(2)}).`;
    return { id: "betflow", name, predictedRed: bestRed, confidence, reason };
  }

  const BINOMIAL_PROB = [0.0625, 0.25, 0.375, 0.25, 0.0625];

  /**
   * GapAnalysis: track khoảng cách từ lần cuối mỗi outcome xuất hiện.
   * Outcome "overdue" (gap >> kỳ vọng) được ưu tiên.
   */
  function gapAnalyzer(history) {
    const name = "Gap Analysis";
    const n = history.length;
    if (n < 5) {
      return {
        id: "gap",
        name,
        predictedRed: 2,
        confidence: 0.12,
        reason: "Cần ít nhất 5 phiên cho Gap Analysis.",
      };
    }

    const gap = [Infinity, Infinity, Infinity, Infinity, Infinity];
    for (let i = n - 1; i >= 0; i--) {
      const r = history[i].red;
      if (gap[r] === Infinity) {
        gap[r] = n - 1 - i;
      }
    }
    for (let r = 0; r <= 4; r++) {
      if (gap[r] === Infinity) gap[r] = n;
    }

    const expectedGap = BINOMIAL_PROB.map((p) => (p > 0 ? 1 / p : 100));
    const overdueScore = gap.map((g, r) => g / expectedGap[r]);

    let bestRed = 2;
    let bestScore = -1;
    for (let r = 0; r <= 4; r++) {
      if (overdueScore[r] > bestScore) {
        bestScore = overdueScore[r];
        bestRed = r;
      }
    }

    const confidence = Math.min(0.82, 0.15 + Math.min(bestScore, 4) * 0.15);
    const gapStr = gap.map((g, r) => `${r}r=${g}`).join(", ");
    const reason = `Gap: [${gapStr}]. Overdue=${bestRed} (score=${bestScore.toFixed(2)}, kỳ vọng=${expectedGap[bestRed].toFixed(1)}).`;
    return { id: "gap", name, predictedRed: bestRed, confidence, reason };
  }

  /**
   * ParityRepeat + Conditional Markov (2x2): khai thác autocorrelation parity.
   * Xây Markov chain 2x2 cho chuỗi chan/le, predict parity tiếp theo
   * dựa trên transition probability. Chọn red count có base rate cao nhất
   * trong parity đó.
   */
  function parityRepeatPredictor(history) {
    const name = "Parity Repeat";
    const n = history.length;
    if (n < 2) {
      return {
        id: "parityRepeat",
        name,
        predictedRed: 2,
        confidence: 0.1,
        parityConfidence: 0.1,
        reason: "Chưa đủ lịch sử.",
      };
    }

    const trans = { 0: { 0: 0, 1: 0 }, 1: { 0: 0, 1: 0 } };
    const decay = 0.98;
    for (let i = 1; i < n; i++) {
      const prev = history[i - 1].red % 2;
      const curr = history[i].red % 2;
      const w = Math.pow(decay, n - 1 - i);
      trans[prev][curr] += w;
    }

    const lastParity = history[n - 1].red % 2;
    const toChan = trans[lastParity][0];
    const toLe = trans[lastParity][1];
    const total = toChan + toLe;

    let predictedParity;
    let prob;
    if (total > 0) {
      prob = Math.max(toChan, toLe) / total;
      predictedParity = toChan >= toLe ? 0 : 1;
    } else {
      predictedParity = lastParity;
      prob = 0.52;
    }

    const predictedRed = predictedParity === 0 ? 2 : 3;
    const confidence = Math.min(0.75, 0.2 + (prob - 0.5) * 3);

    const parityConfidence = Math.min(0.85, confidence + 0.15);

    const pLabel = predictedParity === 0 ? "chan" : "le";
    const reason = `Parity Markov 2x2: P(${pLabel}|${lastParity === 0 ? "chan" : "le"})=${(prob * 100).toFixed(1)}%. → ${predictedRed}`;
    return { id: "parityRepeat", name, predictedRed, confidence, parityConfidence, reason };
  }

  /**
   * BayesianPrior: posterior = (count + k*prior) / (n + k).
   * Base rate prior: [0.078, 0.224, 0.338, 0.282, 0.078] (từ data thực).
   * Dùng recent window để cập nhật posterior, chọn red có posterior cao nhất.
   */
  function bayesianPrior(history) {
    const name = "Bayesian Prior";
    const n = history.length;
    const prior = [0.078, 0.224, 0.338, 0.282, 0.078];
    const k = 5;
    const windowSize = Math.min(n, 30);

    if (n === 0) {
      return {
        id: "bayesian",
        name,
        predictedRed: 2,
        confidence: 0.15,
        parityConfidence: 0.15,
        reason: "Không có lịch sử, dùng prior → red=2.",
      };
    }

    const counts = [0, 0, 0, 0, 0];
    for (let i = n - windowSize; i < n; i++) {
      counts[history[i].red]++;
    }

    const posterior = [];
    for (let r = 0; r <= 4; r++) {
      posterior.push((counts[r] + k * prior[r]) / (windowSize + k));
    }

    let bestRed = 2;
    let bestP = -1;
    for (let r = 0; r <= 4; r++) {
      if (posterior[r] > bestP) {
        bestP = posterior[r];
        bestRed = r;
      }
    }

    const entropy = -posterior.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
    const maxEntropy = Math.log2(5);
    const concentration = 1 - entropy / maxEntropy;
    const confidence = Math.min(0.8, 0.15 + concentration * 0.65);

    const chanPost = posterior[0] + posterior[2] + posterior[4];
    const lePost = posterior[1] + posterior[3];
    const parityConc = Math.abs(chanPost - lePost);
    const parityConfidence = Math.min(0.78, 0.2 + parityConc * 1.5 + concentration * 0.3);

    const pStr = posterior.map((p, r) => `${r}=${(p * 100).toFixed(1)}%`).join(", ");
    const reason = `Bayesian(w=${windowSize},k=${k}): [${pStr}] → ${bestRed}`;
    return { id: "bayesian", name, predictedRed: bestRed, confidence, parityConfidence, reason };
  }

  const PREDICTORS = [
    patternMatcher,
    streakAnalyzer,
    markovPredictor,
    markov2Predictor,
    timePatternAnalyzer,
    regressionToMean,
    cauPatternDetector,
    balanceMomentum,
    crowdPercentPredictor,
    parityRepeatPredictor,
    bayesianPrior,
  ];

  /**
   * Ensemble động: W_i ≈ [ α·C_i + (1-α)·H_i ]^β
   * H_i = γ·(trúng vị) + (1-γ)·(trúng chẵn/lẻ).
   * Mặc định: đa khung — H = φ·H_ngắn + (1-φ)·H_dài (exact/parity blend từng khung rồi trộn γ).
   * Truyền opts.hitWindow để chỉ dùng một cửa (tương thích / giảm chi phí).
   */
  const DYNAMIC_ENSEMBLE = {
    ALPHA: 0.55,
    BETA: 1.5,
    HIT_WINDOW: 20,
    HIT_WINDOW_SHORT: 20,
    HIT_WINDOW_LONG: 200,
    HIT_MULTI_PHI: 0.7,
    H_BASELINE: 0.15,
    HIT_BLEND_EXACT: 0.7,
    H_HIT_SHRINK: 2,
    PARITY_HARD_CUTOFF: null,
    TOP_K: 7,
  };

  /**
   * @param {RoundItem[]} h đã normalize
   * @param {number} windowSize
   */
  function calculateRecentHitRates(h, windowSize) {
    const n = h.length;
    const stats = {};
    ALGO_IDS.forEach((id) => {
      stats[id] = { exactHits: 0, parityHits: 0, total: 0 };
    });
    if (n < 2) {
      const out = {};
      ALGO_IDS.forEach((id) => {
        out[id] = {
          exactRate: DYNAMIC_ENSEMBLE.H_BASELINE,
          parityRate: DYNAMIC_ENSEMBLE.H_BASELINE,
          total: 0,
        };
      });
      return out;
    }
    const ws = Math.max(1, Math.floor(windowSize));
    const startT = Math.max(1, n - ws);
    for (let t = startT; t < n; t++) {
      const past = h.slice(0, t);
      const actual = h[t].red;
      for (let p = 0; p < PREDICTORS.length; p++) {
        const pred = PREDICTORS[p](past);
        const id = pred.id;
        const st = stats[id];
        st.total++;
        if (matchExact(pred.predictedRed, actual)) st.exactHits++;
        if (matchType(pred.predictedRed, actual)) st.parityHits++;
      }
    }
    const out = {};
    const b = DYNAMIC_ENSEMBLE.H_BASELINE;
    const k =
      DYNAMIC_ENSEMBLE.H_HIT_SHRINK != null
        ? Math.max(0, Number(DYNAMIC_ENSEMBLE.H_HIT_SHRINK))
        : 0;
    ALGO_IDS.forEach((id) => {
      const s = stats[id];
      let exactRate;
      let parityRate;
      if (s.total > 0 && k > 0) {
        exactRate = (s.exactHits + k * b) / (s.total + k);
        parityRate = (s.parityHits + k * b) / (s.total + k);
      } else if (s.total > 0) {
        exactRate = s.exactHits / s.total;
        parityRate = s.parityHits / s.total;
      } else {
        exactRate = b;
        parityRate = b;
      }
      out[id] = { exactRate, parityRate, total: s.total };
    });
    return out;
  }

  /**
   * Trộn hit rate hai cửa: exact/parity từng khung được blend φ, sau đó vẫn dùng γ trong ensemblePredict.
   * @param {RoundItem[]} h
   * @param {number} windowShort
   * @param {number} windowLong
   * @param {number} phiOnShort
   */
  function mergeMultiTimeframeHitRates(h, windowShort, windowLong, phiOnShort) {
    const ws = Math.max(3, Math.floor(windowShort));
    let wl = Math.max(3, Math.floor(windowLong));
    if (wl < ws) wl = ws;
    const phi = Math.max(0, Math.min(1, phiOnShort));
    const hrS = calculateRecentHitRates(h, ws);
    const hrL = calculateRecentHitRates(h, wl);
    const out = {};
    ALGO_IDS.forEach((id) => {
      const s = hrS[id];
      const l = hrL[id];
      out[id] = {
        exactRate: phi * s.exactRate + (1 - phi) * l.exactRate,
        parityRate: phi * s.parityRate + (1 - phi) * l.parityRate,
        total: Math.max(s.total, l.total),
        totalShort: s.total,
        totalLong: l.total,
        multiTimeframe: true,
      };
    });
    return out;
  }

  /**
   * @param {RoundItem[]} history
   * @param {{
   *   dynamic?: boolean,
   *   hitWindow?: number,
   *   hitWindowShort?: number,
   *   hitWindowLong?: number,
   *   hitMultiPhi?: number,
   *   alpha?: number,
   *   beta?: number
   * }} [opts] dynamic=false → chỉ C^β (tương đương ensemble cố định).
   * Truyền hitWindow → một cửa H (bỏ qua đa khung). Không truyền → H_ngắn + H_dài theo DYNAMIC_ENSEMBLE.
   */

  const REGIME_BOOST = {
    streaky:      { parityRepeat: 1.15, balance: 1.1, cauPattern: 0.92 },
    alternating:  { cauPattern: 1.15, streak: 1.1, parityRepeat: 0.92 },
    random:       { bayesian: 1.1, regression: 1.08 },
  };

  function detectRegime(history) {
    const W = Math.min(20, history.length);
    if (W < 6) return "random";
    const win = history.slice(-W);

    const runs = [];
    let runLen = 1;
    for (let i = 1; i < win.length; i++) {
      if (win[i].type === win[i - 1].type) runLen++;
      else { runs.push(runLen); runLen = 1; }
    }
    runs.push(runLen);

    const avgRun = runs.reduce((a, b) => a + b, 0) / runs.length;
    const shortRuns = runs.filter((r) => r <= 1).length;
    const longRuns = runs.filter((r) => r >= 3).length;

    let sameCount = 0;
    for (let i = 1; i < win.length; i++) {
      if (win[i].type === win[i - 1].type) sameCount++;
    }
    const autocorr = sameCount / (win.length - 1);

    if (autocorr >= 0.58 || avgRun >= 2.5 || longRuns >= runs.length * 0.4) return "streaky";
    if (autocorr <= 0.38 || avgRun <= 1.4 || shortRuns >= runs.length * 0.65) return "alternating";
    return "random";
  }

  function ensemblePredict(history, opts) {
    const h = normalizeHistory(history);
    const o = opts && typeof opts === "object" ? opts : {};
    const useDynamic = o.dynamic !== false;
    const useSingleHitWindow = o.hitWindow != null;
    const windowSize = useSingleHitWindow
      ? Math.max(3, Math.floor(o.hitWindow))
      : null;
    const windowShort =
      o.hitWindowShort != null
        ? Math.max(3, Math.floor(o.hitWindowShort))
        : DYNAMIC_ENSEMBLE.HIT_WINDOW_SHORT;
    const windowLong =
      o.hitWindowLong != null
        ? Math.max(3, Math.floor(o.hitWindowLong))
        : DYNAMIC_ENSEMBLE.HIT_WINDOW_LONG;
    const multiPhi =
      o.hitMultiPhi != null
        ? Math.max(0, Math.min(1, o.hitMultiPhi))
        : DYNAMIC_ENSEMBLE.HIT_MULTI_PHI;
    const alpha =
      o.alpha != null
        ? Math.max(0, Math.min(1, o.alpha))
        : DYNAMIC_ENSEMBLE.ALPHA;
    const beta =
      o.beta != null
        ? Math.max(0.5, Math.min(6, o.beta))
        : DYNAMIC_ENSEMBLE.BETA;
    const gamma = DYNAMIC_ENSEMBLE.HIT_BLEND_EXACT;
    const b = DYNAMIC_ENSEMBLE.H_BASELINE;

    const currentRound = o.currentRound || null;
    const algorithms = PREDICTORS.map((fn) => fn(h, currentRound));
    let hitRates = null;
    if (useDynamic) {
      if (useSingleHitWindow) {
        hitRates = calculateRecentHitRates(h, windowSize);
      } else {
        hitRates = mergeMultiTimeframeHitRates(
          h,
          windowShort,
          windowLong,
          multiPhi,
        );
      }
    }

    const weights = algorithms.map((a) => {
      const c = Math.max(0.05, Math.min(1, a.confidence));
      if (!hitRates) {
        return Math.pow(c, beta);
      }
      const hi = hitRates[a.id];
      const hExact = hi ? hi.exactRate : b;
      const hParity = hi ? hi.parityRate : b;
      const pCut = DYNAMIC_ENSEMBLE.PARITY_HARD_CUTOFF;
      const parityCut =
        pCut != null && Number.isFinite(Number(pCut)) ? Number(pCut) : null;
      if (
        gamma === 0 &&
        parityCut != null &&
        hi &&
        hi.total > 0 &&
        hParity < parityCut
      ) {
        return 0;
      }
      const Hblend = Math.max(
        0.05,
        Math.min(1, gamma * hExact + (1 - gamma) * hParity),
      );
      const dynamicScore = Math.max(
        0.05,
        Math.min(1, alpha * c + (1 - alpha) * Hblend),
      );
      return Math.pow(dynamicScore, beta);
    });

    let finalWeights = weights.slice();
    if (useDynamic && hitRates) {
      const wSum0 = finalWeights.reduce((s, w) => s + w, 0);
      if (wSum0 <= 0) {
        finalWeights = algorithms.map((a) => {
          const c = Math.max(0.05, Math.min(1, a.confidence));
          return Math.pow(c, beta);
        });
      }
    }

    const regime = h.length >= 6 ? detectRegime(h) : "random";
    const boosts = REGIME_BOOST[regime] || {};
    algorithms.forEach((a, i) => {
      if (boosts[a.id]) finalWeights[i] *= boosts[a.id];
    });

    const topK = DYNAMIC_ENSEMBLE.TOP_K;
    if (useDynamic && hitRates && topK != null && topK > 0 && topK < finalWeights.length) {
      const ranked = finalWeights
        .map((w, i) => ({ w, i }))
        .sort((a, b) => b.w - a.w);
      for (let j = topK; j < ranked.length; j++) finalWeights[ranked[j].i] = 0;
    }

    const tally = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
    algorithms.forEach((a, i) => {
      tally[a.predictedRed] += finalWeights[i];
    });
    let predictedRed = 2;
    let maxW = -1;
    for (let r = 0; r <= 4; r++) {
      if (tally[r] > maxW) {
        maxW = tally[r];
        predictedRed = r;
      }
    }

    let wChan = 0;
    let wLe = 0;
    algorithms.forEach((a, i) => {
      if (finalWeights[i] <= 0) return;
      const conf = Math.max(0.05, a.confidence);
      const pConf = a.parityConfidence != null ? Math.max(0.05, a.parityConfidence) : conf;
      const boost = Math.min(1.25, Math.max(0.8, pConf / conf));
      const pw = finalWeights[i] * boost;
      if (a.predictedRed % 2 === 0) wChan += pw;
      else wLe += pw;
    });
    const wParSum = wChan + wLe;
    const predictedParity = wChan >= wLe ? "chan" : "le";

    const targetMod = predictedParity === "chan" ? 0 : 1;
    if (predictedRed % 2 !== targetMod) {
      let bestR = targetMod === 0 ? 2 : 3;
      let bestW = -1;
      for (let r = 0; r <= 4; r++) {
        if (r % 2 === targetMod && tally[r] > bestW) {
          bestW = tally[r];
          bestR = r;
        }
      }
      predictedRed = bestR;
    }

    const agree = algorithms.filter(
      (a) => a.predictedRed === predictedRed,
    ).length;
    const consensus = algorithms.length ? agree / algorithms.length : 0;

    const parityConfidence =
      wParSum > 0 ? Math.min(0.97, Math.max(wChan, wLe) / wParSum) : 0.1;
    const parityAgree = algorithms.filter(
      (a) => (a.predictedRed % 2 === 0) === (predictedParity === "chan"),
    ).length;
    const parityConsensus = algorithms.length
      ? parityAgree / algorithms.length
      : 0;
    const parityMismatch = predictedRed % 2 !== targetMod;

    let wConf = 0;
    let wSum = 0;
    algorithms.forEach((a, i) => {
      wConf += a.confidence * finalWeights[i];
      wSum += finalWeights[i];
    });
    const confidence = wSum > 0 ? Math.min(0.97, wConf / wSum) : 0.1;


    let topIdx = 0;
    for (let i = 1; i < finalWeights.length; i++) {
      if (finalWeights[i] > finalWeights[topIdx]) topIdx = i;
    }
    const topAlgo = algorithms[topIdx];
    let weightedReason = "";
    if (topAlgo) {
      const pctConf = Math.round(topAlgo.confidence * 100);
      if (hitRates && hitRates[topAlgo.id]) {
        const hi = hitRates[topAlgo.id];
        const hComb = Math.round(
          100 * (gamma * hi.exactRate + (1 - gamma) * hi.parityRate),
        );
        const hasN =
          hi.total > 0 ||
          (hi.multiTimeframe &&
            ((hi.totalShort != null && hi.totalShort > 0) ||
              (hi.totalLong != null && hi.totalLong > 0)));
        if (hasN && hi.multiTimeframe) {
          const ns = hi.totalShort != null ? hi.totalShort : 0;
          const nl = hi.totalLong != null ? hi.totalLong : 0;
          weightedReason = `${topAlgo.name} trọng số động cao nhất (conf ${pctConf}%, H_gộp≈${hComb}%, n_ngắn=${ns}, n_dài=${nl}): ${topAlgo.reason}`;
        } else if (hi.total > 0) {
          weightedReason = `${topAlgo.name} trọng số động cao nhất (conf ${pctConf}%, H_gộp≈${hComb}%, n=${hi.total}): ${topAlgo.reason}`;
        } else {
          weightedReason = `${topAlgo.name} trọng số động (conf ${pctConf}%, H≈baseline): ${topAlgo.reason}`;
        }
      } else {
        weightedReason = `${topAlgo.name} mạnh nhất (${pctConf}%): ${topAlgo.reason}`;
      }
    }

    const parityMargin = wParSum > 0 ? Math.abs(wChan - wLe) / wParSum : 0;

    const bc = parityConsensus * 0.55 + parityMargin * 0.30 + confidence * 0.15;
    const betConfVal = Math.min(0.99, bc);
    const threshold = o.betThreshold != null ? o.betThreshold : 0.70;
    const shouldBet = parityConsensus >= 8 / algorithms.length && parityMargin >= 0.4;
    const betReason = shouldBet
      ? `${parityAgree}/${algorithms.length} đồng thuận, margin ${(parityMargin * 100).toFixed(0)}% — ${regime}`
      : `Chưa đủ: ${parityAgree}/${algorithms.length} đồng thuận, margin ${(parityMargin * 100).toFixed(0)}%`;

    return {
      predictedRed,
      predictedParity,
      parityConfidence,
      parityConsensus,
      parityConsensusCount: parityAgree,
      parityMismatch,
      parityMargin,
      regime,
      outcome: getOutcomeMeta(predictedRed),
      confidence,
      consensus,
      consensusCount: agree,
      algorithmCount: algorithms.length,
      weightedReason,
      algorithms,
      ensembleWeights: finalWeights,
      shouldBet,
      betConfidence: betConfVal,
      betReason,
    };
  }

  /**
   * Walk-forward backtest: ensemble động (mặc định) và ensemble tĩnh (chỉ C^β, opts giống dynamic).
   * @param {RoundItem[]} history full chronological
   * @param {{ burnIn?: number }} opts
   */
  function runBacktest(history, opts) {
    const h = normalizeHistory(history);
    const burnIn = opts && opts.burnIn != null ? opts.burnIn : 40;
    const perAlgo = {};
    ALGO_IDS.forEach((id) => {
      perAlgo[id] = { exact: 0, type: 0, n: 0 };
    });
    const ensembleDyn = { exact: 0, type: 0, n: 0 };
    const ensembleStatic = { exact: 0, type: 0, n: 0 };
    let agreeRed = 0;
    let differRed = 0;
    let dynWinsExactWhenDiffer = 0;
    let statWinsExactWhenDiffer = 0;

    for (let t = burnIn; t < h.length; t++) {
      const past = h.slice(0, t);
      const actual = h[t].red;
      const ensD = ensemblePredict(past);
      const ensS = ensemblePredict(past, { dynamic: false });
      ensembleDyn.n++;
      if (matchExact(ensD.predictedRed, actual)) ensembleDyn.exact++;
      if (matchType(ensD.predictedRed, actual)) ensembleDyn.type++;

      ensembleStatic.n++;
      if (matchExact(ensS.predictedRed, actual)) ensembleStatic.exact++;
      if (matchType(ensS.predictedRed, actual)) ensembleStatic.type++;

      if (ensD.predictedRed === ensS.predictedRed) {
        agreeRed++;
      } else {
        differRed++;
        const dOk = matchExact(ensD.predictedRed, actual);
        const sOk = matchExact(ensS.predictedRed, actual);
        if (dOk && !sOk) dynWinsExactWhenDiffer++;
        if (sOk && !dOk) statWinsExactWhenDiffer++;
      }

      ensD.algorithms.forEach((a) => {
        const st = perAlgo[a.id];
        st.n++;
        if (matchExact(a.predictedRed, actual)) st.exact++;
        if (matchType(a.predictedRed, actual)) st.type++;
      });
    }

    const pct = (x, n) => (n > 0 ? Math.round((10000 * x) / n) / 100 : 0);
    const summarize = (st) => ({
      exact: st.exact,
      type: st.type,
      n: st.n,
      exactPct: pct(st.exact, st.n),
      typePct: pct(st.type, st.n),
    });

    const byAlgo = {};
    ALGO_IDS.forEach((id) => {
      byAlgo[id] = summarize(perAlgo[id]);
    });

    const steps = Math.max(0, h.length - burnIn);
    const sumD = summarize(ensembleDyn);
    const sumS = summarize(ensembleStatic);
    const modeComparison = {
      steps,
      sameRedCount: agreeRed,
      differRedCount: differRed,
      sameRedPct: steps > 0 ? Math.round((10000 * agreeRed) / steps) / 100 : 0,
      dynWinsExactWhenDiffer,
      statWinsExactWhenDiffer,
      exactPctDelta:
        sumD.n > 0
          ? Math.round((sumD.exactPct - sumS.exactPct) * 100) / 100
          : 0,
      typePctDelta:
        sumD.n > 0 ? Math.round((sumD.typePct - sumS.typePct) * 100) / 100 : 0,
    };

    return {
      burnIn,
      totalSteps: steps,
      ensemble: sumD,
      ensembleStatic: sumS,
      modeComparison,
      byAlgo,
    };
  }

  /** Baselines on same indices as backtest */
  function runBaselines(history, opts) {
    const h = normalizeHistory(history);
    const burnIn = opts && opts.burnIn != null ? opts.burnIn : 40;
    let randExact = 0,
      randType = 0;
    let patExact = 0,
      patType = 0;
    const n = h.length - burnIn;
    if (n <= 0) {
      return {
        random: { exactPct: 0, typePct: 0, n: 0 },
        lastRepeats: { exactPct: 0, typePct: 0, n: 0 },
      };
    }

    for (let t = burnIn; t < h.length; t++) {
      const actual = h[t].red;
      const rnd = Math.floor(Math.random() * 5);
      if (rnd === actual) randExact++;
      if (rnd % 2 === actual % 2) randType++;

      const prev = h[t - 1].red;
      if (prev === actual) patExact++;
      if (prev % 2 === actual % 2) patType++;
    }

    const pct = (x) => Math.round((10000 * x) / n) / 100;
    return {
      random: { exactPct: pct(randExact), typePct: pct(randType), n },
      lastRepeats: { exactPct: pct(patExact), typePct: pct(patType), n },
    };
  }

  const XocDiaPrediction = {
    ALGO_IDS,
    OUTCOME_META_VI,
    getOutcomeMeta,
    normalizeHistory,
    patternMatcher,
    streakAnalyzer,
    frequencyBalancer,
    markovPredictor,
    markov2Predictor,
    hotColdAnalyzer,
    timePatternAnalyzer,
    entropyAnalyzer,
    regressionToMean,
    cauPatternDetector,
    balanceMomentum,
    crowdPercentPredictor,
    betFlowAnalyzer,
    gapAnalyzer,
    parityRepeatPredictor,
    bayesianPrior,
    parsePct,
    parseBetStr,
    ensemblePredict,
    runBacktest,
    runBaselines,
    matchExact,
    matchType,
    PREDICTORS,
    DYNAMIC_ENSEMBLE,
    calculateRecentHitRates,
    detectRegime,
  };

  global.XocDiaPrediction = XocDiaPrediction;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = XocDiaPrediction;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
