// Xóc Đĩa Analytics + Prediction — mở rộng app.js với panel dự đoán,
// theo dõi độ chính xác, backtest, cảnh báo và export.

(() => {
  "use strict";

  const POLL_MS = 3000;
  const ALERT_CONF = 0.8;
  const ALERT_CONSENSUS = 0.83;
  const COMPARE_LS_KEY = "xocdia_compare_history_v1";
  const COMPARE_WINDOW_LS = "xocdia_compare_stats_window_v1";
  /** Ngưỡng an toàn cho input N (cửa sổ %) — không cắt lịch sử khi lưu. */
  const STATS_N_MAX = Number.MAX_SAFE_INTEGER;

  const DICE_TO_RED = {
    "4_red": 4,
    "3r_1w": 3,
    "2w_2r": 2,
    "3w_1r": 1,
    "4_white": 0,
  };

  const VI_LABELS = [
    "4 TRẮNG",
    "3 TRẮNG 1 ĐỎ",
    "2 TRẮNG 2 ĐỎ",
    "3 ĐỎ 1 TRẮNG",
    "4 ĐỎ",
  ];

  const ALGO_LABEL_VI = {
    pattern: "Pattern Matching",
    streak: "Streak Analysis",
    markov: "Markov Chain",
    markov2: "Markov Bậc 2",
    time: "Time Pattern",
    regression: "Regression to Mean",
    cauPattern: "Cầu Pattern",
    balance: "Balance Momentum",
    crowd: "Contrarian Crowd",
    parityRepeat: "Parity Repeat",
    bayesian: "Bayesian Prior",
  };

  /** @type {Array<{red:number,type:"chan"|"le",time:Date,round_id:string}>} */
  let masterData = [];
  let autoFollow = true;

  /** Round đang betting (dice_result=null), raw JSON từ server */
  let currentInProgress = null;

  /** Dự đoán ensemble lần trước (trên toàn bộ masterData cũ) để so phiên mới */
  let lastFullEnsemble = null;
  /** Cùng thời điểm, ensemble chỉ C^β (tĩnh) — để lưu vào lịch sử so sánh */
  let lastFullEnsembleStatic = null;

  const accuracy = {
    recent: [],
    maxRecent: 30,
    total: { exact: 0, type: 0, n: 0 },
    selective: { exact: 0, type: 0, n: 0, bet: 0 },
    byAlgo: {},
  };

  [
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
  ].forEach((id) => {
    accuracy.byAlgo[id] = { exact: 0, type: 0, n: 0 };
  });

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** 4 vòng tròn theo số mặt đỏ 0–4 (còn lại là trắng), thứ tự: trắng trước, đỏ sau — dễ đọc nhanh. */
  function boardPatternHtml(red) {
    if (red === undefined || red === null) return "";
    const n = Number(red);
    if (!Number.isFinite(n)) return "";
    const r = Math.max(0, Math.min(4, n | 0));
    const white = 4 - r;
    const title = `${white} Trắng · ${r} Đỏ`;
    const dots = [];
    for (let i = 0; i < white; i++) {
      dots.push(
        '<span class="board-dot board-dot-white" aria-hidden="true"></span>',
      );
    }
    for (let i = 0; i < r; i++) {
      dots.push(
        '<span class="board-dot board-dot-red" aria-hidden="true"></span>',
      );
    }
    return `<span class="board-pattern" title="${escapeHtml(title)}" role="img" aria-label="${escapeHtml(title)}">${dots.join("")}</span>`;
  }

  function algoPredictionsFromEnsemble(P, ensemble) {
    if (!P || !ensemble || !Array.isArray(ensemble.algorithms)) return [];
    const w = ensemble.ensembleWeights;
    return ensemble.algorithms.map((a, i) => ({
      id: a.id,
      red: a.predictedRed,
      door: P.getOutcomeMeta(a.predictedRed).short,
      confidence:
        typeof a.confidence === "number" && Number.isFinite(a.confidence)
          ? a.confidence
          : 0,
      weight:
        w && typeof w[i] === "number" && Number.isFinite(w[i])
          ? w[i]
          : undefined,
    }));
  }

  /** TT nặng ký nhất sau ensemble động (có weight); bản ghi cũ → max confidence. */
  function ensembleLeaderId(predByAlgo) {
    if (!predByAlgo || !predByAlgo.length) return null;
    const hasW = predByAlgo.some(
      (x) => typeof x.weight === "number" && x.weight > 0,
    );
    if (hasW) {
      let best = predByAlgo[0];
      let bestW = best.weight != null ? best.weight : -1;
      for (let i = 1; i < predByAlgo.length; i++) {
        const w = predByAlgo[i].weight != null ? predByAlgo[i].weight : -1;
        if (w > bestW) {
          bestW = w;
          best = predByAlgo[i];
        }
      }
      return best.id;
    }
    let best = predByAlgo[0];
    let bestC = best.confidence != null ? best.confidence : 0;
    for (let i = 1; i < predByAlgo.length; i++) {
      const c = predByAlgo[i].confidence != null ? predByAlgo[i].confidence : 0;
      if (c > bestC) {
        bestC = c;
        best = predByAlgo[i];
      }
    }
    return bestC > 0 ? best.id : null;
  }

  /**
   * @param {Array<{id:string,red:number,door:string,confidence?:number,weight?:number}>} predByAlgo
   * @param {{ actualRed: number, ensemblePredRed?: number }} ctx
   * @param {{ defaultOpen?: boolean }} opts
   */
  function renderCompareAlgoSectionHtml(predByAlgo, ctx, opts) {
    const { actualRed } = ctx || {};
    const defaultOpen = opts && opts.defaultOpen === true;
    if (!predByAlgo || !predByAlgo.length) {
      return `<p class="text-[9px] text-slate-600 mt-2 leading-relaxed">Chưa có chi tiết thuật toán (bản ghi cũ trước khi lưu thêm cột này).</p>`;
    }
    const leaderId = ensembleLeaderId(predByAlgo);
    const rows = predByAlgo
      .map((a) => {
        const label = escapeHtml(ALGO_LABEL_VI[a.id] || a.id);
        const parity = a.red % 2 === 0 ? "Chẵn" : "Lẻ";
        const dots = boardPatternHtml(a.red);
        const hitActual = actualRed != null && a.red === actualRed;
        const isLeader = leaderId != null && a.id === leaderId;
        let rowCls =
          "flex flex-col gap-0.5 rounded px-1 py-1 -mx-0.5 sm:flex-row sm:flex-wrap sm:items-baseline sm:justify-between ";
        if (isLeader) {
          rowCls +=
            "border-l-2 border-violet-400 bg-violet-500/20 text-violet-50 ring-1 ring-inset ring-violet-500/25";
        } else {
          rowCls += "text-slate-400";
        }
        const badges = [];
        if (isLeader) {
          badges.push(
            `<span class="text-[8px] font-black uppercase text-violet-300 shrink-0" title="Trọng số ensemble động cao nhất (conf + phong độ gần đây)">★ trọng số động</span>`,
          );
          if (hitActual) {
            badges.push(
              `<span class="text-[8px] font-black uppercase text-emerald-400/95 shrink-0">trùng kết quả</span>`,
            );
          }
        }
        const badgeHtml = badges.length
          ? `<span class="inline-flex flex-wrap gap-x-1 justify-end sm:pl-2">${badges.join("")}</span>`
          : "";
        return `<div class="${rowCls}">
          <span class="text-[10px] font-medium truncate min-w-0 max-w-full sm:max-w-[48%]" title="${label}">${label}</span>
          <span class="inline-flex flex-wrap items-baseline gap-x-1 gap-y-0 justify-start sm:justify-end text-[10px] min-w-0 flex-1">
            <span class="font-semibold whitespace-nowrap">${parity}</span><span class="text-slate-600 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px] shrink-0">${dots}</span>
            ${badgeHtml}
          </span>
        </div>`;
      })
      .join("");
    const panelHidden = defaultOpen ? "" : " hidden";
    const ariaEx = defaultOpen ? "true" : "false";
    const btnLabel = defaultOpen ? "Thu gọn" : "Mở rộng";
    return `<div class="compare-algo-block mt-2 rounded-lg border border-slate-800/90 bg-slate-950/50 overflow-hidden">
      <div class="flex items-center justify-between gap-2 px-2 py-1.5 bg-slate-900/50 border-b border-slate-800/80">
        <span class="text-[9px] font-black text-slate-500 uppercase tracking-wide">Theo từng thuật toán</span>
        <button type="button" class="compare-algo-toggle shrink-0 text-[9px] font-black text-sky-400 hover:text-sky-300 uppercase tracking-wide px-2 py-0.5 rounded-md border border-sky-500/40 bg-sky-950/40" aria-expanded="${ariaEx}">${btnLabel}</button>
      </div>
      <div class="compare-algo-panel px-2 py-1.5${panelHidden}">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-3 gap-y-0.5">${rows}</div>
        <p class="text-[8px] text-slate-500 mt-2 leading-snug border-t border-slate-800/80 pt-1.5"><b class="text-violet-400/90">★</b> Thuật toán có <b>trọng số ensemble động</b> cao nhất (α·conf + phong độ H trong cửa sổ). <b class="text-emerald-400/80">Trùng kết quả</b> chỉ trên dòng đó khi cửa TT khớp thực tế.</p>
      </div>
    </div>`;
  }

  /** @param {{ predByAlgo?: Array<{ id: string, red: number }> } | null} row */
  function getAlgoPredFromRow(row, algoId) {
    if (!row || !algoId || !row.predByAlgo || !Array.isArray(row.predByAlgo))
      return null;
    return row.predByAlgo.find((x) => x && x.id === algoId) || null;
  }

  function getAlgoIdsForSelect() {
    const P = window.XocDiaPrediction;
    if (P && Array.isArray(P.ALGO_IDS) && P.ALGO_IDS.length) return P.ALGO_IDS;
    return [
      "pattern",
      "streak",
      "frequency",
      "markov",
      "hotcold",
      "time",
      "sentiment",
      "crowd",
    ];
  }

  function ensureAlgoHistorySelectOptions() {
    const sel = document.getElementById("algoHistorySelect");
    if (!sel) return;
    const ids = getAlgoIdsForSelect();
    const needRebuild =
      sel.options.length !== ids.length ||
      !ids.every((id, i) => sel.options[i] && sel.options[i].value === id);
    if (!needRebuild) return;
    const prev = sel.value;
    sel.innerHTML = ids
      .map((id) => {
        const lab = ALGO_LABEL_VI[id] || id;
        return `<option value="${escapeHtml(id)}">${escapeHtml(lab)}</option>`;
      })
      .join("");
    if (prev && ids.includes(prev)) sel.value = prev;
  }

  /**
   * Chuẩn hóa «Số dòng»: không giới hạn trên (tối đa STATS_N_MAX).
   * Để trống hoặc 0 → dùng toàn bộ lịch sử có predByAlgo (practical: hết mảng).
   */
  function parseAlgoHistoryLimit(limInput) {
    const raw =
      limInput && limInput.value != null ? String(limInput.value).trim() : "";
    if (raw === "" || raw === "0") {
      if (limInput) limInput.value = "";
      return STATS_N_MAX;
    }
    let maxN = parseInt(raw, 10);
    if (!Number.isFinite(maxN) || maxN < 1) {
      maxN = 50;
      if (limInput) limInput.value = String(maxN);
      return maxN;
    }
    maxN = Math.min(STATS_N_MAX, maxN);
    if (limInput) limInput.value = String(maxN);
    return maxN;
  }

  function formatAlgoHistoryCapPhrase(maxN) {
    if (maxN >= STATS_N_MAX) return "toàn bộ lịch sử (không giới hạn số dòng)";
    return `tối đa ${maxN.toLocaleString("vi-VN")} lượt dự đoán gần nhất`;
  }

  /** Overview % đúng/sai CL và Vị cho từng TT (cùng ngưỡng N như bảng chi tiết). */
  function renderAlgoHistoryOverview(maxN) {
    const el = document.getElementById("algoHistoryOverview");
    if (!el) return;
    if (!Number.isFinite(maxN) || maxN < 1) {
      const limInput = document.getElementById("algoHistoryLimit");
      maxN = parseAlgoHistoryLimit(limInput);
    }
    const list = loadCompareHistory();
    const ids = getAlgoIdsForSelect();

    if (!list.length) {
      el.innerHTML =
        '<p class="text-[9px] text-slate-500 leading-relaxed">Chưa có lịch sử local — chưa tính được tỉ lệ theo thuật toán.</p>';
      return;
    }

    const cards = ids
      .map((id) => {
        let ex = 0,
          ty = 0,
          n = 0;
        for (let i = 0; i < list.length && n < maxN; i++) {
          const e = list[i];
          const p = getAlgoPredFromRow(e, id);
          if (!p) continue;
          n++;
          if (p.red === e.actualRed) ex++;
          if (p.red % 2 === e.actualRed % 2) ty++;
        }
        const lab = escapeHtml(ALGO_LABEL_VI[id] || id);
        if (!n) {
          return `<div class="rounded-lg border border-slate-800/80 bg-slate-950/55 px-2 py-2 min-w-0">
          <div class="text-[9px] font-black text-slate-500 truncate" title="${lab}">${lab}</div>
          <div class="text-[8px] text-slate-600 mt-1">Không có dữ liệu</div>
        </div>`;
        }
        const tyPct = Math.round((100 * ty) / n);
        const exPct = Math.round((100 * ex) / n);
        const tyWrong = 100 - tyPct;
        const exWrong = 100 - exPct;
        const st = maxAlgoHistoryStreaks(list, id);
        return `<div class="rounded-lg border border-slate-800/80 bg-slate-950/55 px-2 py-2 min-w-0">
        <div class="text-[9px] font-black text-indigo-300/95 truncate" title="${lab}">${lab}</div>
        <div class="mt-1.5 space-y-1 text-[9px] leading-snug tabular-nums">
          <div><span class="text-slate-500 font-bold uppercase tracking-tighter">CL</span> <span class="text-cyan-400 font-black">${tyPct}%</span><span class="text-slate-600"> đúng</span> <span class="text-slate-600">·</span> <span class="text-red-400/90 font-semibold">${tyWrong}%</span><span class="text-slate-600"> sai</span></div>
          <div><span class="text-slate-500 font-bold uppercase tracking-tighter">Vị</span> <span class="text-emerald-400 font-black">${exPct}%</span><span class="text-slate-600"> đúng</span> <span class="text-slate-600">·</span> <span class="text-red-400/90 font-semibold">${exWrong}%</span><span class="text-slate-600"> sai</span></div>
          <div class="text-slate-600 font-mono text-[8px]">n=${n}</div>
        </div>
        <div class="mt-2 pt-1.5 border-t border-slate-800/70">
          <div class="text-[8px] font-black text-slate-500 uppercase tracking-wide">Chuỗi đúng dài nhất</div>
          <div class="text-[8px] text-slate-600 leading-tight mt-0.5">(toàn bộ lịch sử · thứ tự thời gian)</div>
          <div class="mt-1 space-y-0.5 text-[9px] tabular-nums leading-snug">
            <div><span class="text-slate-500 font-medium">Chẵn/Lẻ:</span> <span class="text-cyan-400 font-black">${st.parity.toLocaleString("vi-VN")} phiên</span></div>
            <div><span class="text-slate-500 font-medium">Vị:</span> <span class="text-emerald-400 font-black">${st.exact.toLocaleString("vi-VN")} phiên</span></div>
          </div>
        </div>
      </div>`;
      })
      .join("");

    el.innerHTML = `<div class="text-[8px] text-slate-500 mb-2 leading-relaxed"><span class="font-black text-slate-400 uppercase tracking-wide">Tổng quan</span> — ${formatAlgoHistoryCapPhrase(maxN)} có <code class="text-slate-600">predByAlgo</code> / TT (theo «Số dòng»). <span class="text-slate-600">Chuỗi đúng dài nhất: toàn bộ lịch sử, thứ tự cũ → mới (bỏ qua phiên không có dự đoán TT).</span></div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-2">${cards}</div>`;
  }

  /** Bảng lịch sử dự đoán một TT (từ predByAlgo trong lịch sử local). */
  function renderAlgoHistoryPanel() {
    const limInput = document.getElementById("algoHistoryLimit");
    const maxN = parseAlgoHistoryLimit(limInput);
    renderAlgoHistoryOverview(maxN);

    const tbody = document.getElementById("algoHistoryBody");
    const summary = document.getElementById("algoHistorySummary");
    const sel = document.getElementById("algoHistorySelect");
    if (!tbody || !sel) return;
    ensureAlgoHistorySelectOptions();

    const algoId = sel.value;

    const list = loadCompareHistory();
    const rows = [];
    for (let i = 0; i < list.length && rows.length < maxN; i++) {
      const e = list[i];
      const p = getAlgoPredFromRow(e, algoId);
      if (p) rows.push({ e, p });
    }

    let ex = 0,
      ty = 0;
    const n = rows.length;
    rows.forEach(({ e, p }) => {
      if (p.red === e.actualRed) ex++;
      if (p.red % 2 === e.actualRed % 2) ty++;
    });

    if (summary) {
      if (!n) {
        summary.textContent = !list.length
          ? "Chưa có lịch sử local — chờ phiên mới."
          : `Không có dữ liệu TT «${ALGO_LABEL_VI[algoId] || algoId}» trong ${list.length} mục (bản ghi cũ trước khi lưu predByAlgo?).`;
      } else {
        const exPct = Math.round((100 * ex) / n);
        const tyPct = Math.round((100 * ty) / n);
        summary.textContent = `${n} phiên (mới nhất trước) · Vị ${ex}/${n} (${exPct}%) · CL ${ty}/${n} (${tyPct}%)`;
      }
    }

    if (!n) {
      tbody.innerHTML =
        '<tr><td colspan="5" class="py-5 px-2 text-center text-slate-500 leading-relaxed">—</td></tr>';
      return;
    }

    tbody.innerHTML = rows
      .map(({ e, p }) => {
        const parityHit = p.red % 2 === e.actualRed % 2;
        const exactHit = p.red === e.actualRed;
        const pl = p.red % 2 === 0 ? "Chẵn" : "Lẻ";
        const al = e.actualRed % 2 === 0 ? "Chẵn" : "Lẻ";
        const clMark = parityHit ? "✓" : "✗";
        const viMark = exactHit ? "✓" : "✗";
        const clCls = parityHit ? "text-cyan-400" : "text-red-400";
        const viCls = exactHit ? "text-emerald-400" : "text-red-400";
        const pDots = boardPatternHtml(p.red);
        const aDots = boardPatternHtml(e.actualRed);
        const conf =
          typeof p.confidence === "number" && Number.isFinite(p.confidence)
            ? p.confidence.toFixed(2)
            : "—";
        return `<tr class="border-b border-slate-800/70 last:border-0 hover:bg-slate-900/40">
          <td class="py-1.5 pl-2 pr-1 align-middle font-mono text-slate-400 whitespace-nowrap">${escapeHtml(e.round_id)}</td>
          <td class="py-1.5 px-1 align-middle">
            <div class="flex flex-wrap items-center gap-x-1 text-[10px] leading-tight">
              <span>${pl}</span><span class="text-slate-600 select-none">·</span><span class="inline-flex items-center translate-y-[0.5px]">${pDots}</span>
              <span class="text-slate-500 font-mono text-[9px]" title="Độ tin cậy TT">(${conf})</span>
            </div>
          </td>
          <td class="py-1.5 px-1 align-middle">
            <div class="flex flex-wrap items-center gap-x-1 text-[10px] leading-tight text-white">
              <span>${al}</span><span class="text-slate-600 select-none">·</span><span class="inline-flex items-center translate-y-[0.5px]">${aDots}</span>
            </div>
          </td>
          <td class="py-1.5 px-1 text-center align-middle font-black ${clCls}">${clMark}</td>
          <td class="py-1.5 pr-2 pl-1 text-center align-middle font-black ${viCls}">${viMark}</td>
        </tr>`;
      })
      .join("");
  }

  function loadCompareHistory() {
    try {
      const raw = localStorage.getItem(COMPARE_LS_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (_) {
      return [];
    }
  }

  function saveCompareHistory(arr) {
    try {
      localStorage.setItem(COMPARE_LS_KEY, JSON.stringify(arr));
    } catch (_) {
      /* quota / private mode */
    }
  }

  function getCompareStatsWindowN() {
    const raw = localStorage.getItem(COMPARE_WINDOW_LS);
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 1 && n <= STATS_N_MAX) return n;
    return 100;
  }

  function setCompareStatsWindowN(n) {
    try {
      localStorage.setItem(COMPARE_WINDOW_LS, String(n));
    } catch (_) {
      /* ignore */
    }
  }

  /** Đúng Chẵn/Lẻ theo cửa ensemble (predRed), không theo tín hiệu CL tổng hợp cũ có thể lệch cửa. */
  function compareHistoryParityMatches(e) {
    if (e.predRed != null && e.actualRed != null)
      return e.predRed % 2 === e.actualRed % 2;
    return !!e.parityOk;
  }

  function compareHistoryHasStaticPred(e) {
    return e.predRedStatic != null && Number.isFinite(Number(e.predRedStatic));
  }

  function compareHistoryStaticExactOk(e) {
    if (!compareHistoryHasStaticPred(e) || e.actualRed == null) return false;
    if (e.staticExactOk != null) return !!e.staticExactOk;
    return Number(e.predRedStatic) === e.actualRed;
  }

  function compareHistoryStaticParityOk(e) {
    if (!compareHistoryHasStaticPred(e) || e.actualRed == null) return false;
    if (e.staticParityOk != null) return !!e.staticParityOk;
    return Number(e.predRedStatic) % 2 === e.actualRed % 2;
  }

  /**
   * Chuỗi dự đoán đúng liên tiếp dài nhất trên toàn bộ lịch sử local (thời gian cũ → mới).
   * @param {Array<{ predRed?: number, actualRed?: number, parityOk?: boolean, exactOk?: boolean }>} list newest-first như loadCompareHistory
   * @returns {{ parity: number, exact: number }}
   */
  function maxComparePredictionStreaks(list) {
    if (!list || !list.length) return { parity: 0, exact: 0 };
    let maxP = 0,
      curP = 0,
      maxE = 0,
      curE = 0;
    for (let i = list.length - 1; i >= 0; i--) {
      const e = list[i];
      if (compareHistoryParityMatches(e)) {
        curP++;
        if (curP > maxP) maxP = curP;
      } else curP = 0;
      if (e.exactOk) {
        curE++;
        if (curE > maxE) maxE = curE;
      } else curE = 0;
    }
    return { parity: maxP, exact: maxE };
  }

  /**
   * Chuỗi đúng CL / đúng vị liên tiếp dài nhất cho một TT.
   * Lịch sử newest-first; duyệt cũ → mới. Bỏ qua phiên không có predByAlgo cho TT (không reset chuỗi).
   * @param {ReturnType<typeof loadCompareHistory>} list
   * @param {string} algoId
   * @returns {{ parity: number, exact: number }}
   */
  function maxAlgoHistoryStreaks(list, algoId) {
    if (!list || !list.length || !algoId) return { parity: 0, exact: 0 };
    let maxP = 0,
      curP = 0,
      maxE = 0,
      curE = 0;
    for (let i = list.length - 1; i >= 0; i--) {
      const e = list[i];
      const p = getAlgoPredFromRow(e, algoId);
      if (!p) continue;
      if (p.red % 2 === e.actualRed % 2) {
        curP++;
        if (curP > maxP) maxP = curP;
      } else curP = 0;
      if (p.red === e.actualRed) {
        curE++;
        if (curE > maxE) maxE = curE;
      } else curE = 0;
    }
    return { parity: maxP, exact: maxE };
  }

  /** Cập nhật % chính xác: N phiên mới nhất trong lịch sử local (input = ngưỡng tối đa). */
  function updateCompareStatsPanel() {
    const input = document.getElementById("compareStatsN");
    const list = loadCompareHistory();
    let want = parseInt(input && input.value, 10);
    if (!Number.isFinite(want) || want < 1) want = getCompareStatsWindowN();
    want = Math.max(1, Math.min(want, STATS_N_MAX));

    const effN = list.length === 0 ? 0 : Math.min(want, list.length);
    const slice = effN > 0 ? list.slice(0, effN) : [];
    const ex = slice.filter((e) => e.exactOk).length;
    const ty = slice.filter((e) => compareHistoryParityMatches(e)).length;
    const len = slice.length;
    const exPct = len ? Math.round((100 * ex) / len) : null;
    const tyPct = len ? Math.round((100 * ty) / len) : null;

    const staticSlice = slice.filter(compareHistoryHasStaticPred);
    const staticLen = staticSlice.length;
    const exS = staticSlice.filter(compareHistoryStaticExactOk).length;
    const tyS = staticSlice.filter(compareHistoryStaticParityOk).length;
    const exStaticPct = staticLen ? Math.round((100 * exS) / staticLen) : null;
    const tyStaticPct = staticLen ? Math.round((100 * tyS) / staticLen) : null;

    const pct = (v) => (v == null ? "—" : `${v}%`);
    const bar = (v) => (v == null ? "0%" : `${Math.min(100, v)}%`);

    const elDynEx = document.getElementById("compareStatsDynExactPct");
    const elDynTy = document.getElementById("compareStatsDynParityPct");
    const barDynEx = document.getElementById("compareStatsDynExactBar");
    const barDynTy = document.getElementById("compareStatsDynParityBar");
    const elStEx = document.getElementById("compareStatsStaticExactPct");
    const elStTy = document.getElementById("compareStatsStaticParityPct");
    const barStEx = document.getElementById("compareStatsStaticExactBar");
    const barStTy = document.getElementById("compareStatsStaticParityBar");
    const cap = document.getElementById("compareStatsCaption");
    const cnt = document.getElementById("compareHistoryCount");
    const subN = document.getElementById("compareStatsEffN");
    const subNStatic = document.getElementById("compareStatsStaticEffN");

    if (elDynEx) elDynEx.textContent = pct(exPct);
    if (elDynTy) elDynTy.textContent = pct(tyPct);
    if (barDynEx) barDynEx.style.width = bar(exPct);
    if (barDynTy) barDynTy.style.width = bar(tyPct);
    if (elStEx) elStEx.textContent = pct(exStaticPct);
    if (elStTy) elStTy.textContent = pct(tyStaticPct);
    if (barStEx) barStEx.style.width = bar(exStaticPct);
    if (barStTy) barStTy.style.width = bar(tyStaticPct);
    if (cnt) cnt.textContent = `${list.length.toLocaleString("vi-VN")} mục`;
    if (subN)
      subN.textContent = len > 0 ? `${len.toLocaleString("vi-VN")} phiên` : "—";
    if (subNStatic) {
      if (!list.length || len === 0) subNStatic.textContent = "—";
      else if (staticLen === 0)
        subNStatic.textContent = `0 / ${len.toLocaleString("vi-VN")} (chưa có tĩnh)`;
      else
        subNStatic.textContent = `${staticLen.toLocaleString("vi-VN")} / ${len.toLocaleString("vi-VN")}`;
    }

    if (cap) {
      if (!list.length) {
        cap.textContent =
          "Chưa có lịch sử local — chờ phiên mới từ capture (cùng trình duyệt này).";
      } else if (effN === 0) {
        cap.textContent = "Không đủ dữ liệu.";
      } else {
        let tail = "";
        if (staticLen === 0 && len > 0) {
          tail =
            " Các mục cũ không có dự đoán tĩnh — % Tĩnh sẽ có sau khi tích lũy bản ghi mới.";
        } else if (staticLen > 0) {
          tail = ` Tĩnh tính trên ${staticLen.toLocaleString("vi-VN")} mục (trong ${len.toLocaleString("vi-VN")} mục đang xét).`;
        }
        cap.textContent = `Bạn đặt cửa sổ tối đa ${want.toLocaleString("vi-VN")} — Động: ${len.toLocaleString("vi-VN")} phiên mới nhất trong ${list.length.toLocaleString("vi-VN")} mục tổng.${tail}`;
      }
    }

    const streakParityEl = document.getElementById("compareStreakParity");
    const streakExactEl = document.getElementById("compareStreakExact");
    if (streakParityEl && streakExactEl) {
      if (!list.length) {
        streakParityEl.textContent = "—";
        streakExactEl.textContent = "—";
      } else {
        const st = maxComparePredictionStreaks(list);
        streakParityEl.textContent = `${st.parity.toLocaleString("vi-VN")} phiên`;
        streakExactEl.textContent = `${st.exact.toLocaleString("vi-VN")} phiên`;
      }
    }
  }

  function renderCompareHistoryList() {
    const wrap = document.getElementById("compareHistoryList");
    if (!wrap) return;
    const list = loadCompareHistory();
    if (!list.length) {
      wrap.innerHTML =
        '<p class="text-slate-500 text-xs py-6 text-center leading-relaxed px-2">Chưa có bản ghi.<br><span class="text-slate-600 text-[10px]">Khi <code class="text-slate-400">realtime_capture</code> ghi phiên mới, lịch sử so sánh sẽ tự đầy (lưu trong trình duyệt này).</span></p>';
      updateCompareStatsPanel();
      renderAlgoHistoryPanel();
      return;
    }
    wrap.innerHTML = list
      .map((e) => {
        const parityHit = compareHistoryParityMatches(e);
        const pl =
          e.predRed != null
            ? e.predRed % 2 === 0
              ? "Chẵn"
              : "Lẻ"
            : e.predParity === "chan"
              ? "Chẵn"
              : "Lẻ";
        const al = e.actualParity === "chan" ? "Chẵn" : "Lẻ";
        const stripeClass = e.exactOk
          ? "bg-emerald-500"
          : parityHit
            ? "bg-amber-500"
            : "bg-red-500";
        const pDots = e.predRed != null ? boardPatternHtml(e.predRed) : "";
        const aDots = e.actualRed != null ? boardPatternHtml(e.actualRed) : "";
        const predRS =
          e.predRedStatic != null && Number.isFinite(Number(e.predRedStatic))
            ? Number(e.predRedStatic)
            : NaN;
        const staticValid = Number.isFinite(predRS);
        const ps = staticValid ? (predRS % 2 === 0 ? "Chẵn" : "Lẻ") : null;
        const pStaticDots = staticValid ? boardPatternHtml(predRS) : "";
        const sameDoor =
          staticValid && e.predRed != null && predRS === e.predRed;
        const diffDoor =
          staticValid && e.predRed != null && predRS !== e.predRed;
        const stTy =
          staticValid && e.staticParityOk != null
            ? e.staticParityOk
              ? "✓"
              : "✗"
            : "—";
        const stVi =
          staticValid && e.staticExactOk != null
            ? e.staticExactOk
              ? "✓"
              : "✗"
            : "—";
        const stTyCls =
          staticValid && e.staticParityOk != null
            ? e.staticParityOk
              ? "text-emerald-400"
              : "text-red-400"
            : "text-slate-500";
        const stViCls =
          staticValid && e.staticExactOk != null
            ? e.staticExactOk
              ? "text-emerald-400"
              : "text-red-400"
            : "text-slate-500";
        const staticBlock = staticValid
          ? `<div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mt-1">
                <span class="text-[10px] font-black text-slate-500 uppercase tracking-wide shrink-0">Tĩnh · C^β:</span>
                <span class="inline-flex items-center gap-0 text-slate-300">
                  <span>${ps}</span><span class="text-slate-600 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${pStaticDots}</span>
                </span>
              </div>`
          : `<div class="mt-1 text-[10px] text-slate-600"><span class="font-black text-slate-500 uppercase">Tĩnh:</span> — <span class="text-slate-500 font-normal">(bản ghi cũ, chưa lưu)</span></div>`;
        //Todos: add sameDoor or diffDoor later
        // const staticBlock = staticValid
        // ? `<div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mt-1">
        //       <span class="text-[10px] font-black text-slate-500 uppercase tracking-wide shrink-0">Tĩnh · C^β:</span>
        //       <span class="inline-flex items-center gap-0 text-slate-300">
        //         <span>${ps}</span><span class="text-slate-600 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${pStaticDots}</span>
        //       </span>
        //       ${sameDoor ? '<span class="text-[9px] font-black text-emerald-500/90 uppercase ml-1">Cùng cửa vị</span>' : ""}
        //       ${diffDoor ? '<span class="text-[9px] font-black text-amber-400/90 uppercase ml-1">Khác cửa vị</span>' : ""}
        //     </div>`
        // : `<div class="mt-1 text-[10px] text-slate-600"><span class="font-black text-slate-500 uppercase">Tĩnh:</span> — <span class="text-slate-500 font-normal">(bản ghi cũ, chưa lưu)</span></div>`;
        const pBoard =
          pDots ||
          `<span class="text-slate-500 text-[10px] font-normal">${escapeHtml(e.predDoor)}</span>`;
        const aBoard =
          aDots ||
          `<span class="text-slate-500 text-[10px] font-normal">${escapeHtml(e.actualDoor)}</span>`;
        const clMark = parityHit ? "✓" : "✗";
        const viMark = e.exactOk ? "✓" : "✗";
        const clCls = parityHit ? "text-emerald-400" : "text-red-400";
        const viCls = e.exactOk ? "text-emerald-400" : "text-red-400";
        return `<div class="compare-history-row flex gap-2 rounded-lg border border-slate-800/90 bg-slate-900/50 p-2 mb-2 hover:border-slate-600/80 transition-colors">
          <div class="w-1 shrink-0 rounded-full ${stripeClass}" title="${e.exactOk ? "Đúng vị" : parityHit ? "Đúng CL, sai vị" : "Sai CL"}"></div>
          <div class="min-w-0 flex-1">
            <div class="font-mono text-[10px] text-slate-500">${escapeHtml(e.round_id)}</div>
            <div class="mt-1.5 space-y-1 text-[12px] leading-tight font-semibold">
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span class="text-[10px] font-black text-violet-400/95 uppercase tracking-wide shrink-0">Động:</span>
                <span class="inline-flex items-center gap-0 text-sky-100">
                  <span>${pl}</span><span class="text-slate-500 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${pBoard}</span>
                </span>
              </div>
              ${staticBlock}
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span class="text-[10px] font-black text-slate-400 uppercase tracking-wide shrink-0">Kết quả:</span>
                <span class="inline-flex items-center gap-0 text-white">
                  <span>${al}</span><span class="text-slate-500 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${aBoard}</span>
                </span>
              </div>
            </div>
            <div class="mt-1.5 text-[10px] font-black tracking-wide flex flex-wrap gap-x-3 gap-y-1">
              <span class="text-slate-500">Động</span>
              <span class="${clCls}">CL ${clMark}</span>
              <span class="text-slate-600">·</span>
              <span class="${viCls}">Vị ${viMark}</span>
              ${staticValid ? `<span class="text-slate-600 mx-0.5">|</span><span class="text-slate-500">Tĩnh</span><span class="${stTyCls}">CL ${stTy}</span><span class="text-slate-600">·</span><span class="${stViCls}">Vị ${stVi}</span>` : ""}
            </div>
            ${renderCompareAlgoSectionHtml(e.predByAlgo, { actualRed: e.actualRed, ensemblePredRed: e.predRed }, { defaultOpen: false })}
          </div>
        </div>`;
      })
      .join("");
    updateCompareStatsPanel();
    renderAlgoHistoryPanel();
  }

  /** @param {object | null} [ensembleStatic] ensemble C^β cùng thời điểm (toàn bộ lịch sử trước phiên). */
  function appendCompareHistoryRow(
    roundId,
    ensembleDynamic,
    actualRed,
    ensembleStatic,
  ) {
    const P = window.XocDiaPrediction;
    if (!P || !roundId || !ensembleDynamic) return;
    if (actualRed === undefined || actualRed === null) return;
    let list = loadCompareHistory();
    if (list.length && list[0].round_id === roundId) return;

    const predR = ensembleDynamic.predictedRed;
    const predP =
      ensembleDynamic.predictedParity || (predR % 2 === 0 ? "chan" : "le");
    const pm = P.getOutcomeMeta(predR);
    const am = P.getOutcomeMeta(actualRed);
    const actP = actualRed % 2 === 0 ? "chan" : "le";

    const row = {
      round_id: roundId,
      at: new Date().toISOString(),
      predParity: predP,
      predDoor: pm.short,
      predRed: predR,
      predByAlgo: algoPredictionsFromEnsemble(P, ensembleDynamic),
      actualParity: actP,
      actualDoor: am.short,
      actualRed,
      parityOk: predR % 2 === actualRed % 2,
      exactOk: predR === actualRed,
    };
    if (
      ensembleStatic &&
      ensembleStatic.predictedRed != null &&
      Number.isFinite(Number(ensembleStatic.predictedRed))
    ) {
      const predRS = Number(ensembleStatic.predictedRed);
      const sm = P.getOutcomeMeta(predRS);
      row.predRedStatic = predRS;
      row.predDoorStatic = sm.short;
      row.staticExactOk = predRS === actualRed;
      row.staticParityOk = predRS % 2 === actualRed % 2;
      row.sameDoorStaticDynamic = predRS === predR;
    }

    list.unshift(row);
    saveCompareHistory(list);
    renderCompareHistoryList();
  }

  function recordHits(ensembleSnapshot, actualRed) {
    if (!ensembleSnapshot || !ensembleSnapshot.algorithms) return;
    const pred = ensembleSnapshot.predictedRed;
    const ex = pred === actualRed;
    const ty = pred % 2 === actualRed % 2;
    accuracy.total.n++;
    if (ex) accuracy.total.exact++;
    if (ty) accuracy.total.type++;
    accuracy.recent.push({ exact: ex, type: ty });
    if (accuracy.recent.length > accuracy.maxRecent) accuracy.recent.shift();

    if (ensembleSnapshot.shouldBet) {
      accuracy.selective.bet++;
      accuracy.selective.n++;
      if (ex) accuracy.selective.exact++;
      if (ty) accuracy.selective.type++;
    }

    ensembleSnapshot.algorithms.forEach((a) => {
      const st = accuracy.byAlgo[a.id];
      if (!st) return;
      st.n++;
      if (a.predictedRed === actualRed) st.exact++;
      if (a.predictedRed % 2 === actualRed % 2) st.type++;
    });
  }

  async function fetchRounds() {
    const resp = await fetch("/api/rounds.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  function roundIdToDate(rid) {
    const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(rid || "");
    if (!m) return null;
    return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  }

  function toItem(round) {
    const red = DICE_TO_RED[round.dice_result];
    if (red === undefined) return null;
    let time = null;
    if (round.started_at) {
      const d = new Date(round.started_at);
      if (!isNaN(d)) time = d;
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

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function formatDateForInput(d) {
    return (
      d.getFullYear() +
      "-" +
      pad(d.getMonth() + 1) +
      "-" +
      pad(d.getDate()) +
      "T" +
      pad(d.getHours()) +
      ":" +
      pad(d.getMinutes()) +
      ":" +
      pad(d.getSeconds())
    );
  }

  function getFilterRange() {
    const fromInput = document.getElementById("fromDate").value;
    const toInput = document.getElementById("toDate").value;
    if (!fromInput || !toInput) return null;
    return { from: new Date(fromInput), to: new Date(toInput) };
  }

  function setFilterRange(from, to) {
    document.getElementById("fromDate").value = formatDateForInput(from);
    document.getElementById("toDate").value = formatDateForInput(to);
  }

  function defaultRange() {
    if (masterData.length === 0) {
      const now = new Date();
      return { from: new Date(now.getTime() - 60 * 60 * 1000), to: now };
    }
    return {
      from: masterData[0].time,
      to: masterData[masterData.length - 1].time,
    };
  }

  function filtered() {
    const r = getFilterRange();
    if (!r) return masterData.slice();
    return masterData.filter((it) => it.time >= r.from && it.time <= r.to);
  }

  function getDiceVisual(redCount) {
    let html = '<div class="grid grid-cols-2 gap-1 w-fit mx-auto mt-2">';
    for (let i = 0; i < 4; i++) {
      const cls = i < redCount ? "dot-red" : "dot-white";
      html += `<span class="dot ${cls}"></span>`;
    }
    html += "</div>";
    return html;
  }

  function renderStats(list) {
    document.getElementById("totalRounds").innerText = `${list.length} phiên`;
    const total = list.length || 1;
    const stats = { chan: 0, le: 0, v0: 0, v1: 0, v2: 0, v3: 0, v4: 0 };
    list.forEach((it) => {
      stats[it.type]++;
      stats["v" + it.red]++;
    });

    const chanP = Math.round((stats.chan / total) * 100);
    const leP = list.length ? 100 - chanP : 0;

    document.getElementById("mainStats").innerHTML = `
      <div class="space-y-2">
        <div class="flex justify-between font-black text-xs">
          <span class="text-white uppercase">Chẵn</span>
          <span class="text-white">${chanP}% · ${stats.chan}</span>
        </div>
        <div class="progress-bar"><div class="h-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.4)]" style="width:${chanP}%"></div></div>
      </div>
      <div class="space-y-2">
        <div class="flex justify-between font-black text-xs">
          <span class="text-red-500 uppercase">Lẻ</span>
          <span class="text-red-500">${leP}% · ${stats.le}</span>
        </div>
        <div class="progress-bar"><div class="h-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]" style="width:${leP}%"></div></div>
      </div>
    `;

    let viHtml = "";
    for (let i = 0; i < 5; i++) {
      const p = Math.round((stats["v" + i] / total) * 100);
      viHtml += `
        <div class="bg-slate-800/60 border border-slate-700/50 p-3 rounded-xl text-center hover:bg-slate-800 transition-colors">
          <div class="text-[8px] text-slate-500 font-black mb-1 uppercase">${VI_LABELS[i]}</div>
          <div class="text-xl font-black text-white">${list.length ? p : 0}%</div>
          ${getDiceVisual(i)}
          <div class="text-[9px] text-slate-500 mt-2 font-bold">${stats["v" + i]} phiên</div>
        </div>
      `;
    }
    document.getElementById("viStats").innerHTML = viHtml;
  }

  function renderRoadmap(list) {
    const container = document.getElementById("roadmap");
    container.innerHTML = "";

    const ROWS = 6;
    const matrix = Array.from({ length: ROWS }, () => []);
    let curR = 0;
    let curC = 0;
    let prevType = null;

    const advanceToEmptyColumn = () => {
      curC++;
      while (matrix[0][curC]) curC++;
    };

    list.forEach((item) => {
      if (item.type !== prevType) {
        curR = 0;
        if (prevType !== null) {
          advanceToEmptyColumn();
        }
      } else {
        if (curR + 1 < ROWS && !matrix[curR + 1][curC]) {
          curR++;
        } else {
          curR = 0;
          advanceToEmptyColumn();
        }
      }

      matrix[curR][curC] = item.type;
      prevType = item.type;

      const cell = document.createElement("div");
      cell.className = `cell ${item.type === "chan" ? "cell-chan" : "cell-le"}`;
      cell.innerText = item.red;
      cell.title = `${item.round_id || "?"} — ${VI_LABELS[item.red]}`;
      cell.style.gridRowStart = curR + 1;
      cell.style.gridColumnStart = curC + 1;
      container.appendChild(cell);
    });

    const scroll = document.getElementById("roadmapScroll");
    scroll.scrollLeft = scroll.scrollWidth;
  }

  function render() {
    const list = filtered();
    renderStats(list);
    renderRoadmap(list);
  }

  function pct01(x) {
    return Math.round(Math.max(0, Math.min(1, x)) * 100);
  }

  function updateAlertBanner(ensemble) {
    const el = document.getElementById("alertBanner");
    if (!el || !ensemble) return;
    const strong =
      ensemble.confidence >= ALERT_CONF &&
      ensemble.consensus >= ALERT_CONSENSUS;
    if (strong) {
      el.classList.remove("hidden");
      el.innerHTML = `Cảnh báo: độ tin cậy ${pct01(ensemble.confidence)}% & đồng thuận ${pct01(ensemble.consensus)}% — xem kỹ trước khi hành động.`;
    } else {
      el.classList.add("hidden");
    }

    if (
      strong &&
      typeof Notification !== "undefined" &&
      Notification.permission === "granted"
    ) {
      try {
        new Notification("Xóc đĩa predictor", {
          body: `${ensemble.predictedParity === "chan" ? "CHẴN" : "LẺ"} · ${ensemble.outcome?.title || "cửa"} (${ensemble.predictedRed})`,
        });
      } catch (_) {
        /* ignore */
      }
    }
  }

  function renderDynamicWeights(ensembleFiltered) {
    const panel = document.getElementById("dynamicWeightsPanel");
    const metaEl = document.getElementById("dynamicWeightsMeta");
    const P = window.XocDiaPrediction;
    if (!panel) return;

    const w = ensembleFiltered && ensembleFiltered.ensembleWeights;
    const algos =
      ensembleFiltered && Array.isArray(ensembleFiltered.algorithms)
        ? ensembleFiltered.algorithms
        : [];
    if (!w || !algos.length || w.length !== algos.length) {
      panel.innerHTML =
        '<p class="text-slate-500 text-[10px]">Chưa có dữ liệu trọng số (ensemble).</p>';
      if (metaEl) metaEl.textContent = "";
      return;
    }

    const de = P && P.DYNAMIC_ENSEMBLE;
    if (metaEl && de) {
      metaEl.textContent = `W ≈ [ α·C + (1−α)·H ]^β · α=${de.ALPHA} · β=${de.BETA} · H đa khung: ngắn=${de.HIT_WINDOW_SHORT} · dài=${de.HIT_WINDOW_LONG} · φ=${de.HIT_MULTI_PHI} · (1 cửa: opts.hitWindow, mặc định ${de.HIT_WINDOW}) · γ(vị)=${de.HIT_BLEND_EXACT} · pseudo-count H=${de.H_HIT_SHRINK ?? 0}`;
    }

    const sum = w.reduce((acc, x) => acc + (Number(x) || 0), 0);
    const maxW = Math.max(...w.map((x) => Math.max(0, Number(x) || 0)), 1e-12);
    const pairs = algos.map((a, i) => ({
      label: ALGO_LABEL_VI[a.id] || a.name,
      weight: Math.max(0, Number(w[i]) || 0),
    }));
    pairs.sort((a, b) => b.weight - a.weight);

    panel.innerHTML = pairs
      .map((p) => {
        const share = sum > 0 ? (100 * p.weight) / sum : 0;
        const shareStr = share.toFixed(1);
        const barPct = Math.min(100, Math.round((100 * p.weight) / maxW));
        const wStr =
          p.weight >= 0.0001
            ? p.weight.toFixed(4)
            : p.weight > 0
              ? p.weight.toExponential(1)
              : "0";
        return `<div class="flex items-center gap-2 py-1.5 border-b border-slate-800/50 last:border-0">
          <span class="text-slate-300 truncate flex-1 min-w-0 text-[10px] font-semibold" title="${escapeHtml(p.label)}">${escapeHtml(p.label)}</span>
          <div class="flex-1 min-w-[3.5rem] max-w-[45%] h-2 rounded bg-slate-800 overflow-hidden shrink-0" title="So với W lớn nhất trong 8 TT">
            <div class="h-full bg-violet-500/85 rounded transition-all duration-300" style="width:${barPct}%"></div>
          </div>
          <span class="font-mono text-violet-200 tabular-nums w-[4.25rem] text-right shrink-0 text-[10px]">${wStr}</span>
          <span class="font-mono text-slate-500 w-11 text-right shrink-0 text-[10px]">${shareStr}%</span>
        </div>`;
      })
      .join("");
  }

  function renderEnsembleModeCompare(ensembleStatic, ensembleDynamic) {
    const el = document.getElementById("ensembleModeCompare");
    const P = window.XocDiaPrediction;
    if (!el) return;
    if (
      !P ||
      !ensembleDynamic ||
      !ensembleStatic ||
      !Array.isArray(ensembleDynamic.algorithms) ||
      ensembleDynamic.algorithms.length === 0
    ) {
      el.innerHTML =
        '<p class="text-slate-500 text-[10px] col-span-full">Chưa có dữ liệu.</p>';
      return;
    }
    const ds = ensembleStatic.predictedRed;
    const dd = ensembleDynamic.predictedRed;
    const same = ds === dd;
    const ms = P.getOutcomeMeta(ds);
    const md = P.getOutcomeMeta(dd);
    const ps = ds % 2 === 0 ? "Chẵn" : "Lẻ";
    const pd = dd % 2 === 0 ? "Chẵn" : "Lẻ";
    const badge = same
      ? '<span class="inline-block mt-2 text-[9px] font-black text-emerald-400/95 uppercase tracking-wide">Cùng cửa vị</span>'
      : '<span class="inline-block mt-2 text-[9px] font-black text-amber-400/95 uppercase tracking-wide">Khác cửa vị</span>';
    el.innerHTML = `
      <div class="rounded-lg border border-slate-700/80 bg-slate-950/50 p-3">
        <div class="text-[9px] font-black text-slate-500 uppercase mb-2">Tĩnh · C^β</div>
        <div class="font-bold text-slate-200">${escapeHtml(ms.short)}</div>
        <div class="mt-1 flex flex-wrap items-center gap-x-2 text-[10px] text-slate-400">
          <span>${ps}</span><span class="text-slate-600">·</span><span class="inline-flex items-center">${boardPatternHtml(ds)}</span>
        </div>
        <div class="text-[10px] font-mono text-slate-500 mt-1">Đỏ=${ds}</div>
      </div>
      <div class="rounded-lg border border-violet-600/40 bg-violet-950/20 p-3">
        <div class="text-[9px] font-black text-violet-400 uppercase mb-2">Động · chính</div>
        <div class="font-bold text-violet-100">${escapeHtml(md.short)}</div>
        <div class="mt-1 flex flex-wrap items-center gap-x-2 text-[10px] text-slate-300">
          <span>${pd}</span><span class="text-slate-600">·</span><span class="inline-flex items-center">${boardPatternHtml(dd)}</span>
        </div>
        <div class="text-[10px] font-mono text-slate-400 mt-1">Đỏ=${dd}</div>
        ${badge}
      </div>`;
  }

  function renderPrediction(ensembleDynamic, ensembleStatic) {
    const P = window.XocDiaPrediction;
    if (!P || !ensembleDynamic) return;

    const ensembleFiltered = ensembleDynamic;
    const pred = ensembleFiltered.predictedRed;
    const om = ensembleFiltered.outcome || P.getOutcomeMeta(pred);
    const parityFromRed = pred % 2 === 0 ? "chan" : "le";
    const predictedParity = ensembleFiltered.predictedParity || parityFromRed;
    const parityBig = predictedParity === "chan" ? "CHẴN" : "LẺ";
    const parityMetaHint =
      predictedParity === "chan"
        ? "Nhóm Chẵn gồm: 4 Trắng · Sấp đôi · 4 Đỏ"
        : "Nhóm Lẻ gồm: 3 Trắng 1 Đỏ · 3 Đỏ 1 Trắng";

    const pConf =
      ensembleFiltered.parityConfidence ?? ensembleFiltered.confidence;
    const pCons =
      ensembleFiltered.parityConsensus ?? ensembleFiltered.consensus;
    const pConsN =
      ensembleFiltered.parityConsensusCount ?? ensembleFiltered.consensusCount;

    document.getElementById("predParityBig").textContent = parityBig;
    const metaEl = document.getElementById("predParityMeta");
    if (metaEl) metaEl.textContent = parityMetaHint;
    document.getElementById("predParityConfText").textContent =
      `${pct01(pConf)}%`;
    document.getElementById("predParityConfBar").style.width =
      `${pct01(pConf)}%`;
    document.getElementById("predParityConsensusText").textContent =
      `${pct01(pCons)}% (${pConsN}/${ensembleFiltered.algorithmCount})`;
    document.getElementById("predParityConsensusBar").style.width =
      `${pct01(pCons)}%`;

    document.getElementById("predOutcomeTitle").textContent = om.title;
    const boardEl = document.getElementById("predOutcomeBoard");
    if (boardEl) boardEl.innerHTML = boardPatternHtml(pred);
    document.getElementById("predOutcomeDesc").textContent = om.line;
    document.getElementById("predNumber").textContent = String(pred);
    document.getElementById("predOutcomeParityHint").textContent =
      `Theo cửa này: ${parityFromRed === "chan" ? "Chẵn" : "Lẻ"}`;

    const misEl = document.getElementById("predParityMismatch");
    const mismatch =
      ensembleFiltered.parityMismatch ?? predictedParity !== parityFromRed;
    if (misEl) {
      if (mismatch) {
        misEl.classList.remove("hidden");
        misEl.textContent = `Lưu ý: tín hiệu chẵn/lẻ tổng hợp (${
          predictedParity === "chan" ? "Chẵn" : "Lẻ"
        }) khác với cửa chi tiết được chọn (${om.short} → ${
          parityFromRed === "chan" ? "Chẵn" : "Lẻ"
        }).`;
      } else {
        misEl.classList.add("hidden");
      }
    }

    const confPct = pct01(ensembleFiltered.confidence);
    const consPct = pct01(ensembleFiltered.consensus);

    document.getElementById("predConfBar").style.width = `${confPct}%`;
    document.getElementById("predConfText").textContent = `${confPct}%`;
    document.getElementById("predConsensusText").textContent =
      `${consPct}% (${ensembleFiltered.consensusCount}/${ensembleFiltered.algorithmCount})`;
    document.getElementById("predConsensusBar").style.width = `${consPct}%`;
    document.getElementById("predReason").textContent =
      ensembleFiltered.weightedReason || "—";

    const betBadgeEl = document.getElementById("betBadge");
    if (betBadgeEl) {
      if (ensembleFiltered.shouldBet) {
        betBadgeEl.className = "px-3 py-1 rounded-full text-xs font-bold bg-emerald-600/80 text-white";
        betBadgeEl.textContent = "✓ NÊN BET";
      } else {
        betBadgeEl.className = "px-3 py-1 rounded-full text-xs font-bold bg-red-600/60 text-red-200";
        betBadgeEl.textContent = "✗ BỎ QUA";
      }
    }
    const betReasonEl = document.getElementById("betReason");
    if (betReasonEl) betReasonEl.textContent = ensembleFiltered.betReason || "";
    const betConfEl = document.getElementById("betConfValue");
    if (betConfEl) betConfEl.textContent = `${pct01(ensembleFiltered.betConfidence)}%`;
    const regimeEl = document.getElementById("regimeValue");
    if (regimeEl) {
      const regimeLabels = { streaky: "Chuỗi (Streaky)", alternating: "Xen kẽ (Alternating)", random: "Ngẫu nhiên (Random)" };
      regimeEl.textContent = regimeLabels[ensembleFiltered.regime] || ensembleFiltered.regime || "—";
    }

    renderDynamicWeights(ensembleFiltered);
    renderEnsembleModeCompare(ensembleStatic, ensembleFiltered);

    const tbody = document.getElementById("algoTableBody");
    tbody.innerHTML = "";
    const wArr = ensembleFiltered.ensembleWeights;
    const wStatArr =
      ensembleStatic && Array.isArray(ensembleStatic.ensembleWeights)
        ? ensembleStatic.ensembleWeights
        : null;
    const algs = ensembleFiltered.algorithms;
    let bestIdx = 0;
    if (wArr && wArr.length === algs.length && algs.length > 0) {
      for (let i = 1; i < wArr.length; i++) {
        if ((Number(wArr[i]) || 0) > (Number(wArr[bestIdx]) || 0)) bestIdx = i;
      }
    } else if (algs.length > 0) {
      let maxC = algs[0].confidence;
      for (let i = 1; i < algs.length; i++) {
        if (algs[i].confidence > maxC) {
          maxC = algs[i].confidence;
          bestIdx = i;
        }
      }
    }
    const bestId = algs[bestIdx]?.id || "";
    const sumW =
      wArr && wArr.length ? wArr.reduce((a, x) => a + (Number(x) || 0), 0) : 0;

    ensembleFiltered.algorithms.forEach((a, i) => {
      const ap = a.predictedRed % 2 === 0 ? "Chẵn" : "Lẻ";
      const meta = P.getOutcomeMeta(a.predictedRed);
      const star = a.id === bestId ? " \u2B50" : "";
      const wi = wArr && wArr[i] != null ? Number(wArr[i]) : null;
      const wSt = wStatArr && wStatArr[i] != null ? Number(wStatArr[i]) : null;
      const shareStr =
        wi != null && sumW > 0 ? `${((100 * wi) / sumW).toFixed(1)}%` : "—";
      const wCell =
        wi != null ? (wi >= 0.0001 ? wi.toFixed(4) : wi.toExponential(1)) : "—";
      const wStatCell =
        wSt != null
          ? wSt >= 0.0001
            ? wSt.toFixed(4)
            : wSt.toExponential(1)
          : "—";
      const tr = document.createElement("tr");
      tr.className = "border-b border-slate-800/80";
      tr.innerHTML = `
        <td class="py-2 pr-2 font-bold text-emerald-400">\u2713</td>
        <td class="py-2 pr-2">${ALGO_LABEL_VI[a.id] || a.name}${star}</td>
        <td class="py-2 pr-2 font-bold text-slate-200">${meta.short}</td>
        <td class="py-2 pr-2 font-mono">${ap}</td>
        <td class="py-2 pr-2 text-right text-slate-300">${pct01(a.confidence)}%</td>
        <td class="py-2 pr-2 text-right font-mono text-slate-400">${wStatCell}</td>
        <td class="py-2 pr-2 text-right font-mono text-violet-300">${wCell}</td>
        <td class="py-2 text-right font-mono text-slate-400">${shareStr}</td>
      `;
      tbody.appendChild(tr);
    });

    updateAlertBanner(ensembleFiltered);
  }

  let lastFetchOk = false;
  let lastFetchAt = null;

  function setStatus(text, ok = true) {
    document.getElementById("statusLine").innerText = text;
    const badge = document.getElementById("liveBadge");
    if (ok) {
      badge.classList.remove("text-red-500");
      badge.classList.add("text-emerald-500");
      badge.innerText = "\u25CF System Online";
    } else {
      badge.classList.remove("text-emerald-500");
      badge.classList.add("text-red-500");
      badge.innerText = "\u25CF Offline";
    }
  }

  // ── Live Round Panel ──────────────────────────────────────────────
  const PERCENT_LABELS = {
    chan: { label: "Chẵn", color: "bg-white", text: "text-slate-900" },
    le: { label: "Lẻ", color: "bg-red-500", text: "text-white" },
    "4_red": { label: "4 Đỏ", color: "bg-red-600", text: "text-white" },
    "3r_1w": { label: "3Đ 1T", color: "bg-red-400", text: "text-white" },
    "3w_1r": { label: "3T 1Đ", color: "bg-slate-300", text: "text-slate-900" },
    "4_white": { label: "4 Trắng", color: "bg-white", text: "text-slate-900" },
  };
  const PERCENT_ORDER = ["chan", "le", "4_red", "3r_1w", "3w_1r", "4_white"];
  const BETS_ORDER = ["chan", "le", "4_red", "3r_1w", "3w_1r", "4_white"];

  function parsePctLocal(s) {
    if (!s || s === "-") return null;
    const n = parseFloat(String(s).replace("%", ""));
    return isNaN(n) ? null : n;
  }

  function renderCurrentRound(round) {
    const panel = document.getElementById("liveRoundPanel");
    if (!panel) return;
    if (!round) {
      panel.classList.add("hidden");
      currentInProgress = null;
      return;
    }
    panel.classList.remove("hidden");

    const ridEl = document.getElementById("liveRoundId");
    if (ridEl) ridEl.textContent = round.round_id || "—";

    // Percent bars
    const barsEl = document.getElementById("livePercentBars");
    if (barsEl && round.percent) {
      const hasAnyValid = PERCENT_ORDER.some((k) => parsePctLocal(round.percent[k]) !== null);
      if (!hasAnyValid) {
        barsEl.innerHTML = '<span class="text-[10px] text-amber-400/80 italic">Không nhận diện được percent</span>';
      } else {
        const rows = PERCENT_ORDER.map((key) => {
          const meta = PERCENT_LABELS[key] || { label: key, color: "bg-slate-500", text: "text-white" };
          const val = parsePctLocal(round.percent[key]);
          if (val === null) {
            return `<div class="flex items-center gap-2">
              <span class="w-14 text-[10px] font-bold text-slate-400 shrink-0">${meta.label}</span>
              <div class="flex-1 h-4 rounded-full bg-slate-800/50 flex items-center px-2">
                <span class="text-[9px] text-slate-600 italic">Không nhận diện</span>
              </div>
            </div>`;
          }
          return `<div class="flex items-center gap-2">
            <span class="w-14 text-[10px] font-bold text-slate-400 shrink-0">${meta.label}</span>
            <div class="flex-1 h-4 rounded-full bg-slate-800 overflow-hidden relative">
              <div class="${meta.color} h-full rounded-full transition-all duration-500 flex items-center justify-end pr-1" style="width:${Math.max(val, 2)}%">
                ${val >= 15 ? `<span class="${meta.text} text-[9px] font-black">${val}%</span>` : ""}
              </div>
              ${val < 15 ? `<span class="absolute right-1.5 top-0 h-full flex items-center text-[9px] font-black text-slate-400">${val}%</span>` : ""}
            </div>
          </div>`;
        });
        barsEl.innerHTML = rows.join("");
      }
    } else if (barsEl) {
      barsEl.innerHTML = '<span class="text-[10px] text-slate-600">Chưa có dữ liệu percent</span>';
    }

    // Bets table
    const betsEl = document.getElementById("liveBetsTable");
    if (betsEl && round.bets && Object.keys(round.bets).length > 0) {
      const header = `<div class="grid grid-cols-3 gap-1 text-[9px] font-black text-slate-500 uppercase tracking-wider mb-1.5 px-2">
        <span>Cửa</span><span class="text-right">Tổng cược</span><span class="text-right">Lượt</span>
      </div>`;
      const rows = BETS_ORDER.filter((k) => round.bets[k]).map((key) => {
        const b = round.bets[key];
        const meta = PERCENT_LABELS[key] || { label: key };
        const totalBet = b.total_bet || "—";
        const totalCount = b.total_count || "—";
        return `<div class="grid grid-cols-3 gap-1 px-2 py-1 rounded-lg hover:bg-slate-800/50 transition-colors">
          <span class="font-bold text-slate-300">${meta.label}</span>
          <span class="text-right font-mono text-amber-400 tabular-nums">${totalBet}</span>
          <span class="text-right font-mono text-slate-400 tabular-nums">${totalCount}</span>
        </div>`;
      });
      betsEl.innerHTML = header + rows.join("");
    } else if (betsEl) {
      betsEl.innerHTML = '<span class="text-[10px] text-slate-600">Chưa có dữ liệu bets</span>';
    }

    // Time info
    const timeEl = document.getElementById("liveRoundTime");
    if (timeEl && round.started_at) {
      const d = new Date(round.started_at);
      if (!isNaN(d)) {
        timeEl.textContent = `Bắt đầu: ${d.toLocaleTimeString("vi-VN")}`;
      }
    }
    const ageEl = document.getElementById("liveRoundAge");
    if (ageEl && round.started_at) {
      const d = new Date(round.started_at);
      if (!isNaN(d)) {
        const sec = Math.round((Date.now() - d.getTime()) / 1000);
        ageEl.textContent = sec > 0 ? `(${sec}s trước)` : "";
      }
    }
  }

  async function refresh() {
    const P = window.XocDiaPrediction;
    try {
      const prevLen = masterData.length;
      const raw = await fetchRounds();

      const inProgressRounds = raw.filter((r) => !r.dice_result);
      currentInProgress =
        inProgressRounds.length > 0
          ? inProgressRounds[inProgressRounds.length - 1]
          : null;
      renderCurrentRound(currentInProgress);

      const items = raw.map(toItem).filter(Boolean);
      items.sort((a, b) => a.time - b.time);

      if (lastFullEnsemble && P && items.length > prevLen && prevLen > 0) {
        const actualRed = items[items.length - 1].red;
        const rid = items[items.length - 1].round_id || "";
        recordHits(lastFullEnsemble, actualRed);
        appendCompareHistoryRow(
          rid,
          lastFullEnsemble,
          actualRed,
          lastFullEnsembleStatic,
        );
        const lc = document.getElementById("lastCompare");
        if (lc) {
          const prevPred = lastFullEnsemble && lastFullEnsemble.predictedRed;
          const prevPredStatic =
            lastFullEnsembleStatic &&
            lastFullEnsembleStatic.predictedRed != null &&
            Number.isFinite(Number(lastFullEnsembleStatic.predictedRed))
              ? Number(lastFullEnsembleStatic.predictedRed)
              : null;
          const pParity =
            (lastFullEnsemble.predictedParity ||
              (prevPred % 2 === 0 ? "chan" : "le")) === "chan"
              ? "Chẵn"
              : "Lẻ";
          const aParity = actualRed % 2 === 0 ? "Chẵn" : "Lẻ";
          const ex = actualRed === prevPred;
          const ty = prevPred % 2 === actualRed % 2;
          const exS =
            prevPredStatic != null ? actualRed === prevPredStatic : null;
          const tyS =
            prevPredStatic != null
              ? prevPredStatic % 2 === actualRed % 2
              : null;
          const sameDoorST =
            prevPredStatic != null && prevPred === prevPredStatic;
          const staticCompareLine =
            prevPredStatic != null
              ? `<div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span class="text-[10px] font-black text-slate-500 uppercase tracking-wide shrink-0">Tĩnh · C^β:</span>
                <span class="inline-flex items-center gap-0 text-slate-300">
                  <span>${prevPredStatic % 2 === 0 ? "Chẵn" : "Lẻ"}</span><span class="text-slate-600 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${boardPatternHtml(prevPredStatic)}</span>
                </span>
                ${sameDoorST ? '<span class="text-[9px] font-black text-emerald-500/90 uppercase ml-1">Cùng cửa</span>' : '<span class="text-[9px] font-black text-amber-400/90 uppercase ml-1">Khác cửa</span>'}
              </div>`
              : "";
          lc.innerHTML = `<div class="space-y-2 text-[13px] leading-snug text-sky-50">
            <div class="font-mono text-[10px] text-sky-300/80">${escapeHtml(rid)}</div>
            <div class="space-y-1 font-semibold">
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span class="text-[10px] font-black text-violet-400 uppercase tracking-wide shrink-0">Động:</span>
                <span class="inline-flex items-center gap-0 text-sky-100">
                  <span>${pParity}</span><span class="text-sky-400/70 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${boardPatternHtml(prevPred)}</span>
                </span>
              </div>
              ${staticCompareLine}
              <div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                <span class="text-[10px] font-black text-slate-400 uppercase tracking-wide shrink-0">Kết quả:</span>
                <span class="inline-flex items-center gap-0 text-white">
                  <span>${aParity}</span><span class="text-slate-500 font-normal select-none px-0.5"> - </span><span class="inline-flex items-center translate-y-[0.5px]">${boardPatternHtml(actualRed)}</span>
                </span>
              </div>
            </div>
            <div class="text-[10px] font-black tracking-wide flex flex-wrap gap-x-3 gap-y-1">
              <span class="text-slate-500">Động</span>
              <span class="${ty ? "text-emerald-400" : "text-red-400"}">CL ${ty ? "✓" : "✗"}</span>
              <span class="text-slate-600">·</span>
              <span class="${ex ? "text-emerald-400" : "text-red-400"}">Vị ${ex ? "✓" : "✗"}</span>
              ${
                prevPredStatic != null
                  ? `<span class="text-slate-600 mx-0.5">|</span><span class="text-slate-500">Tĩnh</span><span class="${tyS ? "text-emerald-400" : "text-red-400"}">CL ${tyS ? "✓" : "✗"}</span><span class="text-slate-600">·</span><span class="${exS ? "text-emerald-400" : "text-red-400"}">Vị ${exS ? "✓" : "✗"}</span>`
                  : ""
              }
            </div>
            ${renderCompareAlgoSectionHtml(algoPredictionsFromEnsemble(P, lastFullEnsemble), { actualRed, ensemblePredRed: prevPred }, { defaultOpen: true })}
          </div>`;
        }
      }

      masterData = items;
      lastFetchOk = true;
      lastFetchAt = new Date();

      if (autoFollow && masterData.length) {
        const { from, to } = defaultRange();
        setFilterRange(from, to);
      }

      render();

      const currentRound = currentInProgress
        ? { percent: currentInProgress.percent, bets: currentInProgress.bets }
        : null;

      if (P && masterData.length) {
        lastFullEnsemble = P.ensemblePredict(masterData, { currentRound });
        lastFullEnsembleStatic = P.ensemblePredict(masterData, {
          dynamic: false,
          currentRound,
        });
        const list = filtered();
        renderPrediction(
          P.ensemblePredict(list, { currentRound }),
          P.ensemblePredict(list, { dynamic: false, currentRound }),
        );
      } else if (P) {
        lastFullEnsemble = null;
        lastFullEnsembleStatic = null;
        renderPrediction(
          P.ensemblePredict([], { currentRound }),
          P.ensemblePredict([], { dynamic: false, currentRound }),
        );
      }

      const newlyAdded = Math.max(0, masterData.length - prevLen);
      const when = lastFetchAt.toLocaleTimeString();
      const suffix = newlyAdded ? ` \u00b7 +${newlyAdded} m\u1edbi` : "";
      const liveSuffix = currentInProgress
        ? ` \u00b7 \uD83D\uDD34 ${currentInProgress.round_id || "live"}`
        : "";
      const selStats = accuracy.selective.n > 0
        ? ` \u00b7 Selective: ${pct01(accuracy.selective.type / accuracy.selective.n)}% (${accuracy.selective.type}/${accuracy.selective.n})`
        : "";
      setStatus(
        `C\u1eadp nh\u1eadt ${when} \u00b7 ${masterData.length} round t\u1ed5ng${suffix}${selStats}${liveSuffix}`,
        true,
      );
    } catch (err) {
      lastFetchOk = false;
      setStatus(`L\u1ed7i t\u1ea3i /api/rounds.json: ${err}`, false);
    }
  }

  function setFollow(on) {
    autoFollow = on;
    const label = document.getElementById("followLabel");
    label.innerText = on ? "AUTO" : "FIXED";
    const btn = document.getElementById("resetBtn");
    btn.classList.toggle("bg-emerald-600", on);
    btn.classList.toggle("hover:bg-emerald-500", on);
    btn.classList.toggle("bg-slate-800", !on);
    btn.classList.toggle("hover:bg-slate-700", !on);
  }

  function runBacktestUI() {
    const P = window.XocDiaPrediction;
    const burnEl = document.getElementById("backtestBurnIn");
    const burnIn = Math.max(5, parseInt(burnEl.value, 10) || 40);
    const list = filtered();
    if (list.length <= burnIn) {
      document.getElementById("backtestOut").textContent =
        `C\u1ea7n nhi\u1ec1u h\u01a1n ${burnIn} phi\u00ean trong b\u1ed9 l\u1ecdc.`;
      return;
    }
    const res = P.runBacktest(list, { burnIn });
    const base = P.runBaselines(list, { burnIn });
    const lines = [
      `B\u01b0\u1edbc: ${res.totalSteps} (burn-in ${res.burnIn})`,
      `Ensemble \u0110\u1ED9ng: exact ${res.ensemble.exactPct}% \u00b7 ch\u1ebb/l\u1ebb ${res.ensemble.typePct}%`,
      `Ensemble T\u0129nh (C^\u03B2): exact ${res.ensembleStatic.exactPct}% \u00b7 ch\u1ebb/l\u1ebb ${res.ensembleStatic.typePct}%`,
      `Ch\u00eanh l\u1EC7ch \u0110\u1ED9ng \u2212 T\u0129nh: exact ${res.modeComparison.exactPctDelta >= 0 ? "+" : ""}${res.modeComparison.exactPctDelta} pp \u00b7 CL ${res.modeComparison.typePctDelta >= 0 ? "+" : ""}${res.modeComparison.typePctDelta} pp`,
      `Tr\u00f9ng c\u1eeda v\u1ECB: ${res.modeComparison.sameRedPct}% (${res.modeComparison.sameRedCount}/${res.modeComparison.steps}) \u00b7 Kh\u00e1c c\u1eeda: ${res.modeComparison.differRedCount} b\u01b0\u1EDBc`,
      `Khi kh\u00e1c c\u1eeda, tr\u00fang v\u1ECB: \u0110\u1ED9ng th\u1eafng ${res.modeComparison.dynWinsExactWhenDiffer} \u00b7 T\u0129nh th\u1eafng ${res.modeComparison.statWinsExactWhenDiffer}`,
      `Random:  exact ${base.random.exactPct}% \u00b7 ch\u1ebb/l\u1ebb ${base.random.typePct}%`,
      `L\u1eb7p tr\u01b0\u1edbc: exact ${base.lastRepeats.exactPct}% \u00b7 ch\u1ebb/l\u1ebb ${base.lastRepeats.typePct}%`,
      "---",
      ...Object.keys(res.byAlgo).map(
        (id) =>
          `${ALGO_LABEL_VI[id]}: exact ${res.byAlgo[id].exactPct}% \u00b7 type ${res.byAlgo[id].typePct}% (n=${res.byAlgo[id].n})`,
      ),
    ];
    document.getElementById("backtestOut").textContent = lines.join("\n");
    window.__lastBacktest = { res, base, listMeta: { n: list.length, burnIn } };
  }

  function exportJson() {
    const P = window.XocDiaPrediction;
    const list = filtered();
    const ens = list.length ? P.ensemblePredict(list) : null;
    const ensStat =
      list.length && P
        ? P.ensemblePredict(list, { dynamic: false })
        : P
          ? P.ensemblePredict([], { dynamic: false })
          : null;
    const payload = {
      exportedAt: new Date().toISOString(),
      filteredCount: list.length,
      prediction: ens,
      predictionStatic: ensStat,
      accuracy,
      compareHistoryLocal: loadCompareHistory(),
      compareStatsWindowN: getCompareStatsWindowN(),
      backtest: window.__lastBacktest || null,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `xocdia-predict-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function exportCsv() {
    const rows = [["metric", "value"]];
    rows.push(["total_exact", String(accuracy.total.exact)]);
    rows.push(["total_type", String(accuracy.total.type)]);
    rows.push(["total_n", String(accuracy.total.n)]);
    Object.keys(accuracy.byAlgo).forEach((id) => {
      const s = accuracy.byAlgo[id];
      rows.push([`${id}_exact`, String(s.exact)]);
      rows.push([`${id}_n`, String(s.n)]);
    });
    const esc = (x) => `"${String(x).replace(/"/g, '""')}"`;
    const csv = rows.map((r) => r.map(esc).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `xocdia-metrics-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  document.getElementById("applyBtn").addEventListener("click", () => {
    setFollow(false);
    render();
    if (window.XocDiaPrediction && masterData.length) {
      const list = filtered();
      const P = window.XocDiaPrediction;
      renderPrediction(
        P.ensemblePredict(list),
        P.ensemblePredict(list, { dynamic: false }),
      );
    }
  });
  document.getElementById("resetBtn").addEventListener("click", () => {
    const newState = !autoFollow;
    setFollow(newState);
    if (newState && masterData.length) {
      const { from, to } = defaultRange();
      setFilterRange(from, to);
      render();
      if (window.XocDiaPrediction) {
        const list = filtered();
        const P = window.XocDiaPrediction;
        renderPrediction(
          P.ensemblePredict(list),
          P.ensemblePredict(list, { dynamic: false }),
        );
      }
    }
  });
  ["fromDate", "toDate"].forEach((id) => {
    document
      .getElementById(id)
      .addEventListener("change", () => setFollow(false));
  });

  document
    .getElementById("runBacktestBtn")
    .addEventListener("click", runBacktestUI);
  document
    .getElementById("exportJsonBtn")
    .addEventListener("click", exportJson);
  document.getElementById("exportCsvBtn").addEventListener("click", exportCsv);
  document.getElementById("notifBtn").addEventListener("click", async () => {
    if (typeof Notification === "undefined") return;
    const p = await Notification.requestPermission();
    alert("Notification: " + p);
  });

  const clearHistBtn = document.getElementById("clearCompareHistoryBtn");
  if (clearHistBtn) {
    clearHistBtn.addEventListener("click", () => {
      try {
        localStorage.removeItem(COMPARE_LS_KEY);
      } catch (_) {
        /* ignore */
      }
      renderCompareHistoryList();
    });
  }

  document
    .getElementById("algoHistorySelect")
    ?.addEventListener("change", () => {
      renderAlgoHistoryPanel();
    });
  document
    .getElementById("algoHistoryLimit")
    ?.addEventListener("change", () => {
      renderAlgoHistoryPanel();
    });
  document.getElementById("algoHistoryLimit")?.addEventListener("input", () => {
    renderAlgoHistoryPanel();
  });

  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".compare-algo-toggle");
    if (!btn) return;
    const block = btn.closest(".compare-algo-block");
    if (!block) return;
    const panel = block.querySelector(".compare-algo-panel");
    if (!panel) return;
    panel.classList.toggle("hidden");
    const collapsed = panel.classList.contains("hidden");
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.textContent = collapsed ? "Mở rộng" : "Thu gọn";
  });

  document
    .getElementById("compareStatsApply")
    ?.addEventListener("click", () => {
      const inp = document.getElementById("compareStatsN");
      let n = parseInt(inp && inp.value, 10);
      if (!Number.isFinite(n) || n < 1) n = 100;
      n = Math.max(1, Math.min(n, STATS_N_MAX));
      if (inp) inp.value = String(n);
      setCompareStatsWindowN(n);
      updateCompareStatsPanel();
    });
  document.getElementById("compareStatsN")?.addEventListener("change", () => {
    updateCompareStatsPanel();
  });

  (async () => {
    const compareStatsNInput = document.getElementById("compareStatsN");
    if (compareStatsNInput)
      compareStatsNInput.value = String(getCompareStatsWindowN());
    renderCompareHistoryList();
    const now = new Date();
    setFilterRange(new Date(now.getTime() - 60 * 60 * 1000), now);
    setFollow(true);
    await refresh();
    setInterval(refresh, POLL_MS);
  })();
})();
