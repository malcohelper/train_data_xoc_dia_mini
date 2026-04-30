// Xoc Dia Analytics - client-side renderer.
//
// Pulls every finished round from `/api/rounds.json` (served by
// `analytics/serve.py`), converts each `dice_result` into a (red, type)
// pair, and drives the three panels in `index.html`:
//   1. Chẵn / Lẻ progress card
//   2. Per-dice-combo (vị) stats cards
//   3. Big Road (6-row Baccarat-style bead column)
//
// A background poll refreshes the data every 3 s so the page stays in
// sync with a live capture session.

(() => {
  "use strict";

  const POLL_MS = 3000;

  // Game-side dice_result string -> red-pip count.
  // The game lives in Vietnamese terms:
  //   "4_red"   = 4 đỏ / 0 trắng -> red=4, Chẵn
  //   "3r_1w"   = 3 đỏ 1 trắng  -> red=3, Lẻ
  //   "2w_2r"   = 2 đỏ 2 trắng  -> red=2, Chẵn
  //   "3w_1r"   = 3 trắng 1 đỏ -> red=1, Lẻ
  //   "4_white" = 0 đỏ / 4 trắng -> red=0, Chẵn
  const DICE_TO_RED = {
    "4_red": 4,
    "3r_1w": 3,
    "2w_2r": 2,
    "3w_1r": 1,
    "4_white": 0,
  };

  // Order matches mockup: 4 TRẮNG, 3T1Đ, 2T2Đ, 3Đ1T, 4 ĐỎ (red ascending).
  const VI_LABELS = ["4 TRẮNG", "3 TRẮNG 1 ĐỎ", "2 TRẮNG 2 ĐỎ", "3 ĐỎ 1 TRẮNG", "4 ĐỎ"];

  /** @type {Array<{red:number,type:"chan"|"le",time:Date,round_id:string}>} */
  let masterData = [];
  let autoFollow = true; // whenever new rounds arrive, snap filter to [oldest, now]

  // ---------- fetching ----------

  async function fetchRounds() {
    const resp = await fetch("/api/rounds.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  // round_id is "YYYYMMDD_HHMMSS" in the local timezone of the capture
  // machine. `started_at` (ISO) is more precise but the round_id is
  // always present and parsable. We prefer `started_at` when available.
  function roundIdToDate(rid) {
    const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(rid || "");
    if (!m) return null;
    return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  }

  function toItem(round) {
    const red = DICE_TO_RED[round.dice_result];
    if (red === undefined) return null; // unfinished / unknown dice
    let time = null;
    if (round.started_at) {
      const d = new Date(round.started_at);
      if (!isNaN(d)) time = d;
    }
    if (!time) time = roundIdToDate(round.round_id);
    if (!time) return null;
    return {
      red,
      type: red % 2 === 0 ? "chan" : "le",
      time,
      round_id: round.round_id || "",
    };
  }

  // ---------- date helpers ----------

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

  // ---------- rendering ----------

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
    // matrix[row][col] = type or undefined. Each streak fills a column
    // top-down. When a streak is longer than ROWS (6) it wraps to the
    // next column at row 0 - matching the game's own BẢNG CẦU. This
    // differs from the classic Baccarat "dragon tail" (which hooks
    // right at the last row); the game in question wraps to row 0.
    const matrix = Array.from({ length: ROWS }, () => []);
    let curR = 0;
    let curC = 0;
    let prevType = null;

    // Advance ``curC`` to the next column whose row 0 is empty. Keeps
    // us from overwriting an earlier streak when a long one wraps or
    // when a new streak starts.
    const advanceToEmptyColumn = () => {
      curC++;
      while (matrix[0][curC]) curC++;
    };

    list.forEach((item) => {
      if (item.type !== prevType) {
        // New streak: next empty column, row 0.
        curR = 0;
        if (prevType !== null) {
          advanceToEmptyColumn();
        }
      } else {
        // Same streak: continue down until the column is full, then
        // wrap to row 0 of the next empty column.
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

  // ---------- status + polling ----------

  let lastFetchOk = false;
  let lastFetchAt = null;

  function setStatus(text, ok = true) {
    document.getElementById("statusLine").innerText = text;
    const badge = document.getElementById("liveBadge");
    if (ok) {
      badge.classList.remove("text-red-500");
      badge.classList.add("text-emerald-500");
      badge.innerText = "● System Online";
    } else {
      badge.classList.remove("text-emerald-500");
      badge.classList.add("text-red-500");
      badge.innerText = "● Offline";
    }
  }

  async function refresh() {
    try {
      const raw = await fetchRounds();
      const items = raw.map(toItem).filter(Boolean);
      // Sort by time ascending (rounds_dir sort is already ~lexicographic
      // on round_id, which is also chronological, but be defensive).
      items.sort((a, b) => a.time - b.time);

      const prevCount = masterData.length;
      masterData = items;
      lastFetchOk = true;
      lastFetchAt = new Date();

      // Auto-follow: when a new round lands and the user hasn't pinned
      // a range manually, slide the filter end forward so the Big Road
      // scrolls with live capture.
      if (autoFollow && masterData.length) {
        const { from, to } = defaultRange();
        setFilterRange(from, to);
      }

      render();
      const newlyAdded = Math.max(0, masterData.length - prevCount);
      const when = lastFetchAt.toLocaleTimeString();
      const suffix = newlyAdded ? ` · +${newlyAdded} mới` : "";
      setStatus(
        `Cập nhật ${when} · ${masterData.length} round tổng${suffix}`,
        true
      );
    } catch (err) {
      lastFetchOk = false;
      setStatus(`Lỗi tải /api/rounds.json: ${err}`, false);
    }
  }

  // ---------- wiring ----------

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

  document.getElementById("applyBtn").addEventListener("click", () => {
    setFollow(false);
    render();
  });
  document.getElementById("resetBtn").addEventListener("click", () => {
    const newState = !autoFollow;
    setFollow(newState);
    if (newState && masterData.length) {
      const { from, to } = defaultRange();
      setFilterRange(from, to);
      render();
    }
  });
  // Typing in either date input also disables auto-follow so the user
  // can pin a historical window without it being yanked away every 3 s.
  ["fromDate", "toDate"].forEach((id) => {
    document.getElementById(id).addEventListener("change", () => setFollow(false));
  });

  // Initial load + poll loop.
  (async () => {
    // Seed filter to "last hour" before first fetch so the empty-state
    // looks sane; refresh() will snap it to the real range.
    const now = new Date();
    setFilterRange(new Date(now.getTime() - 60 * 60 * 1000), now);
    setFollow(true);
    await refresh();
    setInterval(refresh, POLL_MS);
  })();
})();
