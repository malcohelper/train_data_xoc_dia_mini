(function () {
  'use strict';

  const CFG_KEY = 'xd_ext_clicker_cfg_v1';
  const AMOUNTS = ['1k', '5k', '10k', '20k', '50k', '200k', '500k', '2m', '5m', '20m', '50m'];
  const POINTS = [
    ['chan', 'Chan'],
    ['le', 'Le'],
    ['chipLeft', 'Chip trai'],
    ['chipRight', 'Chip phai'],
  ];

  let captureKey = null;
  let busy = false;
  let cfg = loadCfg();
  let panel = null;

  function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function loadCfg() {
    try {
      const raw = JSON.parse(localStorage.getItem(CFG_KEY) || '{}');
      return {
        points: normalizePoints(raw.points),
        lastSeq: Number.isFinite(Number(raw.lastSeq)) ? Number(raw.lastSeq) : 0,
        lastIntent: normalizeIntent(raw.lastIntent),
      };
    } catch {
      return { points: {}, lastSeq: 0, lastIntent: null };
    }
  }

  function normalizePoints(points) {
    const out = {};
    if (!points || typeof points !== 'object') return out;
    for (const [key] of POINTS) {
      const p = points[key];
      const x = Number(p && p.x);
      const y = Number(p && p.y);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        out[key] = { x: Math.round(x), y: Math.round(y) };
      }
    }
    return out;
  }

  function normalizeIntent(intent) {
    if (!intent || typeof intent !== 'object') return null;
    const seq = Number(intent.seq || 0);
    return {
      seq: Number.isFinite(seq) ? Math.trunc(seq) : 0,
      side: intent.side === 'le' ? 'le' : 'chan',
      currentAmount: String(intent.currentAmount || ''),
      targetAmount: String(intent.targetAmount || ''),
      steps: Math.trunc(Number(intent.steps || 0)),
      betClicks: Math.max(1, Math.min(256, Math.trunc(Number(intent.betClicks || 1)))),
      intervalMs: Math.trunc(Number(intent.intervalMs || 250)),
      status: intent.status === 'fail' ? 'fail' : intent.status === 'ok' ? 'ok' : 'pending',
    };
  }

  function saveCfg() {
    localStorage.setItem(CFG_KEY, JSON.stringify(cfg));
  }

  function setMsg(text, ok) {
    const el = document.getElementById('xd-ext-msg');
    if (!el) return;
    el.textContent = text || '';
    el.style.color = ok ? '#86efac' : '#fca5a5';
  }

  function pointText(key) {
    const p = cfg.points[key];
    return p ? `${p.x}, ${p.y}` : 'unset';
  }

  function renderPanel() {
    if (!panel) return;
    for (const [key] of POINTS) {
      const el = panel.querySelector(`[data-point-text="${key}"]`);
      if (el) el.textContent = pointText(key);
    }
    const seqEl = panel.querySelector('[data-last-seq]');
    if (seqEl) seqEl.textContent = String(cfg.lastSeq || 0);
    const intentEl = panel.querySelector('[data-last-intent]');
    if (intentEl) intentEl.textContent = intentText(cfg.lastIntent);
  }

  function sideText(side) {
    return side === 'le' ? 'Le' : 'Chan';
  }

  function stepText(steps) {
    const n = Math.trunc(Number(steps || 0));
    if (n > 0) return `phai x${n}`;
    if (n < 0) return `trai x${Math.abs(n)}`;
    return 'khong doi chip';
  }

  function intentText(intent) {
    if (!intent) return 'Last: chua co intent';
    const status = intent.status === 'ok' ? 'OK' : intent.status === 'fail' ? 'FAIL' : 'RUN';
    return `Last #${intent.seq}: ${status} ${sideText(intent.side)} x${intent.betClicks} ${intent.currentAmount}->${intent.targetAmount} - ${stepText(intent.steps)} - ${intent.intervalMs}ms`;
  }

  function createPanel() {
    if (document.getElementById('xd-ext-clicker')) return;
    panel = document.createElement('div');
    panel.id = 'xd-ext-clicker';
    panel.innerHTML = `
      <div class="xd-title">XD Extension <span data-last-seq>0</span></div>
      ${POINTS.map(([key, label]) => `
        <div class="xd-row">
          <span>${label}: <b data-point-text="${key}">unset</b></span>
          <button data-set="${key}">Set</button>
          <button data-test="${key}">Test</button>
        </div>
      `).join('')}
      <div class="xd-last" data-last-intent>Last: chua co intent</div>
      <div id="xd-ext-msg"></div>
    `;
    const style = document.createElement('style');
    style.textContent = `
      #xd-ext-clicker {
        position: fixed;
        z-index: 2147483647;
        right: 12px;
        top: 96px;
        width: 250px;
        padding: 10px;
        border: 1px solid rgba(148, 163, 184, .55);
        border-radius: 8px;
        background: rgba(15, 23, 42, .92);
        color: #e5e7eb;
        font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        box-shadow: 0 16px 40px rgba(0, 0, 0, .35);
      }
      #xd-ext-clicker .xd-title {
        display: flex;
        justify-content: space-between;
        font-weight: 700;
        margin-bottom: 8px;
      }
      #xd-ext-clicker .xd-row {
        display: grid;
        grid-template-columns: 1fr 42px 42px;
        align-items: center;
        gap: 5px;
        margin: 5px 0;
      }
      #xd-ext-clicker button {
        border: 1px solid rgba(148, 163, 184, .45);
        border-radius: 5px;
        background: rgba(30, 41, 59, .95);
        color: #e5e7eb;
        padding: 3px 5px;
        font: inherit;
        cursor: pointer;
      }
      #xd-ext-clicker button:hover { background: rgba(51, 65, 85, .95); }
      #xd-ext-clicker .xd-last {
        margin-top: 8px;
        padding-top: 7px;
        border-top: 1px solid rgba(148, 163, 184, .28);
        color: #bfdbfe;
        line-height: 1.35;
      }
      #xd-ext-msg { min-height: 16px; margin-top: 8px; color: #94a3b8; }
    `;
    document.documentElement.appendChild(style);
    document.documentElement.appendChild(panel);

    panel.addEventListener('click', ev => {
      const setKey = ev.target && ev.target.dataset && ev.target.dataset.set;
      const testKey = ev.target && ev.target.dataset && ev.target.dataset.test;
      if (setKey) {
        captureKey = setKey;
        setMsg(`Click vao diem ${setKey} trong game`, true);
      }
      if (testKey) {
        testPoint(testKey);
      }
    });
    renderPanel();
  }

  function capturePoint(ev) {
    if (!captureKey) return;
    if (panel && panel.contains(ev.target)) return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
    cfg.points[captureKey] = {
      x: Math.round(ev.clientX),
      y: Math.round(ev.clientY),
    };
    const savedKey = captureKey;
    captureKey = null;
    saveCfg();
    renderPanel();
    setMsg(`Saved ${savedKey}: ${pointText(savedKey)}`, true);
  }

  function dispatchAt(point) {
    const x = Math.round(point.x);
    const y = Math.round(point.y);
    const target = document.elementFromPoint(x, y) || document.body;
    const base = {
      bubbles: true,
      cancelable: true,
      composed: true,
      view: window,
      clientX: x,
      clientY: y,
      screenX: Math.round(window.screenX + x),
      screenY: Math.round(window.screenY + y),
      button: 0,
    };
    target.dispatchEvent(new MouseEvent('mousemove', { ...base, buttons: 0 }));
    if (typeof PointerEvent === 'function') {
      target.dispatchEvent(new PointerEvent('pointermove', {
        ...base,
        buttons: 0,
        pointerId: 1,
        pointerType: 'mouse',
        isPrimary: true,
      }));
      target.dispatchEvent(new PointerEvent('pointerdown', {
        ...base,
        buttons: 1,
        pointerId: 1,
        pointerType: 'mouse',
        isPrimary: true,
        pressure: 0.5,
      }));
    }
    target.dispatchEvent(new MouseEvent('mousedown', { ...base, buttons: 1 }));
    target.dispatchEvent(new MouseEvent('mouseup', { ...base, buttons: 0 }));
    if (typeof PointerEvent === 'function') {
      target.dispatchEvent(new PointerEvent('pointerup', {
        ...base,
        buttons: 0,
        pointerId: 1,
        pointerType: 'mouse',
        isPrimary: true,
        pressure: 0,
      }));
    }
    target.dispatchEvent(new MouseEvent('click', { ...base, buttons: 0 }));
  }

  async function testPoint(key) {
    const point = cfg.points[key];
    if (!point) {
      setMsg(`Missing ${key}`, false);
      return;
    }
    dispatchAt(point);
    setMsg(`Test ${key}`, true);
  }

  async function runIntent(intent) {
    if (busy) return { success: false, message: 'content script busy' };
    busy = true;
    const seq = Number(intent.seq || 0);
    const side = intent.side === 'le' ? 'le' : 'chan';
    const steps = Math.trunc(Number(intent.steps || 0));
    const betClicks = Math.max(1, Math.min(256, Math.trunc(Number(intent.betClicks || 1))));
    const interval = Math.max(50, Math.min(600, Math.trunc(Number(intent.intervalMs || 250))));
    const sidePoint = cfg.points[side];
    const stepKey = steps > 0 ? 'chipRight' : 'chipLeft';
    const stepPoint = steps === 0 ? null : cfg.points[stepKey];

    try {
      if (!sidePoint) throw new Error(`Missing point ${side}`);
      if (steps !== 0 && !stepPoint) throw new Error(`Missing point ${stepKey}`);
      cfg.lastIntent = normalizeIntent({ ...intent, status: 'pending' });
      saveCfg();
      renderPanel();
      setMsg(`Intent #${seq}: ${side} ${intent.currentAmount}->${intent.targetAmount}`, true);
      for (let i = 0; i < Math.abs(steps); i += 1) {
        dispatchAt(stepPoint);
        await wait(interval);
      }
      for (let i = 0; i < betClicks; i += 1) {
        dispatchAt(sidePoint);
        if (i < betClicks - 1) await wait(interval);
      }
      cfg.lastSeq = seq;
      cfg.lastIntent = normalizeIntent({ ...intent, status: 'ok' });
      saveCfg();
      renderPanel();
      setMsg(`Done #${seq}: ${side} x${betClicks}`, true);
      return { success: true, message: `clicked ${side} x${betClicks}` };
    } catch (err) {
      cfg.lastSeq = seq;
      cfg.lastIntent = normalizeIntent({ ...intent, status: 'fail' });
      saveCfg();
      renderPanel();
      const message = err && err.message ? err.message : String(err);
      setMsg(message, false);
      return { success: false, message };
    } finally {
      busy = false;
    }
  }

  function installMessageHandler() {
    const ext = typeof browser !== 'undefined' ? browser : chrome;
    if (!ext || !ext.runtime || !ext.runtime.onMessage) return;
    ext.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (!message || message.type !== 'xd-run-intent') return undefined;
      const promise = runIntent(message.intent || {})
        .catch(err => ({
          success: false,
          message: err && err.message ? err.message : String(err),
        }));
      if (typeof sendResponse === 'function') {
        promise.then(result => sendResponse(result));
        return true;
      }
      return promise;
    });
  }

  createPanel();
  document.addEventListener('pointerdown', capturePoint, true);
  installMessageHandler();
})();
