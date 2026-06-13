(function () {
  'use strict';

  const API_BASE = 'http://127.0.0.1:8000';
  const GAME_URLS = [
    '*://gamebai.b5richkids.net/*',
  ];

  const browserApi = typeof browser !== 'undefined' ? browser : null;
  const ext = browserApi || chrome;
  let lastSeq = 0;
  let busy = false;

  function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async function postResult(payload) {
    try {
      await fetch(`${API_BASE}/api/extension/result`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      console.warn('[XD Extension] result post failed', err);
    }
  }

  function tabsQuery(query) {
    if (browserApi) {
      return browserApi.tabs.query(query).catch(() => []);
    }
    return new Promise(resolve => {
      try {
        ext.tabs.query(query, tabs => resolve(tabs || []));
      } catch {
        resolve([]);
      }
    });
  }

  function sendToTab(tabId, message) {
    if (browserApi) {
      const timeout = wait(7000).then(() => ({ success: false, message: 'content script timeout' }));
      const send = browserApi.tabs.sendMessage(tabId, message)
        .then(response => response || { success: false, message: 'content script empty response' })
        .catch(err => ({ success: false, message: err && err.message ? err.message : String(err) }));
      return Promise.race([send, timeout]);
    }
    return new Promise(resolve => {
      let settled = false;
      const finish = value => {
        if (!settled) {
          settled = true;
          resolve(value);
        }
      };
      try {
        ext.tabs.sendMessage(tabId, message, response => {
          finish(response || { success: false, message: 'content script empty response' });
        });
      } catch (err) {
        finish({ success: false, message: String(err) });
      }
      setTimeout(() => finish({ success: false, message: 'content script timeout' }), 7000);
    });
  }

  async function deliverIntent(intent) {
    const seq = Number(intent && intent.seq || 0);
    const tabs = await tabsQuery({ url: GAME_URLS });
    if (!tabs.length) {
      await postResult({
        seq,
        success: false,
        message: 'khong tim thay tab game',
        side: intent.side,
        targetAmount: intent.targetAmount,
      });
      return;
    }

    const tab = tabs.find(t => t.active) || tabs[0];
    const response = await sendToTab(tab.id, { type: 'xd-run-intent', intent });
    await postResult({
      seq,
      success: !!(response && response.success),
      message: response && response.message ? response.message : 'no response',
      side: intent.side,
      targetAmount: intent.targetAmount,
    });
  }

  async function poll() {
    if (busy) return;
    busy = true;
    try {
      const res = await fetch(`${API_BASE}/api/extension/next?since=${encodeURIComponent(lastSeq)}`, {
        cache: 'no-store',
      });
      if (!res.ok) return;
      const data = await res.json();
      const serverSeq = Number(data && data.seq || 0);
      if (serverSeq < lastSeq) {
        console.log('[XD Extension] server sequence reset', { lastSeq, serverSeq });
        lastSeq = 0;
      }
      if (data && data.hasIntent && data.intent && Number(data.intent.seq) > lastSeq) {
        lastSeq = Number(data.intent.seq);
        await deliverIntent(data.intent);
      }
    } catch {
      // Server can be off while the extension is enabled.
    } finally {
      busy = false;
    }
  }

  setInterval(poll, 350);
  wait(500).then(poll);
})();
