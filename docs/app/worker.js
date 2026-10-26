/* 노후나침반 앱 — 계산 일꾼.
 * 브라우저 안에서 파이썬(Pyodide)과 numpy를 띄우고, engine.zip(웹 앱과 같은 retire_sim)을 풀어
 * retire_sim.lite.call(...)을 부른다. 화면은 멈추지 않고, 고객 입력은 기기 밖으로 나가지 않는다.
 *
 * 메시지:  {id, fn, form, extra}  →  {id, ok:true, out} | {id, ok:false, error}
 * 준비 상태: {type:"status", text} / {type:"ready", meta}
 */
const PYODIDE_VERSION = "0.27.7";
const params = new URLSearchParams(self.location.search);
const BASE = params.get("pyodide") || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let ready = null;

function status(text) { self.postMessage({ type: "status", text }); }

async function boot() {
  status("계산 엔진을 내려받는 중");
  importScripts(BASE + "pyodide.js");
  const py = await loadPyodide({ indexURL: BASE });
  status("numpy 준비 중");
  await py.loadPackage("numpy");
  status("노후나침반 엔진 여는 중");
  const [zipBuf, meta] = await Promise.all([
    fetch("engine.zip", { cache: "no-cache" }).then(r => { if (!r.ok) throw new Error("engine.zip " + r.status); return r.arrayBuffer(); }),
    fetch("engine.json", { cache: "no-cache" }).then(r => r.json()),
  ]);
  py.unpackArchive(zipBuf, "zip", { extractDir: "/engine" });
  py.runPython("import sys; sys.path.insert(0, '/engine'); from retire_sim import lite");
  const call = py.globals.get("lite").call;
  self.postMessage({ type: "ready", meta });
  return call;
}

self.onmessage = async (ev) => {
  const { id, fn, form, extra } = ev.data || {};
  if (fn === "__boot") { ready = ready || boot().catch(e => { ready = null; throw e; }); try { await ready; } catch (e) { self.postMessage({ type: "failed", error: String(e) }); } return; }
  try {
    ready = ready || boot();
    const call = await ready;
    const t0 = performance.now();
    const out = JSON.parse(call(fn, JSON.stringify(form || {}), JSON.stringify(extra || {})));
    self.postMessage({ id, ok: true, out, ms: Math.round(performance.now() - t0) });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e && e.message || e) });
  }
};
