/* 노후나침반 앱 — 오프라인용 캐시. 한 번 연 뒤에는 인터넷 없이도 열리고 계산된다.
 * 앱 파일은 '네트워크 먼저'(새 버전을 바로 받음), 계산 엔진 런타임(Pyodide·글꼴)은 '캐시 먼저'(17MB를 매번 받지 않음). */
const CACHE = "nohu-app-v1";
const SHELL = ["./", "index.html", "worker.js", "engine.zip", "engine.json", "manifest.webmanifest", "icon.svg", "icon-180.png", "icon-192.png", "icon-512.png"];

self.addEventListener("install", (e) => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  const runtime = url.hostname === "cdn.jsdelivr.net";
  if (runtime) {
    e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return res;
    })));
  } else if (url.origin === location.origin) {
    e.respondWith(fetch(e.request).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return res;
    }).catch(() => caches.match(e.request, { ignoreSearch: true })));
  }
});
