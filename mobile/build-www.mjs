// 설치형 앱의 화면 묶음(www/)을 만든다.
//   1) ../docs/app (웹과 같은 화면·엔진 묶음)을 복사
//   2) 계산 엔진 런타임(Pyodide + numpy, 약 17MB)을 www/pyodide 에 넣는다 — 앱은 인터넷 없이 첫 계산부터 된다
//   3) index.html 의 BUNDLED_PYODIDE 를 "./pyodide/"로 바꾼다
// 런타임은 PYODIDE_DIR(이미 받아 둔 폴더)이 있으면 거기서, 없으면 jsDelivr에서 받는다.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(HERE, "..", "docs", "app");
const OUT = path.join(HERE, "www");
const PYODIDE_VERSION = "0.27.7";
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const CORE = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"];

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(path.join(OUT, "pyodide"), { recursive: true });
for (const f of fs.readdirSync(SRC)) {
  if (f === "sw.js") continue;                       // 앱에는 오프라인 캐시가 필요 없다(파일이 앱 안에 있다)
  fs.copyFileSync(path.join(SRC, f), path.join(OUT, f));
}

const idx = path.join(OUT, "index.html");
const html = fs.readFileSync(idx, "utf8");
if (!html.includes('const BUNDLED_PYODIDE = "";')) throw new Error("index.html 에 BUNDLED_PYODIDE 표시가 없습니다");
fs.writeFileSync(idx, html.replace('const BUNDLED_PYODIDE = "";', 'const BUNDLED_PYODIDE = "./pyodide/";'));

async function get(name) {
  const dst = path.join(OUT, "pyodide", name);
  if (process.env.PYODIDE_DIR) { fs.copyFileSync(path.join(process.env.PYODIDE_DIR, name), dst); return; }
  const r = await fetch(CDN + name);
  if (!r.ok) throw new Error(`${name} 받기 실패: ${r.status}`);
  fs.writeFileSync(dst, Buffer.from(await r.arrayBuffer()));
}
for (const f of CORE) await get(f);
const lock = JSON.parse(fs.readFileSync(path.join(OUT, "pyodide", "pyodide-lock.json"), "utf8"));
const numpy = lock.packages.numpy.file_name;
await get(numpy);

const size = (d) => fs.readdirSync(d).reduce((s, f) => s + fs.statSync(path.join(d, f)).size, 0);
console.log(`www 준비 완료: 화면 ${(size(OUT) / 1e3).toFixed(0)}KB + 엔진 런타임 ${(size(path.join(OUT, "pyodide")) / 1e6).toFixed(1)}MB (Pyodide ${PYODIDE_VERSION}, ${numpy})`);
