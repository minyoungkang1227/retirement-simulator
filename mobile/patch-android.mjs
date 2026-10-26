// npx cap add android 로 만든 안드로이드 프로젝트에 버전 번호를 넣는다.
//   versionCode = APP_BUILD (GitHub Actions 실행 번호) — 올라가야 기존 설치 위에 업데이트된다
//   versionName = package.json 버전
import fs from "node:fs";
const pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
const f = "android/app/build.gradle";
let g = fs.readFileSync(f, "utf8");
const code = parseInt(process.env.APP_BUILD || "1", 10);
g = g.replace(/versionCode \d+/, `versionCode ${code}`).replace(/versionName "[^"]*"/, `versionName "${pkg.version}"`);
fs.writeFileSync(f, g);
console.log(`android 버전: ${pkg.version} (${code})`);
