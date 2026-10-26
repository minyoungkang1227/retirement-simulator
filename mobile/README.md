# 노후나침반 설치형 앱 (가벼운 모드)

`docs/app/`의 태블릿 화면을 **설치형 앱**으로 감싼 것입니다(Capacitor 7.6.9). 계산 엔진 런타임(Pyodide 0.27.7 + numpy, 약 17MB)을
앱 안에 넣어서 **처음 실행부터 인터넷 없이** 계산하고, 고객 입력은 기기 밖으로 나가지 않습니다. 화면과 엔진은 웹과 같은 파일이라
`docs/app/`이나 `retire_sim/`을 고치면 앱도 같이 바뀝니다.

## 안드로이드 (지금 바로)

빌드는 GitHub가 합니다(`.github/workflows/android.yml`). `main`에 올리면 몇 분 뒤 저장소 **Releases**에
`nohu-compass-<버전>-<빌드번호>.apk`가 올라갑니다.

1. 태블릿에서 저장소의 Releases 페이지를 열고 `.apk`를 눌러 내려받습니다.
2. 처음 한 번 "출처를 알 수 없는 앱 설치"를 허용합니다(설정 → 보안, 또는 설치 화면 안내).
3. 설치 → 홈 화면의 **노후나침반** 아이콘으로 엽니다.

다음 빌드부터는 같은 방법으로 받으면 기존 앱 위에 업데이트됩니다(서명 키 `debug.keystore` 고정, 빌드 번호가 올라감).

## Play 스토어 (다음 단계)

- Google Play 개발자 계정(1회 25달러)이 필요합니다.
- 시험용 키(`debug.keystore`)가 아니라 **정식 업로드 키**로 서명한 AAB가 필요합니다. 키는 저장소에 올리지 말고
  GitHub Secrets에 넣어 빌드에서만 쓰게 해야 합니다.
- 스토어 등록정보: 개인정보처리방침 주소(입력을 저장·전송하지 않는다는 내용), 스크린샷, 금융 앱 관련 신고 항목.

## iPad / 아이폰 (다음 단계)

- Mac + Xcode와 Apple Developer Program(연 99달러)이 필요합니다.
- Mac에서: `cd mobile && npm install && npm run ios && npx cap open ios` → Xcode에서 서명 팀을 고르고 기기에 설치하거나 App Store Connect로 올립니다.
- 앱 안에서 파이썬을 받지 않고 **앱에 넣어 둔 런타임만** 쓰므로, 실행 코드 내려받기를 금지하는 심사 기준(2.5.2)에 걸리지 않게 해 두었습니다.

## 로컬에서

```bash
cd mobile
npm install
npm run www                    # www/ 만들기 (런타임을 jsDelivr에서 받음, PYODIDE_DIR=폴더 로 대체 가능)
npm run android                # 안드로이드 프로젝트 생성 (빌드는 Android Studio 또는 ./gradlew assembleDebug)
```

`www/`, `android/`, `ios/`, `node_modules/`는 만들어지는 폴더라 저장소에 올리지 않습니다.
