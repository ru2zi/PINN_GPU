# 실행 기록

단계별 실행 결과와 그 과정에서 확정된 설계 결정을 적는다(spec 12절). 모든 결과는 `./run_all.sh`로 다시 만들 수 있다.

## 0단계: 환경과 승인 (2026-10-11)
- `requirements.txt`, `records/approval.md`, `records/instrument.md`, `configs/data.yaml`.
- 계산 환경: Python 3.10.12, torch 2.1.0a0, CPU affinity 40, `OMP_NUM_THREADS=1`, 공유 GPU 4장. 대조용 torchdiffeq는 `github_code/torchdiffeq`(0.2.5)를 경로로 불러온다.
- 미기록: 측정 서버의 serial·driver·실내 온도, 담당 교수 확인.
- 8단계 전에 git 저장소가 필요하다(`ACCEPTANCE.md` commit hash).

## 1단계: inventory, 형식·오염 검사, split (`gputherm.pipeline.inventory`)
- run 51, md5 중복 0, 결측 0, phase 순서 정상, 샘플 간격 중앙값 10.08–10.10 ms, 2.4절 parsing 검증값 일치, 제외 0.
- warmup plateau 49.0–51.0 °C, cooldown 끝 25.2–28.0 °C. 0.5 s 넘는 샘플 공백은 I02에만(최대 4.5 s).
- window 수: train 2,090(학습 4,169), validation 1,034, test interp 1,689, test shift 1,030. coverage 미달 0, 이력 부족 0.
- test run에는 형식 검사와 window 수만 계산했다.

## 2단계: 측정 불확도 (`gputherm.pipeline.uncertainty`)
- u_res 0.289 K, u_repeat 0.153 K(조건별 s_i 0.015–0.247 K, 자유도 18), **u_c 0.327 K**, 2u_c 0.653 K → ε_Q = ε_pk = 1.0 K.
- drift 진단: 조건 내 control 회귀로 설명되는 몫 19 %(r = 0.44). u_c에 더하지 않는다.
- u_timing은 5단계에서 계산한다.

## 3단계: 물리 모델 구현과 검증 (대표 parameter)
대표 parameter: τ_fast 0.4 s, τ_slow 41 s, R_ss 0.038 K/W, R_gh = 0.77 R_ss, T_a 27.1 °C.

| 검증 | 결과 |
|---|---|
| 구조적 식별 가능성 | rank 일반형 4/6, 사다리형 4/4(역산 해 1개), 사다리형+α 4/5 |
| 인과성, coverage 필터 | 정수·반정수 초 시작, 이력 부족 window, 합성 15 s 공백 모두 통과 |
| 6.1 해석해 대조 | 상태 ≤ 1.3e-8 K, 구간 평균 ≤ 7.5e-11 K, 시간 가변 α 1.8e-8 K |
| 6.2 RK4 / 입력 격자 | order 4.08–4.15, dt 0.1 s 오차 2.7e-4 K / 0.25 대 0.1 s 0.026 K, 0.5 대 0.1 s 0.27 K |
| 6.3 에너지 수지 | ≤ 1.0e-14 |
| 6.5 정상상태 | 600 W 50.0 °C, 해석 관계와 일치 |
| 6.7 torch 값 / 미분 | ≤ 7.2e-13 K / ≤ 4.2e-9 |
| RC 추정 구현 | batch와 개별 일치, 무효 정답 배제, float32 거부, 실패 초기값 격리, 종료 사유 기록 |

## 4단계 (1): 시간 정렬 검증 (`gputherm.verification.alignment`)
- 단계 1(참 parameter) 시간 정렬 오차 0.0017 K(기준 0.02 K) → 통과. 단계 2/3/4: 0.011 / 0.019 / 0.028 K.
- 출발 상태: last_0p1(세 정의의 차이 < 0.01 K).
- 정수 양자화가 추정을 편향시킨다: run 하나의 fit에서 τ_slow 41.0 → 33.4 s, T_a +0.2 K(초기값 8개 모두 수렴).
- α_ref 동등성(관측기 포함): 평균 0.0014 K, 최대 0.0024 K.

## 확정된 설계 결정 (근거는 spec 해당 절)
- **RC 구조**: 일반형 2-node는 구조적으로 식별되지 않아 사다리형(R_ga → ∞)을 쓴다(5.1).
- **α_ref**: α < 1에서 역산 해가 두 개일 수 있어, α = 1에서 추정한 뒤 큰 근의 가지로 0.75에 옮기고 관측기 포함 예측 차이로 확인한다(5.1).
- **시간 척도**: cooldown은 지수 2개(τ_fast 0.39 s, τ_slow 41 s, fast 비중 0.77)로 기술된다. 1-node 대안은 die 열용량을 0으로 둔 준정적 1-node다(5.3, 5.5).
- **출력 정의**: 0.5 s 구간 시간 평균(적분). 끝점 평균과의 차이는 첫 구간 1.16 K, 첫 60 s RMS 0.11 K(6.1).
- **수치 해법**: Van Loan 블록 지수로 A_d, B_d, G, H를 구하고(뺄셈 방식은 미분 정확도가 2e-5로 떨어졌다), window 예측은 합성곱으로 계산한다(train 손실+미분 0.6 s)(5.6).
- **불확도**: u_control은 u_repeat와 drift를 공유하므로 합성하지 않는다(3).
- **look-back**: L ≤ 180 s(본 구간 전 최소 이력 186 s)(7).
- **데이터**: I01-32B-C는 원샘플이 없어 쓰지 않는다. shift test에는 모델 결과 미열람 조건이 없다(2.5, 4.2).

## 코드 구조
`gputherm/`(config, data, physics, estimation, pipeline, verification). 이전 `src/`에서 함수 본문을 그대로 옮겼고, 전후 결과 파일 20개가 같음을 확인했다.

## 다음
4단계 (2): 6.4 식별 가능성 gate → 5단계 RC 추정과 구조 선택.
