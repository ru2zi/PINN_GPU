# EE5106 PIML 중간 프로젝트 실험 명세

**GPU die 온도의 60 s 예측: 검증된 사다리형 RC baseline과 신경망 closure의 비교**

- 작성일: 2026-10-11
- **측정 장비**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition 1장(96 GB GDDR7, TDP 600 W), KENTECH Real LAB. 모든 run은 power cap 600 W다.
- **계산 환경**: Intel Xeon Platinum 8480C(작업 세션 CPU affinity 40, `OMP_NUM_THREADS=1`), RAM 2 TB, 공유 GPU 4장. Python 3.10.12, PyTorch 2.1.0a0, NumPy 1.22.2, SciPy 1.11.1, pandas 1.5.3(`requirements.txt`).
- **데이터**: `PINN_DATA/`(읽기 전용). 17개 조건 × 3 run = 51 run. 조건 정의는 `PINN_DATA/conditions.csv`.
- **승인**: 측정자이자 데이터 소유자인 작성자가 이 과제에의 사용을 승인한다(`records/approval.md`).
- **준거 문서**: `hw/midterm_project_guidelines_FA26.pdf`(가이드), `hw/midterm_project_template_FA26.ipynb`(템플릿), `PINN_DATA/docs/데이터센터_실험계획서_0315(수정필요).pdf`(실험계획서).
- 단계별 산출물과 통과 조건은 12절에 있다. 실행 기록은 `results/LOG.md`에 남긴다.

---

## 용어

| 용어 | 정의 |
|---|---|
| 조건 | 같은 설정으로 측정한 workload. 예: `I01-8B-B` = Llama-3.1-8B 추론, 동시 요청 32 |
| run | 한 조건의 측정 1회(약 25분). 조건마다 r01–r03. 독립 단위 |
| 본 구간 | phase가 `train`, `checkpoint`, `inference*`, `idle_between`인 구간 |
| window | 예측 1회의 단위. 시작 시각 t₀, horizon 60 s |
| 사전 탐색 | 이 명세 이전에 600 W 45 run에 대해 선형 RC 결과와 요약 통계를 본 분석(0절, 4.2절) |
| Persist | 마지막 측정 온도를 horizon 동안 유지하는 기준 |
| RC | 검증된 baseline. 사다리형 2-node 열 회로, parameter 상수. closure를 뺀 ablation이기도 하다 |
| RC-NN | 제안 모델. RC에 신경망 closure를 더한 모델 |
| DLinear-X | 선형 data-only comparator |
| u_c | 측정 불확도(3절) |
| skill | 1 − RMSE_모델 / RMSE_Persist (같은 window) |
| 모델 결과 미열람 subset | 어떤 모델의 결과도 본 적이 없는 test 조건 I01-32B-A/B. 요약 통계는 형식 검사에서 보았다 |
| T_g, T_h, T_a | die 온도(측정), 두 번째 node 온도(숨은 상태), 유효 주변 온도 |

M과 숫자(M1–M35)는 가이드 rubric 항목만 뜻한다.

---

## 0. 규칙

1. **Test 잠금.** `ACCEPTANCE.md`를 날짜와 함께 commit하기 전에는 test run(4.2절)에 어떤 모델도 적용하지 않고, 지표와 그림을 만들지 않는다. 평가 코드의 guard로 강제한다(9.5절). test run에는 형식 검사와 window 수 계산만 한다.
2. **선택은 train과 validation으로만.** hyperparameter, closure, look-back 길이, 시간 지연, stopping epoch가 해당한다.
3. **원시 데이터 불변.** 파생 결과는 `results/`에 새 파일로 쓴다.
4. **새 측정 없음.** 기존 51 run만 쓴다.
5. **열람 이력 공개.** 사전 탐색에서 test run을 포함한 45 run의 선형 RC 결과와 요약 통계를 보았다. 신경망 모델은 어떤 run에도 적용한 적이 없다. I01-32B run은 형식 검사의 요약 통계(평균 power·온도, 메모리, phase 길이)만 보았다. 이 사실을 보고서의 split 절, 한계 절, `ACCEPTANCE.md`에 적는다. 2.4절 수치는 parsing 검증용이며 tuning에 쓰지 않는다.

---

## 1. 질문과 claim (M1–M4)

### 1.1 질문
측정 power와 계획된 부하로 GPU die 온도를 60 s 앞까지 예측할 때, 검증된 2-node RC에 workload 정보를 받는 신경망 closure를 더하면 처음 보는 workload와 burst 부하에서 더 정확해지는가?

### 1.2 결정과 사용자 (M1)
- **사용자**: AI 학습·추론 서버 운영자.
- **결정**: 다음 60 s의 부하(계획된 power profile)를 배치하기 전에 die 온도의 peak와 swing을 예측하고, 기준을 넘으면 배치를 바꾼다(예: burst 대신 batching).
- **입력 가정**: horizon의 power, GPU 사용률, 메모리 사용률은 계획값으로 주어진다. 평가에서는 측정값을 계획값으로 쓴다. 따라서 입증 대상은 **정확한 미래 부하 정보가 주어진 조건부 예측**이다. 계획 정보가 없을 때의 성능은 9.6절에서 보인다.
- **대상 밖**: 수명 예측, throttling 판정(측정 최대 52 °C), 다른 GPU나 냉각 조건으로의 이전, 설계 등급 열 해석.

### 1.3 Claim (M2)
**주 claim**
> 단일 RTX PRO 6000(cap 600 W, 실험실 실내)에서 정확한 미래 부하 정보가 주어질 때, RC-NN은 같은 목표로 추정한 RC보다 60 s 앞 die 온도를 더 정확하게 예측한다. interp와 shift 각각에서 조건 단위 판정 규칙(9.3절)을 통과한다. interp에서 RMSE와 horizon peak 오차가 5개 조건 중 4개 이상과 조건 평균에서 ε_Q, ε_pk 이하다. physics check(9.2절)를 위반하지 않고, 예측 1회를 C ms 이하에 끝낸다.

**보조 claim (comparator)**
> RC-NN은 interp와 shift에서 조건 단위로 Persist보다 정확하고, shift에서 DLinear-X보다 정확하다.

두 claim은 따로 판정한다. shift의 절대 오차는 기준과 비교해 보고하되 claim의 성패를 정하지 않는다(stress test). ε_Q, ε_pk, C와 판정 규칙은 9.3절에 있으며 `ACCEPTANCE.md`에 고정한다.

### 1.4 범위와 독립 단위 (M3)
- **범위 D**: train·validation 27 run의 전체 기록 기준 die 온도 25–52 °C, 보드 power 10–679 W, 2절의 workload. test의 범위 이탈은 coverage(범위 밖 sample 비율, 최대 초과량)로 보고한다.
- **독립 단위**: run. 같은 조건의 세 run은 연속 측정이라 강하게 상관되므로 split과 판정을 조건 단위로 한다. 실효 표본 수는 조건 수(interp 5, shift 3)에 가깝고, Q7B·8B·32B 조건은 같은 workload 구조를 공유한다.
- **주 QoI**: 60 s horizon RMSE [K], horizon peak 오차 [K]. **보조 QoI**: horizon swing 오차 [K].

### 1.5 증거 사슬

| 가이드 단계 | 이 프로젝트 | 위치 |
|---|---|---|
| question | 60 s 앞 peak·swing 예측에 의한 부하 배치 | 1절 |
| physics | 2-node 에너지 보존식(식별 가능한 사다리형) | 5절 |
| verified reference | 수치 해법: 해석해·독립 구현 대조, 수렴, 입력 격자, 에너지 수지. 데이터: NVML 기록과 u_c | 3절, 6절 |
| data | 직접 측정한 wet-lab 51 run, 설계한 dry-lab 식별 가능성 실험 | 2절, 6.4절 |
| surrogate | RC-NN. 비교: RC(closure 제거), DLinear-X, Persist | 8절 |
| held-out test | 조건 단위 interp·shift test, 모델 결과 미열람 subset 별도 보고 | 4절, 9절 |
| bounded conclusion | 이 GPU, cap 600 W, 측정된 조건 | 13절 |

---

## 2. 데이터와 wet-lab 기록 (M10, M11)

### 2.1 파일 구조
폴더 이름은 실험 내용을 나타내도록 정했다(파일 내용 불변, md5 집합 동일, `PINN_DATA/RENAME_LOG.csv`).

```
PINN_DATA/
├── conditions.csv, RENAME_LOG.csv, docs/
└── single_gpu_rtxpro6000_cap600W/<조건ID>_<train|infer>_<작업>_<모델>_<변형>/rNN/{gpu_samples,phase_summary}.csv
```

| 조건 | 폴더 |
|---|---|
| T02-A / T02-B | `T02-A_train_lora-bf16_llama3.1-8b` / `T02-B_train_qlora-4bit_llama3.1-8b` |
| T02-32B-A / A2 | `T02-32B-A_train_lora_qwen2.5-32b-instruct` / `T02-32B-A2_train_lora_qwen2.5-32b-instruct_seq2048` |
| T04-A / T04-D | `T04-A_train_diffusion_sdxl_batch4` / `T04-D_train_diffusion_flux.1-dev_batch2` |
| I01-{Q7B,8B,32B}-A | `I01-<모델>-A_infer_llm_<모델명>_conc1_saturated` |
| I01-{Q7B,8B,32B}-B | `I01-<모델>-B_infer_llm_<모델명>_conc32_saturated` |
| I01-{Q7B,8B}-C | `I01-<모델>-C_infer_llm_<모델명>_conc1_burst-8req-per-10s` |
| I02-A / B / C | `I02-A_infer_imagegen_sdxl_batch1` / `…_batch4` / `…_burst-4img-per-15s` |

고유 run 51개가 아니면 멈춘다.

### 2.2 조건 정의
조건 ID는 실험계획서 3.2절을 따른다(T02 LLM LoRA 미세조정, T04 diffusion 학습, I01 LLM 추론, I02 이미지 생성 추론). A–F는 원 실험 저장소 `run.py`의 `CONDITION_MATRIX` 정의다(원 저장소 서버에서 전달받음, 이 서버에서 원문 미확인). 이 프로젝트는 600 W 조건만 쓴다.

| 실험 | A | B | C | D |
|---|---|---|---|---|
| T02 | LoRA bf16 | QLoRA 4-bit | (480 W) | (480 W) |
| T04 | SDXL, batch 4 | (480 W) | (360 W) | FLUX.1-dev, batch 2 |
| I01 | concurrency 1, saturated | concurrency 32 | concurrency 1, burst 8 req/10 s | (480 W) |
| I02 | SDXL batch 1 | batch 4 | burst 4 img/15 s | (480 W) |

- I01: prompt medium, max_tokens 128. 모델 접두사 `8B` Llama-3.1-8B-Instruct, `Q7B` Qwen2.5-7B-Instruct, `32B` Qwen2.5-32B-Instruct.
- 원 저장소의 `_policy-*` run(controller 실험)은 쓰지 않는다.

### 2.3 기록 변수와 phase

| column | 단위 | 수집 | 해상도 |
|---|---|---|---|
| `timestamp_ms` | ms (epoch) | 10 ms polling | 샘플 간격 중앙값 10.08–10.10 ms |
| `power_w` | W | NVML `POWER_INSTANT` | 약 0.5 s마다 갱신(중앙값 504 ms) |
| `utilization_pct`, `memory_utilization_pct` | % | NVML | 샘플링 구간 평균 |
| `temperature_c` | °C | NVML GPU sensor | 정수, 양자화 RMS 0.29 K |
| `phase` | – | 실험 단계 태그 | – |

모든 run은 warmup 180 s(600 W 행렬곱) → measure → 본 구간 약 1,200 s → cooldown 120 s 순서다. `measure`는 워크로드 준비(모델 로딩) 구간이며 본 부하가 아니다(조건별 중앙값 4–106 s, I01-32B-A r01만 4,949 s). cooldown 평균 power는 16–28 W다. 모든 통계는 `gpu_samples.csv`에서 계산한다.

### 2.4 Parsing 검증값 (사전 탐색, tuning 금지)
본 구간 평균, 3 run 평균. inventory 결과가 P ±10 W, T ±0.5 K 이상 다르면 parsing 오류로 멈춘다.

| 조건 | 평균 P [W] | 평균 T [°C] | 부하 주기 [s] / 주기당 ΔT [K] | split |
|---|---|---|---|---|
| T02-A / T02-B | 578 / 579 | 48.3 / 49.2 | 약함 / 2.5, 3 | train |
| T02-32B-A / A2 | 595 / 596 | 49.3 / 49.2 | 없음 | train |
| T04-A / I01-Q7B-A | 578 / 424 | 48.4 / 40.2 | 없음 | train |
| T04-D / I01-8B-A / I02-A | 598 / 432 / 588 | 49.7 / 40.7 / 48.6 | 없음 / 없음 / 4.5, 3.7 | validation |
| I01-Q7B-B / I01-8B-B / I02-B | 451 / 501 / 590 | 42.0 / 44.2 / 48.4 | 6.5, 7 / 4.0, 5 / 8.0, 5.7 | test interp |
| I01-32B-A / B | 507 / 509 | 43.2 / 43.7 | 계산하지 않음 | test interp |
| I01-Q7B-C / I01-8B-C / I02-C | 280 / 288 / 275 | 34.7 / 34.9 / 35.2 | 23.5, 10 / 24.0, 10.3 / 24.0, 19 | test shift |

### 2.5 출처와 검사 결과
- **T04-D**: 2026-03-31 측정분(오염 없음). 원 저장소의 T04-D는 이후 T06-A와 동시 실행된 run으로 덮어써졌으므로 쓰지 않는다.
- **I01-32B**: `inference*` 단계만 본 구간으로 쓰고 추론 시작 후 60 s를 평가에서 뺀다(r01은 긴 로딩 뒤라 첫 60 s 평균 power가 493 W, r02·r03은 503·505 W).
- **I01-32B-C**: 원 저장소에 1 s·10 s·60 s 집계 파일만 있고 10 ms 원샘플과 메모리 사용률이 없어 쓰지 않는다.
- **형식·오염 검사(1단계, 51 run)**: 결측 0, md5 중복 0, timestamp 중복 0, phase 순서 정상, warmup plateau 49.0–51.0 °C, cooldown 끝 25.2–28.0 °C. 제외 0. 0.5 s 넘는 샘플 공백은 I02 9 run에만 1–3곳(최대 4.5 s). 일부 T02 run은 warmup 동안 직전 반복의 GPU 메모리(13–65 GB)가 남아 있었으나 warmup power와 plateau는 정상이다. T02-32B-A2 r03은 다른 세션에서 측정했다.
- 처음 업로드의 8-GPU 로그는 공개 데이터셋(FigShare) 사본이라 삭제했다.

### 2.6 전처리 (`gputherm/data/`)
1. md5로 중복을 확인하고 `results/inventory.csv`를 만든다. 시간축은 run 첫 행 기준 t [s]다.
2. **0.1 s 입력 격자**: bin k = [0.1k, 0.1(k+1))의 샘플 평균. 빈 bin은 앞 값으로 채운다(인과성 유지). 채운 값은 모델 입력에만 쓴다.
3. **window**: t₀는 0.5 s 격자 경계, horizon은 [t₀, t₀ + 60).
   - 인과성: 관측기와 look-back은 t₀ 이전 정보만 쓴다. horizon 입력(P, u, mu)만 t₀ 이후 측정값(계획값 가정)이다.
   - 출발 상태: T_g(t₀) = [t₀ − 0.1, t₀) 원샘플 평균(6.6절에서 확정).
   - 출력: 0.5 s 구간 시간 평균 (1/0.5)∫T_g dt. 끝점 평균으로 근사하지 않는다(차이는 6.1절).
   - 기록 시작 전 이력이 필요하면 첫 샘플 값으로 채운다.
4. **평가 정답**: 원샘플에서 직접 만든 0.5 s 구간 평균(120개). 원샘플 25개 미만인 bin은 무효, 유효 bin이 108개 미만인 window는 제외한다. inventory, 학습, 평가는 같은 window 필터(`gputherm.data.windows.usable_windows`)를 쓴다.
5. 정규화 통계는 train으로만 fit한다(최종 재학습 때 train + validation).
6. power–온도 시간 지연 보정은 0이다.
7. **인과성 test**: t₀ 이후 원샘플을 바꿔도 출발 온도, 관측기 상태, look-back이 변하지 않아야 한다(정수 초, 반정수 초, 이력 부족 window). coverage 필터는 합성 공백으로 검사한다.

### 2.7 Wet-lab 기록

| 항목 | 내용 | 산출물 |
|---|---|---|
| 승인 | 측정자이자 소유자인 작성자(2026-10-11). 담당 교수 확인은 받으면 첨부 | `records/approval.md` |
| 설계 문서 | 실험계획서(2026-03): factor, 수준, 반복 3회, 측정 구간, 수집 사양 | appendix 인용 |
| factor | workload, 모델, 양자화, 동시 요청 수, batch, traffic 패턴(saturated, burst). cap 600 W 고정 | `conditions.csv` |
| 장비 | 2.3절, serial·driver·실내 온도는 미기록 | `records/instrument.md` |
| 안정성 점검 | 매 run의 warmup(600 W, 180 s)을 반복 control로 쓴다. 절대 교정을 대신하지 않는다 | `results/control.csv` |
| 실행 순서 | 같은 조건 r01–r03을 연속 실행, 무작위화 없음. 조건 순서 I02 → I01-8B → I01-32B → T04-A, I01-Q7B → T02-32B-A2 → T04-D → T02-A, T02-B, T02-32B-A | `records/run_order.csv` |
| 제외 규칙 | phase 순서 이상, 결측 1 % 초과, cooldown 평균 50 W 초과 또는 util 5 % 초과. plateau가 중앙값에서 2 K 이상 벗어나면 제외 후보. 해당 0 | `results/exclusions.csv` |

### 2.8 M10 근거
1. 직접 계획·수행한 wet-lab 측정(서면 계획, 반복 3회, 고정 측정 구간, 10 ms 원기록).
2. 코드로 재현되는 선별 절차(형식·오염 검사, 제외 규칙, phase 의미 정정, I01-32B 처리).
3. 설계한 dry-lab 실험: 두 번째 node를 측정하지 않을 때의 구조적·실용적 식별 가능성(6.4절, 개발용 시나리오 80, 잠긴 시나리오 20, 해상도 3단계).

---

## 3. 측정 불확도 (M17)

reference는 NVML이 기록한 die 온도다. 모든 오차는 이 기록에 대한 오차이며 sensor의 절대 정확도와 구분한다. u_c는 알려진 성분만으로 낸 추정이고, train·validation으로만 계산한다(`results/uncertainty.json`).

| 성분 | 정의 | 값 |
|---|---|---|
| u_res | 정수 양자화, 1/√12 | 0.289 K |
| u_repeat | 조건별 반복 3회의 본 구간 평균 T 표본 SD s_i의 pooled 값 √(Σs_i²/9) | 0.153 K (s_i 0.015–0.247 K) |
| u_control | warmup plateau의 조건 내 pooled SD (진단용) | 0.350 K |
| u_timing | power 0.5 s 지연 시 validation RMSE 변화 | 5단계에서 계산 |
| u_cal | sensor 절대 정확도 | 미상, u_c에서 제외 |

- **u_c = √(u_res² + u_repeat²) = 0.327 K**, 2u_c = 0.653 K. u_control은 u_repeat와 같은 drift를 포함할 수 있어 합성하지 않는다(JCGM 100:2008 5.2절). 대신 조건 평균을 뺀 본 구간 온도를 조건 평균을 뺀 plateau에 회귀해 u_repeat²를 분해한다: 설명 몫 19 %(r = 0.44). 인과적 해석은 하지 않는다.
- u_repeat는 본 구간 평균의 반복성이며, 시간별 온도나 peak의 재현성을 뜻하지 않는다.
- **용도**: 절대 오차 기준 ε_Q, ε_pk의 하한(9.3절). 짝비교 기준에는 쓰지 않는다(같은 window의 차이에서는 run 간 변동이 상쇄된다).

---

## 4. Split (M12, M13)

### 4.1 원칙
조건 단위로 나눈 뒤 window를 자른다. 같은 조건의 run은 한 split에만 속한다. run ID는 `results/splits.json`에 고정한다.

### 4.2 Split

| split | 조건 | run | 평가 window (10 s) | 학습 window (5 s) | 본 구간 P [W] | 본 구간 T [°C] |
|---|---|---|---|---|---|---|
| train | T02-A, T02-B, T02-32B-A, T02-32B-A2, T04-A, I01-Q7B-A | 18 | 2,090 | 4,169 | 11–664 | 26–52 |
| validation | T04-D, I01-8B-A, I02-A | 9 | 1,034 | – | 82–673 | 30–51 |
| test interp | I01-Q7B-B, I01-8B-B, I02-B, **I01-32B-A, I01-32B-B** | 15 | 1,689 | – | (83–689) | (30–51) |
| test shift | I01-Q7B-C, I01-8B-C, I02-C | 9 | 1,030 | – | (80–674) | (28–49) |

window 수는 1단계 결과(`tables/T1_ranges.csv`)다. 괄호의 test 범위는 사전 탐색 값이며 9단계 coverage에서 다시 계산한다. 굵은 조건이 모델 결과 미열람 subset이다.

- **validation 구성**: 학습, LLM 추론, 이미지 추론에서 하나씩. train에 없는 추론 모델과 workload 종류를 포함한다.
- **I01-32B를 test에 둔 이유**: 모델 결과를 본 적 없는 유일한 run이며, power·온도가 train 범위 안이고 forcing 형태가 train의 saturated 추론과 같아 interpolation 정의에 맞는다.
- **열람 이력**: I01-32B를 뺀 test 18 run은 사전 탐색에서 선형 RC 결과를 보았다. 잠금은 이후의 누수를 막지만 이전 열람을 없애지 않는다. 따라서 평가를 **사전 열람 이력이 있는 평가**(I01-Q7B-B, I01-8B-B, I02-B, shift 3개)와 **모델 결과 미열람 평가**(I01-32B-A/B)로 구분해 보고한다. shift test에는 미열람 run이 없다.
- **결론 규칙 (사전 고정)**: interp 전체의 통과가 미열람 조건의 성공을 대신하지 않는다. I01-32B-A/B 각각의 판정(RC 대비 부호, ε_Q, ε_pk)을 따로 적는다. 두 조건 모두 RC보다 나쁘면 "모델 결과를 본 적 없는 workload에서는 개선이 확인되지 않았다", 하나만 나으면 "두 미열람 조건에 걸친 일관된 개선은 확인되지 않았다"고 쓴다.
- **최종 재학습 (사전 고정)**: train + validation 27 run으로 ① 정규화 통계 재계산, ② RC를 α = 1에서 재추정하고 5.1절 규칙으로 α_ref 재결정, ③ 그 RC를 고정하고 α_ref에 맞는 closure 재학습(epoch는 validation best로 고정), ④ DLinear-X를 고른 λ, L로 재풀이. 같은 RC checkpoint를 baseline과 RC-NN에 쓴다.
- 가이드의 wet-lab 계획 기준(독립 run 12 이상, locked test unit 3 이상)은 수량으로 충족한다. 미열람 locked test 요건은 충족하지 않는다.

### 4.3 Interpolation과 shift (M13)
- **Interpolation**: 평균 power 451–590 W로 train 범위 안의 처음 보는 조건. B 조건은 4–8 s 주기의 작은 부하 변동(주기당 ΔT 5–7 K)이 있다.
- **Shift (burst)**: train의 주기적 power swing은 최대 약 45 W(2.5–4 s 주기)이고 큰 변화는 warmup·cooldown step뿐이다. burst는 320–480 W swing이 약 24 s 주기로 반복된다. swing 진폭은 train 최대의 7–11배, 주기당 온도 swing은 10–19 K(train 최대 약 4 K), 주기는 τ_slow보다 짧다(Π₁ ≈ 0.6). test 쪽 강도는 사전 탐색 값이다.

---

## 5. 물리 모델 (M5–M9)

### 5.1 지배 방정식 (M5)
출발점은 PI-DLinear(AlShaikh Saleh et al., KDD 2026, 식 7–8)의 2-node lumped RC다. 각 node의 에너지 보존식이며, 부호와 항은 참고 구현(`github_code/PI-DLinear`의 `rc_physics_loss_from_observed_temps`)과 일치함을 확인했다.

```
일반형: C_g dT_g/dt = α P − (T_g − T_a)/R_ga − (T_g − T_2)/R_g2
        C_2 dT_2/dt = (1 − α) P − (T_2 − T_a)/R_2a + (T_g − T_2)/R_g2
```

**구조적 식별 가능성.** 두 번째 node를 측정하지 않으면 die 온도 응답은 전달함수 (b₁s + b₀)/(s² + a₁s + a₀)의 계수 4개로 정해진다. 일반형의 parameter는 6개이고 계수 map의 Jacobian rank는 4이므로, 잡음이 없어도 2개 방향을 정할 수 없다(`gputherm/verification/structural.py`).

따라서 R_ga → ∞인 **사다리형**을 쓴다. die는 두 번째 node를 거쳐서만 주변으로 열을 내보내고, 보드 power 중 αP는 die로, (1 − α)P는 두 번째 node로 들어간다.

```
RC: C_g dT_g/dt = α P − (T_g − T_h)/R_gh
    C_h dT_h/dt = (1 − α) P + (T_g − T_h)/R_gh − (T_h − T_a)/R_ha
```

전달함수 계수: b₁ = α/C_g, b₀ = (αR_gh + R_ha)/(C_g C_h R_gh R_ha), a₁ = 1/(C_h R_ha) + 1/(C_h R_gh) + 1/(C_g R_gh), a₀ = 1/(C_g C_h R_gh R_ha). 정상상태 T_g = T_a + P(R_ha + αR_gh).

**두 번째 node의 해석.** T_h는 die보다 느리게 반응하는 숨은 유효 상태다. 느린 성분으로 계산한 열용량(약 5 kJ/K, 5.5절)은 방열판 하나보다 커서, fan 속도 변화나 섀시 공기 등이 섞였을 수 있다. 관측기의 T_h가 실제 방열판 온도인지는 검증할 수 없으므로, 두 번째 node에 대한 판단은 die 온도 예측 성능과 추가 상태의 예측 효과(6.4절 규칙 2)로 한정한다.

**α와 α_ref.**
- α = 1에서는 parameter 4개가 계수에서 유일하게 정해진다. α < 1에서는 역산식
  ```
  C_g = α/b₁,   x² − (a₁ − αb₀/b₁)x + (1 − α)a₀ = 0 (x = 1/(C_g R_gh)),
  R_gh = 1/(C_g x),   R_ha = b₀/a₀ − αR_gh,   C_h = x/(a₀R_ha)
  ```
  가 양의 해를 두 개 가질 수 있다(예: α = 0.75에서 (C_g, C_h, R_gh, R_ha) = (1000, 100, 0.01, 0.03)과 (1000, 87.72, 0.012, 0.0285)). 상수 α는 다른 parameter와 함께 정할 수 없다(rank 4/5).
- **규칙 (사전 고정)**:
  1. RC는 α = 1에서 추정한다.
  2. 계수를 위 역산식의 큰 근(α = 1의 해에서 연속으로 이어지는 가지)으로 α_ref = 0.75에 옮긴다. 판별식이 양수이고 parameter가 모두 양수여야 한다.
  3. 관측기를 포함한 파이프라인에서 두 RC의 60 s 예측을 비교한다. run별 window RMS 차이 평균의 validation 9 run 최댓값이 0.02 K 미만이면 α_ref = 0.75를 쓴다(`results/rc_alpha_ref.json`).
  4. 2 또는 3을 만족하지 않으면 α_ref = 1을 쓴다.
- α_ref = 0.75는 후보 α(u, mu)가 정상상태 온도를 양방향으로 보정할 수 있게 한다. 최종 재학습에서도 같은 규칙을 적용하며(동등성은 27 run 최댓값), closure 식은 최종 checkpoint의 α_ref를 따른다.

**PI-DLinear와의 차이.** 논문은 memory 온도를 측정했고(MIT Supercloud V100, 1분 간격), RC parameter를 RLS로 고정했으며(시간상수 수 시간), 온도로 power를 예측했다. 여기서는 두 번째 node가 숨은 상태라 식별 가능한 사다리형을 쓰고, 10 ms 데이터에서 시간상수 수십 초의 parameter를 train으로 추정하며, power로 온도를 예측한다.

### 5.2 기호와 단위 (M6)

| 기호 | 의미 | 단위 | 비고 |
|---|---|---|---|
| T_g | die 온도 | °C | 측정 |
| T_h | 두 번째 node(유효 열 저장부) 온도 | °C | 숨은 상태 |
| T_a | 유효 주변 온도 | °C | 전역 상수로 추정. cooldown 점근값 26.6–27.7 °C |
| P | 보드 전체 power | W | 측정, 전량 열로 가정 |
| α | die로 들어가는 power 비율 | – | RC에서 α_ref, 후보 α에서 0.5–1 |
| C_g, C_h | 열용량 | J/K | > 0 |
| R_gh, R_ha | die–두 번째 node, 두 번째 node–주변 열저항 | K/W | > 0 |
| u, mu | GPU 사용률, 메모리 사용률 | % | closure 입력 |

차원: C dT/dt [W] = P [W] − ΔT/R [W].

### 5.3 조건과 closure (M7)
- **초기 상태**: T_g(t₀)는 측정값(2.6절). T_h는 관측기로 정한다. run 시작(T_h = T_g(0))부터 t₀까지 측정 T_g와 P로
  ```
  C_h dT_h/dt = (1 − α)P − (T_h − T_a)/R_ha + (T_g,측정 − T_h)/R_gh
  ```
  를 적분한다. 감쇠율 (1/R_ha + 1/R_gh)/C_h > 0이므로 초기값 오차는 지수적으로 사라진다. 과거 측정만 쓰므로 운영 중에도 계산할 수 있다. 관측기 시간상수가 본 구간 전 최소 이력 186 s의 1/5(37 s)보다 긴지 보고한다.
- **경계 조건**: T_a는 run 내·run 간 상수. fan 효과는 R_ha에 흡수된다.
- **closure 후보** (validation으로 하나 선택. 두 후보 모두 P 전체가 모델로 들어가 보드 power 기준 에너지 수지가 성립한다):
  - 후보 R_ha(T_g): 두 번째 node–주변 열저항이 die 온도의 함수. GPU 온도를 따르는 fan 제어를 나타낸다.
  - 후보 α(u, mu): die로 바로 가는 power 비율이 workload의 함수. die 밖 발열(memory, VRM)의 비중 변화를 나타낸다. 고정한 RC 위의 보정 함수로만 해석한다.
- **준정적 1-node** (6.4절이 2-node를 기각할 때): 사다리형의 C_g → 0 극한.
  ```
  C_h dT_h/dt = P − (T_h − T_a)/R_ha,    T_g = T_h + αP R_gh
  ```
  cooldown에서 die 온도 변화의 약 77 %가 1 s 안에 일어나므로(5.5절), die 열용량만 가진 1-node 대신 이 형태를 쓴다. αR_gh가 곱으로만 나타나므로 α_ref 고정은 정확한 재parameter화다. 출발 상태는 T_h(t₀) = T_g(t₀) − αP(t₀⁻)R_gh다. 두 closure를 그대로 쓸 수 있다. 이 구조가 선택되면 6.1–6.3, 6.5, 6.7 검증을 이 구조로 다시 통과해야 한다.
- **입력**: power는 입력 격자 한 칸(0.1 s) 안에서 일정하다. horizon 입력은 계획값이다.

### 5.4 가정과 생략한 물리 (M8)
- **가정**: die 온도는 단일 값(lumped). 열저항 선형(후보 R_ha가 완화). fan 효과는 R에 흡수. horizon 입력은 정확히 알려져 있다. T_a는 전역 상수다.
- **T_a 가정의 영향**: run별 점근값 차이는 약 ±0.55 K다. T_a는 두 번째 node를 거쳐 작용하므로 느린 모드(τ_slow ≈ 41 s) 하나로 근사하면, 측정 T_g에서 출발한 예측에 남는 비율은 horizon 끝 1 − e^(−60/41) ≈ 0.77(0.42 K), horizon 평균 1 − (41/60)(1 − e^(−60/41)) ≈ 0.47(0.26 K), RMS ≈ 0.52(0.29 K)다. 단일 모드 크기 추정이며 엄밀한 상한이 아니다. 9.6절 민감도로 확인한다.
- **생략**: fan 동특성, die 내부 공간 분포, VRM·memory 온도, 실내 온도 변화, 누설 전력의 온도 의존(사전 탐색에서 +0.1–4 W/K).
- **유효 범위**: 1.4절 범위 D, 이 GPU와 chassis, 1 s보다 긴 시간 척도.

### 5.5 척도 (M9)

| 척도 | 값 | 의미 |
|---|---|---|
| τ_fast | 0.39 s (0.28–0.68) | die. τ_fast/0.5 s ≈ 0.8로 power 해상도 근처 |
| τ_slow | 41 s (33–49) | 두 번째 node와 주변 |
| η_fast (fast 성분 비중) | 0.77 (0.61–0.86) | power 변화에 대한 die 온도 응답의 77 %가 1 s 안에 일어난다 |
| R_ss | (50.0 − 27.1) K / 600 W ≈ 0.038 K/W | warmup plateau와 점근값 |
| Π₁ = burst 주기 / τ_slow | 24/41 ≈ 0.6 | burst 한 주기 안에 느린 성분이 평형에 이르지 못한다 |
| Π₂ = horizon / τ_slow | 60/41 ≈ 1.5 | 60 s 예측이 느린 모드를 1.5배 덮는다 |
| R_ss × ΔP | 0.038 × 320 ≈ 12 K | burst 온도 swing(10–19 K)과 같은 크기 |

근거: train·validation cooldown(높은 power에서 내려온 26 run)을 구간 평균 지수 모델 A(τ/Δ)(e^(−t/τ) − e^(−(t+Δ)/τ))로 fit했다(`gputherm/verification/cooldown_modes.py`). 지수 1개 RMS 0.68 K, 지수 2개 0.32 K(양자화 0.29 K 수준)로 두 시간 척도가 존재한다. 정수 양자화는 시간상수 추정을 편향시킬 수 있으므로(6.6절) 위 값은 근사다. 5단계 RC 추정 뒤 다시 계산한다.

### 5.6 수치 해법과 parameter 추정 (`gputherm/physics/`, `gputherm/estimation/rc_fit.py`)
- **정확 이산화 (RC, 후보 α)**: 입력 v = [P, T_a, αP]에 대해 A가 상수이므로, 입력 격자 Δ마다
  ```
  x_{k+1} = A_d x_k + B_d v_k,   ∫_step x dt = G x_k + H v_k
  ```
  이다. A_d, B_d, G, H는 Van Loan 블록 행렬 하나의 지수로 계산한다(작은 차의 상쇄가 없어 값과 미분이 정확하다). window 예측은 impulse 응답과의 합성곱으로 계산한다(순차 계산과 동일, 6.7절). 입력이 구간 안에서 일정하다는 가정 아래 적분 오차는 0이며, 입력 표현 오차는 6.2절에서 정량화한다.
- **후보 R_ha(T_g)**: RK4, dt = 0.1 s. 입력 변화 시점은 step 경계와 일치시킨다.
- **추정**: 대상은 C_g, C_h, R_gh, R_ha, T_a(α = 1, log 재parameter화. 준정적 1-node는 C_h, R_ha, R_gh, T_a).
  - 목표함수: train 학습 window(5 s stride)에서 관측기 상태로 출발한 60 s 예측의 MSE. 유효 정답 bin을 먼저 고른 뒤 모든 bin에 같은 가중치를 준다. RC-NN도 같은 목표를 쓴다. 평가 집계(window별 RMSE의 run 평균, 9.1절)는 별개다.
  - 최적화: L-BFGS(strong Wolfe). 초기값은 τ_fast 0.2–10 s, τ_slow 15–100 s, R_ss 0.03–0.05 K/W, R_gh 비율 0.2–0.9, T_a 25–29 °C 범위의 Latin hypercube 20개 이상.
  - 초기값별로 손실, 최종 gradient 크기, 반복 수, 평가 수, 종료 사유를 기록하고, 유한한 결과와 수렴한 결과를 구분한다. 실패한 초기값은 격리하고 유한한 결과만 비교한다.
  - 결과와 시간상수는 `results/rc_params.json`에 저장한다.

---

## 6. 검증과 dry-lab 실험 (M10, M15–M18)

대표 parameter: τ_fast 0.4 s, τ_slow 41 s, R_ss 0.038 K/W, R_gh = 0.77 R_ss(C_g 13.7 J/K, C_h 4,660 J/K, R_gh 0.0294 K/W, R_ha 0.0088 K/W, α = 1, T_a 27.1 °C). 6.1–6.3, 6.5는 RC 추정 뒤 추정값으로 다시 실행한다. 모든 검증은 float64이며 실패하면 종료 코드 1로 멈춘다. 결과는 `results/verification/`에 저장한다.

### 6.1 독립 구현 대조 (M15)
- 일정 power step(25 → 600 W)의 해석해와 정확 이산화(numpy, torch), RK4(dt 0.001 s), `torchdiffeq` dopri5(rtol 1e-10)를 상태와 0.5 s 구간 평균(해석적 적분) 모두에서 비교한다. 기준 1e-6 K.
- 측정 power와 시간 가변 α(1 − 0.3u/100)에서 정확 이산화와 RK4(dt 0.01 s)를 비교한다. 기준 1e-4 K.

### 6.2 수렴과 입력 격자 (M16)
- RK4(후보 R_ha의 대표 감소형): dt = 0.1/k (k = 1, 2, 4, 8), 기준해 k = 16. 기준: 관측 order 3.5–4.5, dt = 0.1 s 오차 0.01 K 이하.
- 입력 격자: validation run에서 원샘플을 t₀ 기준 Δ = 0.5, 0.25, 0.1 s 구간으로 평균한 입력으로 60 s를 예측한다. 기준: Δ = 0.25 s와 0.1 s의 horizon RMS 차이 0.05 K 이하(넘으면 0.05 s를 추가해 기본 격자를 다시 정한다).

### 6.3 에너지 수지 (진단)
∫P dt = C_gΔT_g + C_hΔT_h + ∫(T_h − T_a)/R_ha dt를 확인한다. 정확 이산화는 상태 적분 G, H로, RK4는 누적 에너지를 상태에 추가해 계산한다(Runge–Kutta는 선형 불변량을 보존한다). 기준: 상대 잔차 1e-6. 구조상 성립하므로 A, B와 열 흐름 항의 조립을 확인하는 진단이다.

### 6.4 식별 가능성 dry-lab 실험 (gate)
RC 추정 전에 실행하며, 주 모델 구조(사다리형 2-node 또는 준정적 1-node)의 후보를 정한다.

**(A) 구조적 식별 가능성 (판정 없음).** 일반형 4/6, 사다리형 4/4의 Jacobian rank와, 같은 die 온도 응답을 내는 일반형 parameter 두 조합을 보인다. 개별 parameter 복원 오차는 여기서만 보고한다.

**(B) 실용적 식별 가능성 (설계 사전 고정).**
- 판정량은 전달함수 계수로 정의한 응답 특성이다: 극점 λ에 대해 τ = −1/λ(τ_fast, τ_slow), R_ss = b₀/a₀, 영점 시간상수 τ_z = b₁/b₀. 두 번째 모드는 τ_fast와 τ_z로 나타난다. fast 성분 비중은 η_fast = (τ_z − τ_fast)/(τ_slow − τ_fast)다.
- 시나리오: 사다리형 parameter를 로그 균등 Latin hypercube로 뽑아 τ_slow 25–100 s, R_ss 0.03–0.05 K/W, τ_fast 0.2–10 s인 것만 남긴다. τ_fast/0.5 s(0.4–20, 로그 구간 5개) × η_fast(0.02–0.1, 0.1–0.9) 칸마다 개발용 8개, 잠긴 2개(합계 80, 20). 데이터 추정값(τ_fast/0.5 s ≈ 0.8, η_fast ≈ 0.77)은 최저 해상도 구간과 큰 η_fast 구간에 속한다.
- 기록 생성: train run의 측정 power, 정수 반올림, 0.5 s power hold. factor는 시나리오, 양자화 유무, hold 유무, fit 구간의 step event 포함 여부다. 시뮬레이션 dt는 0.25, 0.1, 0.05 s.
- fit: 5.6절 절차, 초기값 20개 이상. 최솟값의 1 % 안의 fit을 모두 남긴다.
- 평가: fit에 쓰지 않은 forcing(validation run power)에서 τ_fast·τ_z·η_fast 복원 오차, 참값 대비 60 s 예측 RMS, 남긴 fit들 사이의 60 s 예측 변동(최댓값).

**결정 규칙 (dt = 0.1 s, 양자화·hold 포함 조건).**
1. η_fast 구간마다, 개발용 시나리오의 τ_fast/0.5 s 구간별 중앙값이 (i) τ_fast, τ_z 복원 오차 30 % 이하, (ii) 참값 대비 예측 RMS 0.1 K 미만, (iii) 예측 변동 0.1 K 미만을 그 구간과 상위 모든 구간에서 만족하는 가장 낮은 구간의 하한을 r*로 정한다. 잠긴 시나리오 중 r* 이상인 것으로 확인한다. 두 η_fast 구간 모두에서 r*가 없거나 확인에 실패하면 준정적 1-node로 정한다. r*는 이 forcing, 잡음, 모드 크기에 조건부인 경험적 기준이다.
2. r*가 있으면 5단계에서 준정적 1-node와 2-node를 train으로 fit하고, 2-node의 추정 τ_fast/0.5 s가 해당 r* 이상이고 validation RMSE가 0.05 K 이상 좋을 때만 2-node를 쓴다.
3. 준정적 1-node로 정해지면 RC와 RC-NN을 그 구조로 바꾸고 검증을 다시 한다(5.3절).
4. 결과를 `results/LOG.md`와 `ACCEPTANCE.md`에 기록한다.

### 6.5 정상상태 거동 (M18)
일정 power 25–600 W에서 정상상태 T_g가 단조 증가·유한하고, P = 0이면 T_a로 돌아가며, T_a + P(R_ha + αR_gh)와 일치하는지 확인한다.

### 6.6 시간 정렬 검증
알려진 RC(대표 parameter)에 train run(T02-A/r01)의 측정 power(0.01 s 앞 값 유지)를 넣어 참 궤적을 만들고, 같은 참 궤적에서 관측 기록만 단계적으로 바꿔 실제와 같은 전처리·fit·평가에 넣는다. 단계별 영향은 서로 독립 성분이 아니므로 더하지 않는다.

| 단계 | 기록과 모델 입력 | 참 parameter RMS (출발 상태 last_0p1 / last_0p5 / last_sample) | fit parameter RMS | fit τ_fast / τ_slow (참 0.40 / 41.0 s) |
|---|---|---|---|---|
| 1 | 원샘플 시각의 참 온도, 참 입력(0.01 s) | **0.0017** / 0.011 / 0.0002 K | – | – |
| 2 | 단계 1 기록, 0.1 s bin 입력 | 0.011 / 0.017 / 0.011 K | 0.009 K | 0.41 / 41.2 s |
| 3 | + 정수 반올림 | 0.019 / 0.021 / 0.020 K | 0.064 K | 0.36 / 33.4 s |
| 4 | + 샘플 공백 3곳 | 0.028 / 0.030 / 0.029 K | 0.070 K | 0.36 / 33.4 s |

- 통과 기준(단계 1, 참 parameter, 0.02 K 이하)을 충족한다.
- 출발 상태는 단계 4에서 세 정의의 차이가 0.01 K 미만이므로 사전 규칙에 따라 last_0p1이다.
- 정수 양자화는 시간상수 추정을 편향시킨다(run 하나의 fit에서 τ_slow −18 %, T_a +0.2 K, 모든 초기값 수렴). 측정 해상도의 한계이며, gate와 5.5절 척도 해석에 반영한다.
- α_ref 동등성(단계 4, 관측기 포함): α = 1과 α_ref = 0.75 RC의 예측 차이 평균 0.0014 K, 최대 0.0024 K.

### 6.7 Torch 구현의 값과 미분
- 값: torch 정확해·관측기와 numpy 구현, 합성곱과 순차 계산(시간 가변 α, batch)을 비교한다. 관측기는 긴 run, 빠른 감쇠(−λΔ = 800, 25, 20.1, 15), 짧은 입력을 포함한다. 기준 1e-9 K.
- 미분: 관측기 → 60 s 예측 → 손실의 C_g, C_h, R_gh, R_ha, T_a, α에 대한 자동 미분을 5점 중심 유한차분과 비교한다. 기준 상대 오차 1e-5.
- fit 구현: batch와 개별 실행의 일치, 무효 정답의 배제, float32 거부, 실패 초기값 격리, 종료 사유 기록.

### 6.8 검증 결과 요약 (대표 parameter)

| 검증 | 결과 | 기준 |
|---|---|---|
| 6.1 해석해 대조 (상태 / 구간 평균) | ≤ 1.3e-8 K / ≤ 7.5e-11 K | 1e-6 K |
| 6.1 시간 가변 α | 1.8e-8 K | 1e-4 K |
| 6.1 끝점 평균과 구간 평균의 차이 (참고) | 첫 0.5 s 1.16 K, 첫 60 s RMS 0.11 K | – |
| 6.2 RK4 order / dt 0.1 s 오차 | 4.15, 4.08, 4.12 / 2.7e-4 K | 3.5–4.5 / 0.01 K |
| 6.2 입력 격자 0.25 s 대 0.1 s (0.5 s 대 0.1 s) | 0.026 K (0.27 K) | 0.05 K |
| 6.3 에너지 수지 | ≤ 1.0e-14 | 1e-6 |
| 6.5 정상상태 (600 W) | 50.0 °C, 해석 관계와 일치 | – |
| 6.6 시간 정렬 | 0.0017 K | 0.02 K |
| 6.7 값 / 미분 | ≤ 7.2e-13 K / ≤ 4.2e-9 | 1e-9 K / 1e-5 |

---

## 7. 예측 과제 (M19)

| 항목 | 정의 | 모양 (batch B) |
|---|---|---|
| 출력 | horizon 60 s의 T_g, 0.5 s 구간 평균 | (B, 120) |
| 미래 입력 | [P, u, mu]. RC 계열 0.1 s, DLinear-X 1 s | (B, 600, 3) / (B, 60, 3) |
| look-back (DLinear-X) | [T_g, P, u, mu], 1 s, L ∈ {60, 120, 180} s | (B, L, 4) |

- RC 계열은 look-back 대신 관측기 상태와 출발 온도를 쓴다.
- SM clock은 온도 정보를 담으므로(cap 구간에서 −2 – −5 MHz/K) 입력에서 뺀다.
- L ≤ 180 s: 본 구간 전 이력이 최소 186 s이므로 모든 window가 실제 측정 이력만 쓴다(L = 300, 600 s이면 각각 8 %, 33 %의 window가 기록 시작 전 값을 필요로 한다). 180 s는 τ_slow의 4.4배다.
- DLinear-X의 1 s bin은 window마다 t₀ 기준으로 만든다: 과거 [t₀ − j − 1, t₀ − j), 미래 [t₀ + j, t₀ + j + 1).
- window: 학습 5 s stride(train), 평가 10 s stride. 모든 모델과 L이 같은 window 집합을 쓴다. warmup·cooldown 구간은 보조 지표로 따로 보고한다.

---

## 8. 모델 (M14, M19–M22, M27)

### 8.1 모델

| 이름 | 역할 | 정의 | 학습 parameter |
|---|---|---|---|
| Persist | 기준 | 마지막 측정 T_g 유지 | 없음 |
| RC | baseline, closure 제거 ablation | 사다리형 2-node(또는 준정적 1-node), parameter 상수 | 5개(준정적 1-node 4개) |
| RC-NN | 제안 | RC + closure(5.3절의 두 후보 중 validation으로 하나) | closure만, RC는 고정 |
| DLinear-X | 선형 data-only comparator | DLinear(Zeng et al. 2023)에 알려진 미래 입력 항 추가 | 선형 가중치 |

- RC와 RC-NN은 같은 물리 구조와 같은 RC parameter를 쓰므로, 짝비교는 closure의 추가 효과를 보인다.
- RC-NN과 DLinear-X는 물리 구조, 비선형성, 모델 형태가 모두 다르다. 이 비교의 결론은 "선형 data-only comparator 대비"로 한정한다.
- 논문의 throttle loss는 쓰지 않는다(throttling 없음).

### 8.2 Closure
**후보 α(u, mu)**: α_k = α_ref + 0.25·tanh(MLP([u_k, mu_k])) ∈ (0.5, 1), MLP 2-16-16-1(tanh), 337개. 출력층을 0으로 초기화해 RC에서 출발한다. α_ref = 1이면 α_k = 1 − 0.5·sigmoid(MLP)(출력 bias −6). 정상상태 보정 폭은 ±0.25·P·R_gh다.

**후보 R_ha(T_g)**: 증가형 R_0(1 + Σ_{j=1}^{8} softplus(w_j)σ((T − c_j)/s)) 또는 감소형 R_0 / (1 + Σ…). c_j는 25–55 °C 등간격, s = 3 K. 학습 parameter는 R_0와 w_j 9개다. R_0는 RC의 R_ha에서, w_j는 −10에서 출발한다(초기 보정 1.0004배 이하). 방향은 두 형태의 train loss로 정한다. 증가형은 P·dR_ha/dT ≥ 1이면 정상상태가 유일하지 않을 수 있으므로 9.2절에서 판정한다.

**학습**: RC parameter를 고정하고 closure만 학습한다. 목표함수는 5.6절과 같다. 관측기는 run마다 한 번 계산한다. 초기화 직후 RC와의 예측 차이를 기록한다.

**선택 (사전 고정)**: validation run별 RMSE 평균이 낮은 후보. 차이가 0.05 K 미만이면 parameter가 적은 R_ha(T_g). 두 후보의 결과를 `results/val/closure_selection.json`에 기록하고, 고른 closure의 모양을 학습 범위와 함께 보고한다(F8).

### 8.3 DLinear-X
- look-back 각 채널을 moving average(kernel 25 s, 양 끝 replicate padding)로 trend와 나머지로 나누고, 미래 입력과 함께 선형 결합한다. parameter 수는 (8L + 181) × 120(L = 180 s에서 1.9×10⁵).
- **학습: ridge 닫힌 해.** 특징이 많고(최대 1,621) window가 겹치며 시차 간 상관이 높아, 무제약 최소제곱은 불안정하고 경사하강 결과는 학습 설정에 좌우된다. 목적함수 (1/n)‖Y − XW‖² + λ‖W‖²의 유일해를 thin SVD로 구한다(W = V diag(s/(s² + nλ)) UᵀY). 특징과 출력은 train 통계로 표준화하고 절편은 벌점에서 뺀다.
- **선택 (사전 고정)**: λ(10⁻³–10⁴ 로그 격자 15개) × L(60, 120, 180 s)을 validation RMSE로 고른다. λ가 격자 끝이면 2 decade씩 넓힌다(10⁻⁶, 10⁷까지). 최솟값에서 0.05 K 안이면 짧은 L, 큰 λ 순으로 고른다.
- **보조 확인 (부록)**: 원 DLinear 방식(Adam, lr {1e-4, 1e-3}, batch 32, 최대 200 epoch, patience 10, seed 0–4). seed 평균 validation RMSE가 닫힌 해보다 0.05 K 이상 좋으면 주 방법을 Adam으로 바꾸고 test는 seed 평균 ± SD로 보고한다.
- 비선형 data-only 모델(GRU, TCN 등)과의 비교는 범위 밖이다.

### 8.4 학습 설정 (M20)

| 항목 | DLinear-X | RC-NN |
|---|---|---|
| loss | 평균 제곱 오차 + λ‖W‖² | 유효 bin MSE (5.6절) |
| 풀이 | 닫힌 해(thin SVD) | Adam, closure만 |
| batch | 전체 | {64, 256} |
| 학습률 | – | {3e-4, 1e-3, 3e-3} |
| 정규화 | λ | weight decay 1e-4 |
| stopping | – | validation RMSE, patience 20, 최대 300 epoch |

### 8.5 선택과 반복성 (M21, M22)
- RC-NN은 seed 0–4로 학습하고 평균 ± SD를 보고한다. 판정은 seed 평균으로 하며, 조건별로 RC 대비 부호가 같은 seed 수를 T3에 싣는다. 통과한 조건에서 그 수가 3 미만이면 결론에 적는다.
- RC는 다중 초기값의 최적해, 수렴 여부, 해의 분포를 보고한다. DLinear-X는 선택한 λ, L과 validation 곡선을 보고한다.
- 학습 곡선, gradient norm, 실패한 학습을 기록한다(A1).

### 8.6 비용 (M27)
- **판정**: 예측 1회 latency(현재 관측기 상태에서 60 s 예측 하나, batch 1, CPU 1 thread, warm-up 3회 후 100회의 중앙값). 관측기는 측정마다 한 step씩 갱신되므로 운영 부담은 예측 1회로 대표된다.
- **보조**: 관측기 1 step, run 전체 처리 시간, 학습 시간, peak 메모리, parameter 수. GPU 시간은 공유 노드라 보조로만 보고한다.

---

## 9. 평가 (M23–M26)

### 9.1 지표와 짝비교
모든 모델은 같은 window, mask, metric code를 쓴다(M23). 단위는 K다.

| 지표 | 정의 |
|---|---|
| RMSE (주 QoI) | window별 horizon RMSE의 run 평균 |
| peak 오차 (주 QoI) | window별 \|max T̂ − max T\|의 run 평균 |
| swing 오차 | window별 \|(max − min)T̂ − (max − min)T\|의 run 평균 |
| e∞, bias | run별 최대 절대 오차, 평균 (예측 − 측정) |
| skill | run별 1 − RMSE_모델 / RMSE_Persist |

- **판정 단위는 조건**이다(3 run 평균). 조건 수가 적어 부호 규칙은 유의성 검정이 아니라 사전에 정한 판정 규칙이다(우연히 5개 중 4개 이상 같은 방향일 확률 0.19, 3개 중 3개 0.125).
- 조건별 분해 표(T3), lead time별 RMSE 곡선(F5)을 보고한다. 조건 단위 cluster bootstrap과 run 단위 bootstrap CI는 부록에 둔다.
- interp는 전체와 모델 결과 미열람 subset을 따로 보고한다.

### 9.2 Physics check (M25)
고른 closure에 해당하는 항목만 판정한다.

| check | 대상 | 방법 | 기준 |
|---|---|---|---|
| 열 폭주 안정성 | R_ha(T_g) | P 0–690 W × T 20–60 °C에서 P·dR_ha/dT. 정상상태 T_g = T_a + P(R_ha(T_g) + αR_gh)의 20–80 °C 해 | 최댓값 < 1, 모든 P에서 해 1개 |
| closure 범위 | R_ha(T_g) | 20–60 °C에서 양수·유한 | 위반 0 |
| idle 복귀 | 두 후보 | held-out run의 측정 cooldown 120 s를 재생, 끝 10 s 평균 비교 | RC-NN 끝 온도 오차가 RC보다 0.5 K 넘게 크지 않음 |
| α 외삽 (보고) | α(u, mu) | train (u, mu) 볼록 껍질 밖 시점과 그때 α의 경계 근접 비율 | 판정 없음 |
| 정상상태 단조성 (확인) | 두 후보 | 25–600 W | dT_ss/dP > 0 |
| 에너지 수지 (진단) | 두 후보 | 6.3절 | 1e-6 |

idle 복귀를 RC 대비로 판정하는 이유는 전역 T_a가 run별 주변 온도 차이를 설명하지 못하기 때문이다(5.4절). 절대 오차와 T_a 아래 하강 여부는 보고만 한다. DLinear-X에도 단조성과 idle 복귀를 계산해 보고한다.

### 9.3 Acceptance table (`ACCEPTANCE.md`, M4)
validation 결과와 u_c를 본 뒤, test 평가 전에 수치와 날짜를 고정한다.

**기준의 근거**
- ε = 1.0 K는 결정 허용치다. 같은 조건 반복 간 swing 차이가 이미 약 1 K이므로 그보다 작은 차이는 배치 판단을 뒷받침하지 않는다. 하한은 2u_c(0.653 K)다. 조건별 swing 대비 비율은 보조로 보고한다.
- 짝비교 개선폭: 상대 10 %(RC 대신 RC-NN을 쓸 최소 개선), 절대 0.1 K(수치 오차 0.01 K의 10배).

| 구분 | 항목 | 정의 | 기준 |
|---|---|---|---|
| 주 | Primary QoI | interp 조건별 RC-NN RMSE | ε_Q = max(1.0 K, 2u_c). 5개 중 4개 이상과 조건 평균이 기준 이하 |
| 주 | Peak QoI | interp 조건별 RC-NN peak 오차 | ε_pk = max(1.0 K, 2u_c). 같은 규칙 |
| 주 | RC 비교 | 조건별 RMSE_RC − RMSE_RC-NN | interp 5개 중 4개 이상, shift 3개 모두에서 RC-NN이 낫고, 각 test의 조건 평균 개선이 상대 10 % 이상, 절대 0.1 K 이상 |
| 주 | Physics | 9.2절 | 위반 0 |
| 주 | Cost | 예측 1회 latency | C = 100 ms 이하(power 갱신 주기 0.5 s 안에 재예측) |
| comparator | Persist 비교 | 조건별 skill | interp 4/5 이상, shift 3/3에서 skill > 0 |
| comparator | DLinear-X 비교 | shift 조건별 RMSE 차이 | shift 3/3에서 RC-NN이 낫다 |
| 보조 | DLinear-X 비교 (interp) | interp 조건별 차이 | 4/5 이상 |
| 보조 | shift 절대 오차 | shift RMSE, peak | ε와 비교해 보고 |
| 보조 | 미열람 subset | I01-32B-A/B의 주 판정 | 4.2절 결론 규칙 |
| – | Stress | shift | 오차 증가가 예상된다. RC-NN이 ε을 넘으면 burst 부하에는 관측 오차만큼 margin을 두거나 쓰지 않는다. RC보다 나쁜 조건에서만 RC로 돌아간다 |

comparator(DLinear-X) 포함 여부와 설정은 LOCK 전에 확정하고, 포함한 모델은 함께 한 번에 평가한다. fail한 항목은 결론에 그대로 반영한다.

### 9.4 Shift와 실패 분석 (M26)
- 오차를 lead time, 조건, burst on/off 단계로 나눈다.
- RC-NN의 window RMSE가 가장 큰 window 하나를 분석한다. 원인을 closure(RC와 RC-NN의 차이, closure 입력의 외삽), 입력 표현(0.25 s 격자 재실행. 판정을 바꾸지 않는 사후 진단), T_a(단조 누적 오차와 5.4절 크기 비교) 순으로 판단한다. 오차 상위 3개 window는 부록에 둔다.

### 9.5 Test guard
평가 모듈은 test split을 받으면 `results/LOCK`의 존재와 그 안의 `ACCEPTANCE.md` sha256 일치를 확인한다. LOCK은 `ACCEPTANCE.md` commit 뒤 한 번 만들고 commit hash를 적는다. test 평가는 모든 모델에 대해 한 번 실행하며 결과를 덮어쓰지 않는다.

### 9.6 민감도 (validation만, 부록)
- T_a ±1 K에 대한 RMSE와 bias 변화(5.4절 근사와 비교).
- 계획값 민감도: horizon 입력을 직전 10 s 평균 유지로 바꾼 성능.
- 미래 입력 해상도: RC 계열에 1 s 평균 입력을 준 성능. 변화가 RC-NN과 DLinear-X의 validation 차이의 절반을 넘으면 결과 해석에 적는다.

---

## 10. 그림과 표 (M22, M33)
모든 축에 물리량, 기호, 단위를 적고, caption에 split, 방법, 표본 수, 뒷받침하는 결론을 적는다.

| 번호 | 내용 |
|---|---|
| F1 | 모델 schematic(선택된 구조, 관측기, closure 위치) |
| F2 | reference 검증(해석해 대조, 수렴 order, 입력 격자) |
| F3 | split coverage와 4.3절 강도 |
| F4 | held-out trajectory(interp, shift 각 1개)와 실패 사례 |
| F5 | lead time별 RMSE |
| F6 (T2) | 종합 evidence table |
| T3 | 조건별 분해 표 |
| F7 | 식별 가능성(rank 표, 칸별 복원 오차·예측 변동, r*). 지면에서는 F2와 한 그림으로 합친다 |
| F8 | 선택된 closure와 학습 범위 |
| A1–A7 | 학습 진단, 탈락 closure, DLinear-X Adam, bootstrap, 실패 상위 window, 민감도, control |

**Evidence table 열**: 모델, split(train / validation / interp 전체 / interp 미열람 / shift), run 수, window 수, RMSE(seed 평균 ± SD), skill, peak·swing 오차, bias, e∞, coverage, physics check, parameter 수, 학습 시간, latency, acceptance 판정.

---

## 11. 코드와 재현성 (M28–M30)

```
PINN/
├── configs/data.yaml   경로, 격자, horizon, stride, split, 정답 규칙
├── gputherm/
│   ├── config.py       설정과 경로
│   ├── data/           io, quality(형식·오염 검사), windows(window 필터), grid(입력 격자, window 입력, 정답)
│   ├── physics/        ladder(모델·전달함수), discretize(정확 이산화), simulate(예측), observer
│   ├── estimation/     rc_fit(60 s window loss 추정)
│   ├── pipeline/       inventory(1단계), uncertainty(2단계)
│   └── verification/   structural, causality, exact, refine, balance, behavior, alignment, grad, fit_selftest, cooldown_modes
├── records/, results/, tables/
├── README.md, requirements.txt, run_all.sh
└── ACCEPTANCE.md, report.ipynb, figures/   (8–10단계)
```

라이브러리(`data`, `physics`, `estimation`)에는 실행 코드가 없고, 실행 모듈은 `python -m gputherm.<하위패키지>.<모듈>`로 실행한다. `run_all.sh`가 처음부터 모든 결과를 다시 만든다. 절대 경로는 `configs/data.yaml`에만 둔다. 원시 데이터는 제출물에 넣지 않고 md5 목록과 접근 방법을 README에 적는다.

---

## 12. 실행 순서와 통과 조건

| 단계 | 작업 | 통과 조건 | 상태 |
|---|---|---|---|
| 0 | 환경·승인 기록 | 버전 고정, 승인 기록 | 완료 |
| 1 | inventory, 형식 검사, control, split | 고유 run 51, 2.4절 일치 | 완료 |
| 2 | 측정 불확도 | train·validation만 | 완료 |
| 3 | 물리 모델 구현, 6.1–6.3·6.5·6.7 | 6.8절 기준 | 완료 |
| 4 | 6.6 시간 정렬, 6.4 gate | 시간 정렬 0.02 K, 결정 규칙 적용 | 6.6 완료 |
| 5 | RC 추정과 구조 선택, α_ref 결정, 검증 재실행 | 초기값 20 이상, 선택은 validation | |
| 6 | RC-NN 두 후보(seed 5), closure 선택, DLinear-X, 9.6절 | 선택은 validation | |
| 7 | train + validation 최종 재학습 | 4.2절 순서 | |
| 8 | `ACCEPTANCE.md` 고정, LOCK | 날짜, commit hash, 수치 | |
| 9 | test 평가 1회 | 모든 모델 동시, 덮어쓰기 없음 | |
| 10 | 그림, 표, `report.ipynb`, 보고서 | 새 kernel 재실행 성공 | |

**우선순위**: 주 claim과 M10에 직접 쓰이는 항목(데이터 검사, u_c, 검증, gate, Persist·RC·RC-NN·DLinear-X, 9.3절 판정, F1–F8, T3)이 필수이고, 탈락 closure, DLinear-X Adam, bootstrap, 민감도, control 그림은 부록이다. 개발은 RC 검증과 구조 선택 → RC와 RC-NN 비교 → 잠금과 평가 순으로 하며, comparator 설정은 LOCK 전에 확정한다.

---

## 13. 보고서, 한계, 인용 (M31–M35)

**보고서 구성(5–7쪽)**: 질문과 claim(0.5쪽), 물리 모델(1), 검증과 식별 가능성(1), 데이터와 split(1), 모델과 학습(0.75), 결과(1.5), 한계와 결론(0.75).

**결론 범위 (M34)**: RTX PRO 6000 1장, cap 600 W, 측정된 조건, 정확한 계획 입력. 실제 장치 결과와 dry-lab 결과를 구분한다. 다음 과제로 새로 측정해 처음부터 잠근 test, 다른 냉각 조건·GPU에서의 반복, run별 주변 온도 추정, 비선형 data-only 모델과의 비교를 적는다.

**한계**
- 판정 조건 수가 적고(interp 5, shift 3), 조건끼리도 workload 구조를 공유한다.
- test 대부분은 사전 탐색에서 선형 RC 결과를 본 조건이며, 열람 영향은 interp의 I01-32B로만 점검한다. shift에는 미열람 조건이 없다.
- 전역 T_a로 run별 주변 온도 차이(±0.55 K, RMSE 기여 약 0.29 K)를 설명하지 못한다. closure가 이와 상관된 drift를 흡수했을 가능성은 idle 복귀 check로 일부 점검한다.
- 두 번째 node의 물리적 정체를 확인할 수 없다. 일반형 2-node는 구조적으로 식별되지 않아 사다리형을 썼다.
- 정수 °C와 0.5 s power 갱신으로 1 s 미만 응답을 다루지 않고, 양자화가 시간상수 추정을 편향시킨다(6.6절). sensor 절대 정확도는 알 수 없다.
- horizon 입력은 측정값이며 실제 계획의 오차는 다루지 않는다.
- DLinear-X 비교는 물리 구조와 모델 형태의 차이를 함께 포함한다.
- 반복 run은 연속 실행되었고 무작위화되지 않았다. I01-32B-A r01은 긴 로딩 뒤 측정되었다. 일부 T02 run은 warmup 중 이전 반복의 메모리가 남아 있었다.

**인용 (M35)**
- AlShaikh Saleh, Chawla, Bayhan, Abu-Rub, Ghrayeb, "A Physics-Aware Framework for Short-Term GPU Power Forecasting of AI Data Centers," KDD 2026, doi:10.1145/3770855.3818033, arXiv:2605.04074(학회명과 DOI는 공식 저장소 README의 bibtex 기준).
- Raissi, Perdikaris, Karniadakis, J. Comput. Phys. 378, 686–707, 2019.
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (DLinear).
- Chen et al., Neural ODE, NeurIPS 2018 (torchdiffeq).
- Van Loan, "Computing integrals involving the matrix exponential," IEEE TAC 23(3), 1978.
- JCGM 100:2008 (GUM).
- KENTECH Real LAB 실험계획서(2026-03). 소프트웨어 버전은 `requirements.txt`. AI 도구(Claude Code) 사용 범위를 밝힌다.

**Evidence map**

| 항목 | 위치 | 항목 | 위치 |
|---|---|---|---|
| M1 결정 | 1.2 | M19 입출력과 구조 | 7, 8 |
| M2 claim | 1.3 | M20 loss·최적화 | 5.6, 8.4 |
| M3 범위·독립 단위 | 1.4 | M21 선택·반복성 | 8.2, 8.5 |
| M4 acceptance | 9.3 | M22 학습 진단 | 8.5, A1 |
| M5 지배 방정식 | 5.1 | M23 짝비교 | 9.1 |
| M6 기호·단위 | 5.2 | M24 QoI | 9.1 |
| M7 조건·closure | 5.3 | M25 physics check | 9.2 |
| M8 가정·생략 | 5.4 | M26 shift·실패 분석 | 4.3, 9.4 |
| M9 척도 | 5.5 | M27 비용 | 8.6 |
| M10 실험 | 2.7, 2.8, 6.4 | M28 clean execution | 11, 12 |
| M11 provenance | 2.1–2.5 | M29 README | 11 |
| M12 split·누수 | 0, 4.1, 4.2 | M30 설정·원시 증거 | 11 |
| M13 coverage·stress | 4.2, 4.3 | M31 실패·제외 기록 | 2.5, 2.7, 5.6 |
| M14 baseline 적합성 | 7, 8.1, 8.3 | M32 보고서 논리 | 1.5, 13 |
| M15 독립 정확성 | 6.1, 6.7 | M33 그림·표기 | 10 |
| M16 refinement | 6.2, 6.6 | M34 결론 범위 | 13 |
| M17 reference 오차 | 3 | M35 인용·안전 | 13 |
| M18 물리 거동 | 6.5, 9.2 | | |
