# GPU die 온도의 60 s 예측: 검증된 사다리형 RC와 신경망 closure

EE5106 PIML 중간 프로젝트. 설계와 판정 기준은 [`PINN_experiment_spec.md`](PINN_experiment_spec.md)(이하 spec)에, 실행 기록은 [`results/LOG.md`](results/LOG.md)에 있다.

## 문제

단일 RTX PRO 6000(power cap 600 W)에서 측정 power와 계획된 부하로 GPU die 온도를 60 s 앞까지 예측한다. 검증된 2-node RC(사다리형)에 workload 정보를 받는 신경망 closure를 더하면, 처음 보는 workload(interp)와 burst 부하(shift)에서 RC보다 정확해지는지를 조건 단위 판정 규칙으로 평가한다(spec 1절, 9.3절).

| 모델 | 역할 |
|---|---|
| Persist | 마지막 측정 온도 유지 |
| RC | 검증된 baseline, closure를 뺀 ablation |
| RC-NN | 제안 모델(RC + closure, RC parameter 고정) |
| DLinear-X | 선형 data-only comparator |

## 저장소 구성

```
PINN/
├── PINN_experiment_spec.md   실험 명세
├── configs/data.yaml         경로, 격자, horizon, stride, split, 정답 규칙
├── gputherm/                 코드 (아래)
├── run_all.sh                전체 재실행
├── requirements.txt          패키지 버전
├── records/                  승인, 장비, run 순서
├── results/                  inventory, split, 불확도, 검증 결과, LOG.md
├── tables/                   표
├── PINN_DATA/                원시 데이터 (읽기 전용, 제출물 제외)
├── github_code/              참고 구현 (인용만)
└── hw/                       과제 가이드와 템플릿
```

```
gputherm/
├── config.py                 설정 읽기, 프로젝트 경로
├── data/
│   ├── io.py                 run 탐색, 원시 기록 읽기, split 소속
│   ├── quality.py            형식·오염 검사, 제외 규칙
│   ├── windows.py            window 시작 시각, 정답 coverage 필터
│   └── grid.py               0.1 s 입력 격자, 출발 상태, 관측기 입력, look-back, 원샘플 정답
├── physics/
│   ├── ladder.py             사다리형 RC 정의, 전달함수, 정상상태, 시간상수, α_ref 변환
│   ├── discretize.py         Van Loan 정확 이산화(numpy·torch), 구간 평균
│   ├── simulate.py           정확해, 합성곱 예측, 해석해, RK4
│   └── observer.py           두 번째 node 관측기(numpy·torch)
├── estimation/
│   └── rc_fit.py             60 s window loss 추정, multistart
├── pipeline/
│   ├── inventory.py          1단계: inventory, 형식 검사, control, split
│   └── uncertainty.py        2단계: 측정 불확도
└── verification/             수치 검증과 dry-lab 실험 (실패하면 종료 코드 1)
    ├── common.py             대표 parameter, 공통 입출력
    ├── structural.py         구조적 식별 가능성 (spec 5.1)
    ├── causality.py          인과성, coverage 필터 (2.6)
    ├── exact.py              독립 구현 대조 (6.1)
    ├── refine.py             RK4 수렴, 입력 격자 (6.2)
    ├── balance.py            에너지 수지 (6.3)
    ├── behavior.py           정상상태 거동 (6.5)
    ├── alignment.py          시간 정렬 (6.6)
    ├── grad.py               torch 값·미분 (6.7)
    ├── fit_selftest.py       RC 추정 구현 (6.7)
    └── cooldown_modes.py     cooldown 시간 척도 (5.5)
```

라이브러리(`data`, `physics`, `estimation`)에는 실행 코드가 없다. 실행 모듈(`pipeline`, `verification`)은 `python -m gputherm.<하위패키지>.<모듈>`로 실행한다.

## 환경

- Python 3.10.12, PyTorch 2.1.0a0, NumPy 1.22.2, SciPy 1.11.1, pandas 1.5.3, SymPy 1.12 (`requirements.txt`).
- 대조용 독립 구현은 `github_code/torchdiffeq`(0.2.5)를 경로로 불러온다.
- 모든 계산은 float64이며, 비용 판정은 CPU 1 thread(`OMP_NUM_THREADS=1`) 기준이다.

## 데이터

`PINN_DATA/single_gpu_rtxpro6000_cap600W/`에 17개 조건 × 3 run = 51 run(10 ms NVML 기록)이 있다. 조건 정의는 `PINN_DATA/conditions.csv`, 파일별 md5는 `results/inventory.csv`에 있다. 원시 데이터는 제출물에 포함하지 않는다. 경로는 `configs/data.yaml`에서만 지정한다.

| split | 조건 | run |
|---|---|---|
| train | T02-A, T02-B, T02-32B-A, T02-32B-A2, T04-A, I01-Q7B-A | 18 |
| validation | T04-D, I01-8B-A, I02-A | 9 |
| test interp | I01-Q7B-B, I01-8B-B, I02-B, I01-32B-A, I01-32B-B | 15 |
| test shift | I01-Q7B-C, I01-8B-C, I02-C | 9 |

test run에는 `ACCEPTANCE.md`가 고정되기 전까지 어떤 모델도 적용하지 않는다(spec 0절).

## 실행

```bash
./run_all.sh                                   # 전체 재실행 (약 7분)
python -m gputherm.pipeline.inventory          # 1단계
python -m gputherm.pipeline.uncertainty        # 2단계
python -m gputherm.verification.exact          # 개별 검증 예
```

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 환경·승인 기록 | 완료 |
| 1 | inventory, 형식·오염 검사, control, split | 완료 |
| 2 | 측정 불확도 (u_c = 0.327 K) | 완료 |
| 3 | 물리 모델 구현과 검증 | 완료 |
| 4 | 시간 정렬 검증, 식별 가능성 gate | 시간 정렬 완료 |
| 5–10 | RC 추정, RC-NN·DLinear-X, 최종 재학습, 잠금, test 평가, 보고서 | 예정 |

## 검증 결과 (대표 parameter)

| 검증 | 결과 | 기준 |
|---|---|---|
| 해석해 대조 (상태 / 구간 평균) | ≤ 1.3e-8 K / ≤ 7.5e-11 K | 1e-6 K |
| RK4 관측 order, dt 0.1 s 오차 | 4.08–4.15, 2.7e-4 K | 3.5–4.5, 0.01 K |
| 입력 격자 0.25 s 대 0.1 s | 0.026 K | 0.05 K |
| 에너지 수지 상대 잔차 | ≤ 1.0e-14 | 1e-6 |
| 시간 정렬 (합성 기록) | 0.0017 K | 0.02 K |
| torch 값 / 미분 상대 오차 | ≤ 7.2e-13 K / ≤ 4.2e-9 | 1e-9 K / 1e-5 |

## 규약

- **단위**: 온도 °C(차이는 K), power W, 시간 s, 열용량 J/K, 열저항 K/W.
- **입력 격자**: 0.1 s bin 평균, 빈 bin은 앞 값으로 채운다.
- **출력**: 0.5 s 구간 시간 평균 (1/0.5)∫T_g dt. 정답도 원샘플의 0.5 s 구간 평균이다.
- **출발 상태**: T_g(t₀) = [t₀ − 0.1, t₀) 원샘플 평균, T_h(t₀)는 관측기.
- **parameter**: dict(`Cg`, `Ch`, `Rgh`, `Rha`, `alpha`). RC는 α = 1에서 추정한 뒤 α_ref로 옮긴다(spec 5.1절).

## 참고

- AlShaikh Saleh et al., KDD 2026 (PI-DLinear), `github_code/PI-DLinear`
- Zeng et al., AAAI 2023 (DLinear), `github_code/LTSF-Linear`
- Raissi et al., J. Comput. Phys. 2019, `github_code/PINNs`
- Chen et al., NeurIPS 2018 (torchdiffeq), `github_code/torchdiffeq`
- Van Loan, IEEE TAC 1978 (행렬 지수 적분)
