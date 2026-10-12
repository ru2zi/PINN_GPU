# 측정 장비와 계산 환경

## 측정 장비 (데이터를 만든 GPU)

| 항목 | 내용 | 출처 |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition 1장 | 실험계획서 4.2.2절 |
| 메모리 | 96 GB GDDR7 | 실험계획서 4.2.2절 |
| TDP | 600 W. 모든 run은 power cap 600 W(100 %) | 실험계획서, `conditions.csv` |
| 수집 방식 | NVML C API(ctypes), 10 ms polling | 실험계획서 5.2절 |
| power 필드 | `NVML_FI_DEV_POWER_INSTANT`. 값 갱신 주기 약 0.5 s(중앙값 504 ms) | 데이터 확인(spec 2.3절) |
| 온도 | GPU sensor, 정수 °C | 데이터 확인 |
| 실험 장소 | KENTECH Real LAB | 실험계획서 |
| serial | (미기록) | 측정 서버 `nvidia-smi -q` 필요 |
| driver / CUDA / NVML 버전 | (미기록) | 측정 서버 `nvidia-smi -q` 필요 |
| chassis, 냉각 구성 | (미기록) | |
| 실내 온도 | (미기록). 대신 cooldown 점근값 26.6–28.1 °C(train·validation)를 보고한다 | spec 5.2절 |

교정 대체로 매 run의 warmup(600 W, 180 s)을 반복 control 측정으로 쓴다(spec 2.7절, `results/control.csv`).

## 계산 환경 (이 프로젝트를 실행하는 서버, 2026-10-11 확인)

| 항목 | 내용 |
|---|---|
| CPU | Intel Xeon Platinum 8480C, 2 socket, 224 thread. 이 작업 세션의 CPU affinity는 40개, `OMP_NUM_THREADS=1` |
| RAM | 2 TB |
| GPU | NVIDIA GPU 4장(power limit 700 W, driver 580.126.09). 다른 사용자와 공유하는 노드 |
| Python | 3.10.12 |
| 패키지 | `requirements.txt` |
| 비용 판정 | CPU 1 thread(`torch.set_num_threads(1)`) 시간으로 한다(spec 8.6절) |
| 참고 코드 commit | LTSF-Linear 0c11366, PI-DLinear 787cd5e, PINNs 932f50a, torchdiffeq 657943a(0.2.5) |
