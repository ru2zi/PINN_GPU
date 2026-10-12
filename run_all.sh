#!/usr/bin/env bash
# 처음부터 모든 결과를 다시 만든다 (spec 11–12절). 각 단계는 실패하면 종료 코드 1로 멈춘다.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1     # __pycache__를 만들지 않는다
run() { python -m "gputherm.$1"; }

# 1–2단계: 데이터
run pipeline.inventory            # inventory, 형식·오염 검사, control, split
run pipeline.uncertainty          # 측정 불확도 (train·validation)

# 3단계: 구현 검증 (대표 parameter)
run verification.structural       # 5.1 구조적 식별 가능성
run verification.causality        # 2.6 인과성, coverage 필터
run verification.exact            # 6.1 독립 구현 대조
run verification.refine           # 6.2 수렴, 입력 격자
run verification.balance          # 6.3 에너지 수지
run verification.behavior         # 6.5 정상상태 거동
run verification.grad             # 6.7 torch 값·미분
run verification.fit_selftest     # 6.7 RC 추정 구현
run verification.cooldown_modes   # 5.5 시간 척도 (train·validation)

# 4단계
run verification.alignment        # 6.6 시간 정렬 (약 5분)
