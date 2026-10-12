"""gputherm: GPU die 온도의 60 s 예측을 위한 사다리형 RC, closure, 검증 도구 (PINN_experiment_spec.md).

하위 패키지
  data          원시 기록 읽기, 형식·오염 검사, window 필터, 입력 격자
  physics       사다리형 RC 정의, 정확 이산화, 예측, 관측기
  estimation    60 s window loss에 의한 RC 추정
  pipeline      단계 실행(inventory, 측정 불확도)
  verification  수치 검증과 dry-lab 실험
"""
