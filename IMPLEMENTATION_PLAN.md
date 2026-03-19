# I-JEPA 로컬 체크포인트 로딩 / TensorBoard 모니터링 정리 메모

## 0. 진행 현황 업데이트

### 이번 작업에서 완료한 내용
- `models/ijepa_wrapper.py`를 **최소 체크포인트 로더** 중심으로 단순화했다.
- `train/train_semantic_inpaint.py`를 **로컬 체크포인트 inspection + TensorBoard logging** 전용 스크립트로 정리했다.
- 불필요한 dry-run 학습 경로, stage 관리, optimizer grouping, fallback module inference를 제거했다.
- `configs/downstream_semantic_inpaint.yaml`를 최소 설정만 남기도록 축소했다.
- `README.md`를 현재 실제 동작 기준으로 다시 작성했다.

### 이번 작업에서 제거한 과한 설계(over-engineering)
1. **학습 파이프라인 선구현**
   - 아직 실제 데이터셋/학습 루프가 없는데 stage와 optimizer 정책이 먼저 들어가 있었다.
   - 현재 요구사항에는 필요하지 않아 제거했다.
2. **체크포인트 기반 모듈 구조 추론**
   - linear/MLP fallback을 state dict에서 추론하는 로직은 데모용으로는 흥미롭지만 유지보수 비용이 크다.
   - 현재는 “로컬 체크포인트를 안정적으로 읽고 요약한다”는 범위로 축소했다.
3. **dummy training metrics 기록**
   - synthetic loss와 random tensor 기반 수치는 실제 모니터링 가치가 낮다.
   - 실제 checkpoint 상태를 보여주는 메트릭만 남겼다.

### 남은 작업
- 실제 I-JEPA backbone 생성 코드가 정해지면 `LoadedIJEPA` 결과를 바탕으로 아키텍처에 state dict를 주입하는 thin adapter를 추가한다.
- 필요 시 TensorBoard에 component별 layer name histogram 혹은 key preview를 추가한다.
- 실제 downstream 학습이 필요해질 경우, inspection 스크립트와 분리된 별도 train entrypoint를 새로 만든다.

---

## 1. 현재 목표

현재 저장소의 1차 목표는 다음 한 줄로 정리한다.

> 로컬에 있는 I-JEPA 사전학습 체크포인트를 읽고, encoder / predictor / target encoder 상태를 TensorBoard로 빠르게 확인한다.

즉, 지금은 “semantic inpainting 학습기 전체”보다 **체크포인트 연결과 모니터링 경로를 단순하고 명확하게 유지하는 것**이 우선이다.

---

## 2. 단순화 후 구조

### 2.1 체크포인트 로더
- 입력: 로컬 checkpoint path
- 처리:
  - `torch.load(..., map_location="cpu")`
  - nested dict 또는 `state_dict` flat dict에서
    - `encoder`
    - `predictor`
    - `target_encoder` / `teacher_encoder` / `ema_encoder`
    를 추출
- 출력:
  - `LoadedIJEPA`
  - 요약 메트릭(`loaded 여부`, `tensor 개수`, `parameter 개수`)

### 2.2 TensorBoard 로깅
- inspection 결과 메트릭 dict를 생성한다.
- 숫자는 scalar, 문자열은 text로 기록한다.
- 목적은 “학습 loss 시뮬레이션”이 아니라 “체크포인트 상태 가시화”다.

---

## 3. 향후 확장 원칙

### 원칙 1. 로딩과 학습을 분리한다
체크포인트 inspection 로직은 가볍게 유지하고, 실제 학습 코드는 별도 entrypoint에 둔다.

### 원칙 2. 아키텍처 추론보다 명시적 adapter를 선호한다
backbone이 정해지면 `build_vit_huge_from_meta_checkpoint(...)` 같은 명시적 함수로 연결한다.

### 원칙 3. 현재 쓸모없는 추상화는 넣지 않는다
stage preset, optimizer policy, dummy loss는 실제 사용 시점에 도입한다.

---

## 4. 현재 실행 예시

```bash
python -m train.train_semantic_inpaint /path/to/ijepa_checkpoint.pt --log-dir runs/ijepa_checkpoint
```

```bash
tensorboard --logdir runs/ijepa_checkpoint
```
