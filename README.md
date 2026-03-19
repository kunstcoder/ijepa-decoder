# I-JEPA Local Checkpoint Loader for Semantic Inpainting Experiments

이 저장소는 현재 **로컬에 있는 I-JEPA 사전학습 체크포인트를 불러오고**, 각 구성요소 상태를 **TensorBoard로 기록/모니터링**하는 최소 구현으로 정리되어 있다.

기존에는 dry-run 학습 단계, stage preset, optimizer grouping, dummy decoder 학습 경로까지 한 파일에 함께 들어 있어 실제 목적 대비 구조가 다소 복잡했다. 현재는 다음 두 가지에 집중한다.

1. 로컬 체크포인트 로드
2. TensorBoard 메트릭 기록

---

## 현재 구현 범위

### 1. 체크포인트 로딩

구현 위치: `models/ijepa_wrapper.py`

- `IJEPAWrapper.load(...)`
  - 로컬 `.pt` / `.pth` 체크포인트를 CPU로 로드한다.
- 지원하는 구성요소
  - `encoder`
  - `predictor`
  - `target_encoder` / `teacher_encoder` / `ema_encoder`
- 지원하는 체크포인트 형태
  - nested dict
  - `state_dict` 기반 flat dict
- 반환값
  - `LoadedIJEPA`
  - 각 구성요소의 tensor state dict와 요약 메트릭을 함께 제공한다.

### 2. TensorBoard 로깅

구현 위치: `train/train_semantic_inpaint.py`

- `inspect_checkpoint(...)`
  - 체크포인트를 로드한 뒤 메트릭 dict를 생성한다.
- `create_tensorboard_writer(...)`
  - `SummaryWriter`를 생성한다.
- `log_metrics_to_tensorboard(...)`
  - 문자열은 text로,
  - 숫자는 scalar로 기록한다.

기록되는 예시는 다음과 같다.

- `encoder_loaded`
- `predictor_loaded`
- `target_encoder_loaded`
- `encoder_parameter_count`
- `predictor_parameter_count`
- `target_encoder_parameter_count`
- `checkpoint_path`

---

## 제거/단순화한 내용

이번 정리에서 다음 항목을 기본 경로에서 제거했다.

- dry-run synthetic training step
- stage preset(`decoder_warmup`, `predictor_finetune`, `end_to_end`)
- optimizer parameter grouping
- dummy encoder/predictor/teacher 생성 경로
- checkpoint state dict로부터 module 구조를 추론하는 fallback loader
- 실제 checkpoint inspection 목적과 직접 관련 없는 학습 메트릭 계산

이유는 현재 요구사항이 **“사전학습 모델을 로컬에서 로드하고 TensorBoard로 모니터링”**하는 것이기 때문이다. 학습 파이프라인은 필요해질 때 다시 별도 모듈로 추가하는 편이 유지보수에 유리하다.

---

## 실행 방법

### 의존성

- Python 3.10+
- PyTorch
- tensorboard
- pytest

예시 설치:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch tensorboard pytest
```

### CLI 실행

```bash
python -m train.train_semantic_inpaint /path/to/ijepa_checkpoint.pt --log-dir runs/ijepa_checkpoint
```

TensorBoard 실행:

```bash
tensorboard --logdir runs/ijepa_checkpoint
```

브라우저에서 기본적으로 다음 주소를 연다.

```text
http://localhost:6006
```

---

## 설정 파일

예시 설정 파일:

```text
configs/downstream_semantic_inpaint.yaml
```

현재는 최소 필드만 유지한다.

```yaml
checkpoint_path: /path/to/ijepa_checkpoint.pt

tensorboard:
  enabled: true
  log_dir: runs/ijepa_checkpoint
  flush_secs: 10
```

---

## 테스트

현재 테스트는 다음을 검증한다.

- patch mask 변환
- pixel mask 적용
- hole token scatter
- flat / nested checkpoint 로딩
- checkpoint inspection summary 생성
- TensorBoard logging helper 동작

실행:

```bash
pytest
```
