# I-JEPA Decoder Baseline for Semantic Inpainting

이 저장소는 이제 **실제로 학습이 가능한 경량 semantic inpainting baseline**을 포함한다. 현재 구현은 공식 I-JEPA 원본 학습 코드를 재현하는 것이 아니라, 다음 두 목적에 초점을 둔다.

1. 로컬 이미지 폴더만으로 바로 학습 가능한 downstream baseline 제공
2. 선택적으로 로컬 I-JEPA 체크포인트를 읽어 TensorBoard에 요약 정보 기록

즉, 지금 기준으로는 **"학습 가능 여부"에 대한 답은 예**이며, 학습은 `train/train_semantic_inpaint.py`와 YAML 설정 파일로 바로 수행할 수 있다.

---

## 현재 구현 범위

### 1. 실제 학습 루프

구현 위치: `train/train_semantic_inpaint.py`

포함 내용:

- `ImageFolderDataset`
  - `jpg/png/bmp` 이미지를 재귀적으로 읽는다.
  - PIL 기반으로 정사각형 리사이즈 후 `[C, H, W]` 텐서로 변환한다.
- `RandomBlockMaskSampler`
  - patch 단위 정사각형 hole mask를 무작위 생성한다.
- `SemanticInpaintingModel`
  - patch tokenizer
  - transformer + cross-attention context predictor
  - progressive convolution decoder
  - 를 묶어서 hole token 예측 + 복원 이미지를 만든다.
- `run_training(...)`
  - DataLoader 생성
  - mask 샘플링
  - semantic loss + reconstruction loss 계산
  - AdamW step 수행
  - TensorBoard 기록
- `load_config(...)`
  - YAML 설정 파일을 읽어서 학습/로깅 하이퍼파라미터를 한 번에 로드한다.

### 2. 체크포인트 inspection

구현 위치: `models/ijepa_wrapper.py`, `train/train_semantic_inpaint.py`

- `IJEPAWrapper.load(...)`
  - 로컬 `.pt` / `.pth` 체크포인트를 CPU로 로드한다.
- 지원하는 구성요소
  - `encoder`
  - `predictor`
  - `target_encoder` / `teacher_encoder` / `ema_encoder`
- `inspect_checkpoint(...)`
  - 체크포인트를 읽고 component별 tensor 개수 / parameter 개수 / 로드 여부를 TensorBoard에 남긴다.

> 참고: 현재 baseline trainer는 **공식 I-JEPA backbone을 복원해서 weight를 직접 주입하는 단계까지는 구현하지 않았다.** 대신 로컬 checkpoint inspection과 학습용 baseline을 분리해 두었다.

---

## 요구사항 설치

`requirements.txt` 기준 설치:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

주요 패키지:

- torch
- numpy
- Pillow
- tensorboard
- pytest
- PyYAML

---

## 설정 파일 기반 학습

기본 설정 파일은 `configs/downstream_semantic_inpaint.yaml` 이다.

예시:

```yaml
checkpoint_path: ""
image_dir: data/train_images
epochs: 10
steps_per_epoch: null
seed: 0
device: cpu

tensorboard:
  enabled: true
  log_dir: runs/semantic_inpaint
  flush_secs: 10
  image_log_interval: 50
  max_images: 4

progress:
  enabled: true
  refresh_rate: 1

data:
  image_dir: data/train_images
  image_size: 64
  batch_size: 8
  num_workers: 0
  shuffle: true

mask:
  patch_size: 16
  min_hole_patches: 1
  max_hole_patches: 4

model:
  token_dim: 128
  decoder_hidden_dim: 256

optimizer:
  lr: 0.001
  weight_decay: 0.0001

loss:
  lambda_jepa: 1.0
  lambda_rgb: 0.5
```

학습 실행 시에는 데이터셋 로딩 직후 `Loaded ... images from ...` 형태의 요약이 먼저 출력되고, `tqdm`이 설치되어 있으면 epoch별 progress bar가 함께 표시된다. `tqdm`이 없더라도 step별 loss 로그를 stdout으로 남기도록 되어 있다.

또한 Food-101처럼 `train/<class_name>/*.jpg` 아래에 **symlink된 이미지 파일**이 들어 있는 구조도 그대로 읽는다. 반대로 깨진 symlink가 섞여 있으면 시작 시점에 즉시 에러를 내서 어느 링크가 문제인지 확인할 수 있다.

학습 실행 커맨드는 짧게 유지할 수 있다.

```bash
python -m train.train_semantic_inpaint --config configs/downstream_semantic_inpaint.yaml
```

짧은 smoke test를 하고 싶다면 YAML에서 다음 값들만 줄이면 된다.

- `epochs: 1`
- `steps_per_epoch: 2`
- `data.batch_size: 2`
- `tensorboard.enabled: false`

---

## 체크포인트 inspection 병행

로컬 I-JEPA 체크포인트 요약도 함께 남기고 싶다면 YAML에서 `checkpoint_path`를 채우면 된다.

```yaml
checkpoint_path: /path/to/ijepa_checkpoint.pt
```

기록되는 메트릭 예시:

- `encoder_loaded`
- `predictor_loaded`
- `target_encoder_loaded`
- `encoder_parameter_count`
- `predictor_parameter_count`
- `target_encoder_parameter_count`
- `train/loss`
- `train/semantic_loss`
- `train/reconstruction_loss`
- `train/images/input`, `train/images/mask`, `train/images/masked_input`, `train/images/reconstruction`, `train/images/comparison`

---

## TensorBoard

학습 중에는 TensorBoard의 Images 탭에서 원본 이미지, hole mask, mask 적용 입력, 복원 결과, 그리고 이들을 가로로 이어 붙인 비교 스트립을 확인할 수 있다. 기본값으로는 50 step마다 최대 4장까지 기록하며, `tensorboard.image_log_interval`, `tensorboard.max_images`로 조절할 수 있다.


```bash
tensorboard --logdir runs
```

기본 접속 주소:

```text
http://localhost:6006
```

---

## 테스트

현재 테스트는 다음을 검증한다.

- patch mask 변환
- pixel mask 적용
- hole token scatter
- flat / nested checkpoint 로딩
- checkpoint inspection summary 생성
- YAML config 로딩
- TensorBoard logging helper 동작
- random block mask 생성
- baseline model forward shape
- 이미지 폴더 DataLoader 생성
- 1-step 학습 smoke test

실행:

```bash
pytest
```
