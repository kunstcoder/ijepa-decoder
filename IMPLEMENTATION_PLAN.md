# Meta I-JEPA Downstream Semantic Hole Reconstruction 개발 계획

## 1. 목표 정의

### 1.1 최종 목표
- **목표 태스크**: 입력 이미지의 일부 영역을 가린 뒤, 해당 **마스킹 영역의 semantic latent를 복원**하고, 이를 이용해 이미지 공간에서 hole을 재구성한다.
- **비목표**:
  - I-JEPA 자체를 픽셀 복원형 MAE로 재구성하지 않는다.
  - pretrained encoder/predictor/teacher를 새로 학습하는 것은 우선순위에서 제외한다.
  - RGB dense inpainting만을 위한 저수준 texture 복원은 1차 목표가 아니다.
- **핵심 질문**: 논문과 공식 README 시각화 예시처럼, **latent 수준의 semantic prediction만으로도 masked region의 의미적 복원**이 가능한 downstream pipeline을 만들 수 있는가?

### 1.2 설계 원칙
- I-JEPA는 **pixel regressor가 아니라 semantic prior**로 활용한다.
- masked hole에 대한 위치 정렬된 predictor token을 최대한 보존한다.
- decoder는 predictor가 만든 hole semantic token을 **공간 정보를 유지한 채** 픽셀 공간으로 펼치는 역할에 집중한다.
- reconstruction loss만으로 끌고 가지 않고, 가능한 한 **JEPA semantic objective를 보조 손실로 유지**한다.

---

## 2. 시스템 개요

### 2.1 기본 아키텍처
권장 기본 구조는 다음과 같다.

1. **Context encoder**
   - visible patch만 입력받아 context token을 생성한다.
2. **Predictor**
   - masked hole 위치마다 semantic latent를 예측한다.
3. **Target encoder (teacher / EMA)**
   - 원본 이미지에서 masked hole의 target semantic latent를 생성한다.
4. **New decoder**
   - predictor output을 full token grid에 scatter한 뒤, hole region RGB 혹은 patch representation을 복원한다.

즉, 전체 파이프라인은 아래와 같다.

```text
masked image / visible patches
    -> encoder
    -> predictor(masked positions only)
    -> predicted hole latent z_hat
    -> scatter to full patch grid
    -> decoder
    -> reconstructed hole image / patch output
```

동시에 teacher branch는 다음 역할을 담당한다.

```text
original image
    -> target encoder
    -> target hole latent h
```

학습 시에는 `z_hat`과 `h`를 semantic latent level에서 정렬한다.

### 2.2 핵심 해석
- predictor 출력은 이미 **“이 hole 위치에 어떤 semantic content가 있어야 하는가”**를 담은 **position-aligned token**으로 볼 수 있다.
- 따라서 README의 visualization처럼 average pooling 후 generative decoder로 넘기는 방식은 가능하지만, **inpainting처럼 위치 정합이 필요한 태스크**에는 적합하지 않다.
- downstream hole reconstruction에서는 **patch별 predictor token 전체를 유지**한 채 decoder conditioning으로 사용하는 것이 핵심이다.

---

## 3. 구현 범위와 단계별 산출물

## Phase 0. 환경 및 체크포인트 적재

### 목표
- official checkpoint에서 필요한 모듈만 안정적으로 불러오는 최소 실행 경로를 만든다.

### 할 일
- pretrained checkpoint loader 구현
  - `encoder`
  - `predictor`
  - `target_encoder`
- downstream 실험용 entrypoint 구성
  - train script
  - eval / visualization script
- config 분리
  - backbone 관련 설정
  - mask 생성 설정
  - decoder 설정
  - loss weight 설정

### 산출물
- checkpoint loading 유틸
- downstream config yaml 또는 dataclass
- dry-run 가능한 inference 스크립트

---

## Phase 1. 마스크 인터페이스 정리

### 목표
- 공식 I-JEPA mask 표현을 downstream reconstruction 목적에 맞게 확장한다.

### 배경 문제
공식 코드는 patch token 시퀀스에 대해 **“keep할 patch index를 gather”**하는 방식이다. 이 구조는 whole-patch masking에는 잘 맞지만, 자유형 irregular mask에서는 다음 문제가 발생한다.

- patch 중간만 가려지는 경우 **부분 가시 정보 누수**가 생길 수 있다.
- 샘플마다 mask 길이가 다르면 tensor shape 정렬이 까다롭다.
- 기존 multi-block collator는 길이 차이를 `min_keep_*` 기준으로 잘라 shape을 맞추는데, 이는 downstream dense task에 불리할 수 있다.

### 구현 전략
1. **binary hole mask -> patch mask 변환기 구현**
   - 입력: `[B, 1, H, W]` binary mask
   - 출력: patch 단위 visible / hole index
2. **pixel masking 선반영 옵션 추가**
   - irregular hole에서 leakage를 줄이기 위해 patchify 전에
     `x_ctx = x * (1 - mask)`
     를 적용하는 옵션을 둔다.
3. **variable-length mask 처리 개선**
   - 1차 버전: patch hole 개수를 고정 또는 bucket화
   - 2차 버전: padding + attention mask 방식으로 일반화
4. **mask sampler 2종 지원**
   - I-JEPA와 가까운 `multi-block mask`
   - 실제 inpainting용 `irregular/free-form hole mask`

### 우선순위 결정
- 초기 프로토타입에서는 **whole-patch aligned mask**를 먼저 사용한다.
- 이후 irregular hole을 지원하되, 반드시 **patch leakage 방지 처리**를 같이 넣는다.

### 산출물
- mask conversion 유틸
- multi-block / irregular sampler
- padding-aware collator 또는 batch formatter

---

## Phase 2. Predictor output을 decoder 입력으로 변환

### 목표
- predictor가 반환한 hole token을 full spatial token grid로 복원한다.

### 구현 전략
1. predictor output `z_hat_hole` 확보
   - shape 예시: `[B, N_hole, D]`
2. hole index를 이용해 full grid로 scatter
   - 결과: `[B, N_patch, D]`
3. visible patch representation과 결합
   - 옵션 A: visible 위치에는 encoder context token 사용
   - 옵션 B: visible 위치에는 teacher/encoder의 detached token 사용
   - 옵션 C: visible RGB patch embedding을 별도 fusion
4. 2D spatial reshape
   - `[B, H_p, W_p, D]` 또는 `[B, D, H_p, W_p]`
5. decoder로 전달

### 권장 기본안
- **visible 위치는 encoder token 유지**
- **hole 위치는 predictor token scatter**
- 결과적으로 full token canvas를 만들고, decoder는 이를 받아 hole만 복원

### 산출물
- `scatter_hole_tokens()` 유틸
- full token canvas builder
- decoder input adapter

---

## Phase 3. Decoder MVP 구현

### 목표
- semantic token grid를 실제 hole reconstruction으로 연결하는 최소 decoder를 만든다.

### 권장 1차 decoder
빠른 프로토타입에서는 deterministic decoder가 적합하다.

#### 구조 예시
- 입력: full token grid `[B, D, H_p, W_p]`
- backbone:
  - Conv upsampling block
  - 또는 lightweight ViT/Transformer decoder
- 출력:
  - hole RGB patch
  - 또는 전체 RGB 이미지 예측 후 hole loss만 계산

### 권장 이유
- 구현 단순성
- latent semantic signal이 실제 spatial reconstruction으로 이어지는지 빠르게 검증 가능
- diffusion 계열보다 디버깅 비용이 낮음

### 후속 고도화 옵션
1. **visible image conditioning 강화**
   - masked image / visible pixel feature를 decoder에 추가 입력
2. **U-Net형 decoder**
   - 다중 스케일 feature로 경계 복원 개선
3. **DiT / diffusion decoder**
   - high-fidelity texture가 필요해질 때 확장

### 주의점
- README visualization 스타일처럼 **average pooled predictor token**을 쓰지 않는다.
- hole reconstruction은 공간 배치가 중요하므로 **patch-level spatial token 전체 유지**가 필수다.

### 산출물
- baseline deterministic decoder
- token-to-image projection head
- masked-hole-only inference 함수

---

## Phase 4. Loss 설계

### 목표
- semantic prediction 능력을 유지하면서 decoder reconstruction 성능을 학습한다.

### 기본 손실식
다음과 같은 multi-objective가 권장된다.

```text
L_total = λ_jepa * L_semantic + λ_rgb * L_recon + λ_aux * L_aux
```

세부 항목은 아래와 같다.

#### 1) Semantic latent loss
- `L_semantic = SmoothL1(z_hat_hole, h_teacher_hole)`
- I-JEPA의 본래 objective를 유지하는 핵심 손실
- predictor가 teacher latent geometry를 유지하도록 유도

#### 2) Pixel reconstruction loss
- `L_recon = || M ⊙ (x_hat - x) ||_1`
- loss는 반드시 hole 영역에만 적용
- 옵션:
  - Charbonnier
  - masked L2
  - perceptual loss 혼합

#### 3) Auxiliary losses (선택)
- LPIPS on hole region
- patch adversarial / feature matching
- boundary consistency loss

### 권장 초기 비중
- 1차 실험은 semantic loss 비중을 높인다.
- 예시:
  - `λ_jepa = 1.0`
  - `λ_rgb = 0.5`
  - `λ_aux = 0.0 ~ 0.1`
- reconstruction 품질이 전혀 안 나오면 decoder capacity를 먼저 올리고, semantic loss 비중은 천천히 조절한다.

### 산출물
- combined loss module
- hole-only masked loss 유틸
- loss ablation config

---

## Phase 5. 학습 전략

### 목표
- pretrained I-JEPA의 semantic prior를 최대한 보존하면서 downstream decoder를 학습한다.

### 학습 단계 제안

#### Stage A. Decoder warm-up
- encoder / predictor / target encoder는 freeze
- decoder만 학습
- 목적: predictor token이 reconstruction에 실제로 usable한지 검증

#### Stage B. Predictor light fine-tuning
- predictor와 decoder만 학습
- encoder는 freeze 유지
- 목적: downstream hole semantics에 predictor를 적응

#### Stage C. Optional end-to-end fine-tuning
- encoder 일부 block unfreeze
- low learning rate 적용
- 목적: 태스크 특화 적응

### teacher 업데이트 전략
- pretrained target encoder를 그대로 freeze하거나,
- predictor/encoder를 미세조정할 경우 EMA teacher 유지 옵션 제공

### 추천 시작점
- **Stage A부터 시작**한다.
- decoder-only로도 semantic hole completion이 보이는지 먼저 확인한다.

### 산출물
- freeze/unfreeze schedule
- optimizer parameter group
- EMA update 옵션

---

## 4. 데이터 및 마스킹 전략

### 4.1 데이터셋 선택
초기 검증은 복잡한 open-domain 데이터보다 구조가 단순한 벤치부터 시작하는 것이 좋다.

#### 추천 순서
1. ImageNet subset / COCO subset
2. Places / ADE20K 같은 scene-heavy 데이터
3. 실제 downstream 도메인 특화 데이터

### 4.2 마스크 분포
단일 irregular hole만 쓰기보다 분포를 혼합한다.

#### 권장 조합
- 50%: I-JEPA 스타일 multi-block mask
- 30%: large rectangle / box mask
- 20%: irregular free-form mask

### 이유
- I-JEPA pretrained prior와의 정합성 유지
- inpainting 형태 일반화 확보
- semantic completion과 spatial completion을 동시에 학습

### 4.3 해상도 전략
- 초기 실험: backbone patch size에 잘 맞는 입력 해상도 사용
- patch size 14 계열이면 token grid가 깔끔하게 나오는 해상도 우선
- irregular mask 실험은 patch alignment 영향을 반드시 로그로 추적

---

## 5. 실험 설계

## 5.1 반드시 확인할 핵심 가설

### 가설 A
**Predictor latent만으로 hole의 semantic layout을 복원할 수 있다.**

검증 방법:
- decoder-only warm-up 실험
- hole region reconstruction 시 클래스/객체 대략 형태가 살아나는지 확인

### 가설 B
**Average pooling 없이 patch-level token 전체를 유지해야 spatially aligned reconstruction이 가능하다.**

검증 방법:
- full token conditioning vs pooled latent conditioning 비교
- hole 내부 구조 보존 정도 비교

### 가설 C
**semantic loss를 같이 둬야 I-JEPA prior가 덜 무너진다.**

검증 방법:
- RGB-only 학습 vs RGB+JEPA 학습 비교
- downstream reconstruction 및 latent alignment 비교

### 가설 D
**mask distribution을 multi-block과 irregular의 혼합으로 구성할 때 가장 안정적이다.**

검증 방법:
- multi-block only
- irregular only
- mixed schedule

---

## 5.2 베이스라인 실험 표

| ID | 설정 | 목적 |
|---|---|---|
| B0 | visible image만 입력하는 일반 inpainting decoder | 비교 기준선 |
| B1 | I-JEPA predictor token + deterministic decoder | 최소 제안 방식 |
| B2 | B1 + semantic loss | JEPA objective 유지 효과 확인 |
| B3 | B2 + irregular mask support | 실제 hole mask 확장 |
| B4 | B2 + stronger decoder/U-Net | decoder 병목 확인 |
| B5 | B2 + diffusion/DiT decoder | 고화질 확장 |

---

## 5.3 평가지표

### 정량 지표
- hole-only L1 / L2
- PSNR / SSIM on masked region
- LPIPS on masked region
- patch-level latent alignment error

### 정성 지표
- 객체 존재 여부가 semantic하게 맞는가
- hole 내부 구조가 주변 문맥과 일관적인가
- texture보다도 **형태 / 배치 / semantic plausibility**가 맞는가

### 권장 시각화
- original image
- masked input
- predicted hole only
- merged reconstruction
- teacher latent similarity map
- predictor token attention or similarity visualization

---

## 6. 구현 우선순위

## Sprint 1. 최소 실행 경로
- checkpoint loading
- patch-aligned mask support
- predictor hole token 추출
- scatter + simple conv decoder
- hole L1 reconstruction

### 성공 기준
- 단일 이미지 inference가 동작한다.
- hole 영역에서 대략적인 semantic structure가 복원된다.

## Sprint 2. JEPA objective 재도입
- teacher latent loss 추가
- decoder-only vs predictor+decoder fine-tuning 비교
- mixed mask schedule 적용

### 성공 기준
- latent alignment가 안정화된다.
- RGB-only 대비 semantic consistency가 개선된다.

## Sprint 3. irregular hole 지원
- binary mask -> patch mask 변환
- pre-patch pixel masking 적용
- padding-aware collator 도입

### 성공 기준
- 자유형 마스크에서도 leakage 없이 학습이 가능하다.
- patch-aligned 실험 대비 성능 저하 원인을 분석할 수 있다.

## Sprint 4. 고도화
- stronger decoder 또는 diffusion decoder
- visible feature fusion 강화
- perceptual loss / LPIPS 추가

### 성공 기준
- semantic plausibility를 유지하면서 시각 품질이 향상된다.

---

## 7. 기술적 리스크와 대응

### 리스크 1. irregular hole에서 patch leakage 발생
**대응**
- patch-aligned mask부터 시작
- pixel masking 선반영
- mask coverage ratio 로깅

### 리스크 2. decoder가 너무 약해서 latent 유효성을 오판
**대응**
- 최소 2종 decoder capacity 실험
- decoder bottleneck과 latent bottleneck을 분리해서 해석

### 리스크 3. RGB loss가 semantic prior를 붕괴시킴
**대응**
- semantic loss 유지
- warm-up 단계에서 backbone freeze
- low LR fine-tuning

### 리스크 4. predictor output과 reconstruction target의 정합 문제
**대응**
- scatter index 검증용 unit test 작성
- patch visualization으로 위치 alignment 확인

### 리스크 5. README식 latent visualization을 inpainting에 그대로 오해 적용
**대응**
- pooled latent 실험은 비교용으로만 유지
- 기본 경로는 항상 patchwise token conditioning 사용

---

## 8. 권장 코드 구조

```text
project/
  configs/
    downstream_semantic_inpaint.yaml
  models/
    ijepa_wrapper.py
    hole_token_scatter.py
    decoder_baseline.py
    losses.py
  data/
    mask_samplers.py
    collators.py
    datasets.py
  train/
    train_semantic_inpaint.py
    eval_semantic_inpaint.py
  notebooks_or_scripts/
    visualize_reconstruction.py
```

### 모듈 책임 분리
- `ijepa_wrapper.py`
  - checkpoint load
  - encoder/predictor/teacher forward
- `hole_token_scatter.py`
  - hole index 처리
  - full grid canvas 복원
- `decoder_baseline.py`
  - token grid -> image
- `losses.py`
  - semantic + RGB 결합 손실
- `mask_samplers.py`
  - multi-block / irregular sampler
- `collators.py`
  - variable-length mask batch 정렬

---

## 9. 최종 판단 기준

다음 질문에 답할 수 있으면 프로젝트 1차 목표를 달성한 것이다.

1. **I-JEPA predictor latent만으로 hole 내부의 semantic structure를 복원할 수 있는가?**
2. **patchwise latent conditioning이 pooled latent보다 inpainting에 유리한가?**
3. **semantic loss를 유지할 때 reconstruction의 semantic plausibility가 실제로 개선되는가?**
4. **irregular hole에서도 pretrained prior를 해치지 않고 확장 가능한가?**

---

## 진행 현황 업데이트

### 완료된 작업
- [x] Sprint 1 최소 실행 경로용 코드 스캐폴딩 추가
- [x] patch-aligned mask 변환기 및 pixel masking 유틸 추가
- [x] predictor hole token scatter 유틸과 baseline conv decoder 추가
- [x] semantic + hole-only RGB 결합 loss 추가
- [x] dry-run 가능한 학습 엔트리포인트 및 기본 설정 파일 추가
- [x] scatter 정합성과 dry-run 경로를 검증하는 테스트 추가

### 다음 작업
- [ ] 실제 I-JEPA 체크포인트 구조에 맞춘 로더 구체화
- [ ] irregular/free-form mask sampler 추가
- [ ] decoder warm-up / predictor fine-tuning stage 분리
- [ ] 시각화 및 eval 스크립트 보강

## 10. 실행 권장안 요약

가장 현실적인 첫 구현은 아래 순서다.

1. pretrained `encoder + predictor + target_encoder` 로드
2. patch-aligned box / multi-block mask부터 시작
3. predictor hole token을 full token grid로 scatter
4. simple deterministic decoder로 hole RGB 복원
5. `SmoothL1(latent) + masked L1(rgb)`로 학습
6. decoder-only warm-up 후 predictor 소폭 fine-tuning
7. 이후 irregular mask와 stronger decoder로 확장

즉, **I-JEPA를 직접 픽셀 오토인코더로 바꾸지 말고, hole semantic latent prior + reconstruction decoder 구조로 사용하는 것**이 본 계획의 핵심이다.
