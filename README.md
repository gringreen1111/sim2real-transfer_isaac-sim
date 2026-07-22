# Sim-to-Real Transfer for Weld Bead Segmentation

NVIDIA Isaac Sim으로 생성한 용접 비드 합성데이터로 U-Net segmentation 모델을 학습하고, 실데이터 대비 sim-to-real 전이 성능을 검증한 실험 코드입니다.

synth & real ex)
<img width="128" height="72" alt="frame_0014" src="https://github.com/user-attachments/assets/9265151a-d38e-414c-83b2-41b4fc6b3106" />
<img width="128" height="72" alt="rgb_00282" src="https://github.com/user-attachments/assets/2216af02-1118-4312-bbe5-fdcb025622d9" />

**ResNet34 / MiT-B2** 두 backbone에 대해 **synth-only / real-only / synth+real(2단계 파인튜닝)** 세 가지 학습 방식을 적용해 총 6개 모델을 비교했습니다.

---

## 1. 실험 설정

| 항목 | 내용 |
|---|---|
| 합성데이터 | Isaac Sim Replicator, Domain Randomization 적용 — 5,300장 |
| 실데이터 | Intel RealSense D405 촬영 용접 시편 — 251장 |
| 모델 | U-Net + ResNet34(+SCSE) / U-Net + MiT-B2(SegFormer) |
| Loss | BCEWithLogitsLoss + Dice (smooth=1.0) |
| 평가 지표 | IoU, Dice |

### Backbone별 학습 설정 차이

| | ResNet34 | MiT-B2 |
|---|---|---|
| 해상도 | 1280×1280 | 1024×1024 |
| Optimizer | Adam (lr 1e-4 단일) | AdamW (encoder 3e-5 / decoder 2e-4) |
| Scheduler | ReduceLROnPlateau | Linear warmup → Cosine decay |
| AMP | BF16 | FP16 / BF16 |
| Batch | 6 | 2 × accum 4 (유효 8) |
| Decoder attention | SCSE | 미사용 |

synth+real은 단순 concat이 아니라 **합성 best 가중치 로드 → 실데이터 미세조정**의 2단계 순차 파인튜닝입니다.

---

## 2. 결과

test set 크기가 조건마다 달라(251 / 39 / 26장), 공정 비교를 위해 **모든 모델이 공통으로 포함하는 held-out 실이미지 26장**으로 6개 모델을 재평가했습니다.

| Backbone | 학습 방식 | IoU | Dice |
|---|---|---|---|
| ResNet34 | synth-only | 0.169 | 0.207 |
| MiT-B2 | synth-only | **0.492** | 0.590 |
| ResNet34 | real-only | 0.936 | 0.967 |
| ResNet34 | synth+real | 0.926 | 0.961 |
| MiT-B2 | real-only | **0.940** | 0.969 |
| MiT-B2 | synth+real | 0.931 | 0.964 |

### 학습 시간 (MiT-B2 기준)

| | 시간 |
|---|---|
| real-only (1단계) | 약 42분 |
| synth+real (합성 사전학습 248분 + 파인튜닝 21분) | 약 269분 |

Isaac Sim 렌더링 시간은 위 수치에 포함되지 않았습니다.

---

## 3. 핵심 결론

**1) 합성만 학습했을 때 backbone 차이가 결정적이다.**
MiT-B2(IoU 0.492)가 ResNet34(0.169)를 크게 앞섭니다. 합성 in-domain val IoU는 두 모델 모두 0.99 수준이었지만 실이미지로 넘어가면 격차가 벌어집니다. 트랜스포머 backbone이 sim-to-real 도메인 갭에 훨씬 강건합니다.

**2) 실데이터를 쓰면 backbone 차이는 사라진다.**
두 backbone 모두 IoU 0.93~0.94로 사실상 동률입니다. backbone 선택의 이점은 실데이터가 부족하거나 합성에 의존할 때만 드러납니다.

**3) 합성 사전학습이 최종 정확도를 올리지는 못했다.**
real-only 0.940 vs synth+real 0.931 (ResNet은 0.936 vs 0.926). 26~39장 test에서는 측정 노이즈 범위로, 통계적으로 동률로 보는 것이 타당합니다. 파인튜닝 단계 수렴이 빠른 것(ep1 val IoU 0.48~0.54)은 사실이나, 1단계 합성 사전학습 비용을 제외했을 때의 이야기입니다.

**단, 이 결론은 합성에 가장 불리한 조건(실데이터 풍부 + 학습/테스트 분포 일치 + 단발성 학습)에서 나온 것입니다.** 합성데이터의 가치는 (a) 합성 base 모델을 여러 라인·부품에 재사용하는 경우, (b) 실데이터·라벨이 부족한 경우, (c) 배포 환경이 학습 분포와 달라지는 경우에 드러날 수 있으며, 본 실험이 다루지 않은 영역입니다.

---

## 4. 한계 및 후속 실험

### 한계
1. **Precision·Recall 산출 불가** — 추론 로그에 픽셀 단위 TP/FP/FN이 기록되지 않음
2. **순수 backbone A/B 비교가 아님** — 해상도(1280/1024), optimizer(Adam/AdamW), LR 체계, 스케줄러, AMP, 증강, 데이터 분할이 모두 달라 각 파이프라인에 튜닝된 결과의 비교임
3. **Inference time 직접 비교 불가** — 해상도가 다름 (ResNet34 ≈22ms/45FPS @1280px, MiT-B2 ≈36ms/28FPS @1024px)
4. 합성 생성 조건과 실이미지 수집 조건이 코드·로그에 남아 있지 않아 실험 노트로 보완 필요
5. ResNet34 학습 코드에 학습 시간 로깅이 없어 MiT-B2와 시간 비교 불가

### 권고
합성데이터의 실제 효용을 정량화하려면 **데이터 효율 곡선 실험**이 필요합니다. 실데이터를 25/50/100/188장으로 줄여가며 real-only와 synth+real을 각각 학습해 두 곡선을 비교하면, 소량 데이터 구간에서 합성 사전학습의 이득(또는 무이득)을 명확히 확인할 수 있습니다. 추가로 분포 밖(다른 재질·조명) 소규모 test set으로 강건성을 별도 평가하는 것이 바람직합니다.

---

## 5. 파일 구성

| 파일 | 설명 |
|---|---|
| `new_isaac_train.py` | ResNet34 — 합성데이터 학습 (synth-only) |
| `new_isaac_train_resnet_real.py` | ResNet34 — 실데이터 단독 학습 (real-only) |
| `new_isaac_train_resnet_fin.py` | ResNet34 — 합성 사전학습 후 실데이터 파인튜닝 (synth+real) |
| `new_isaac_train_mit.py` | MiT-B2 — 합성데이터 학습 (synth-only) |
| `new_isaac_train_mit_real.py` | MiT-B2 — 실데이터 단독 학습 (real-only) |
| `new_isaac_train_mit_fin.py` | MiT-B2 — 합성 사전학습 후 실데이터 파인튜닝 (synth+real) |
| `Manuscript.pdf` | 상세 실험 보고서 (학습 파라미터, 학습 곡선, 전체 결과) |

---

