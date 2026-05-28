<h1 align="center">제11회 AIKUTHON</h1>

<p align="center">
  <b>Data Augmentation</b> · <b>Label Unbalance</b> · <b>CV model</b> · <b>attention rollout</b>
</p>

---

## Overview

| 항목 | 내용 |
| --- | --- |
| 팀원 | 조수빈, 엄혜진 |
| 대회 | 제11회 AIKUTHON |
| 핵심 접근 | Data Augmentation, Label Unbalance, CV model, attention rollout |
| 사용 모델 | ViT, SWIN Transformer, EfficientNet-b0 |
| 주요 기법 | Albumentations, Label Smoothing, WeightedRandomSampler, CosineAnnealing, Ensemble, Optuna, TTA |
| 최종 리더보드 성능 | AUC 0.89, 2nd Place |

## Contents

1. [사전 조사](#1-사전-조사)
2. [Data Preprocessing](#2-data-preprocessing)
3. [모델](#3-모델)
4. [실험 결과](#4-실험-결과)
5. [Conclusion](#5-conclusion)

---

## 1. 사전 조사

### 대회 설명

> **Task**  
> 본 대회에서 제공되는 데이터셋은 기술학술부원들이 직접 촬영한 약 1,450장의 음식 이미지를 포함합니다. 참가자는 주어진 이미지가 어떤 기기(iPhone/Galaxy)로 촬영되었는지 판별하는 이진 분류기(Binary Classifier)를 학습하게 됩니다.

| 구분 | 내용 |
| --- | --- |
| binary-class classification 모델 개발 | 이미지를 입력으로 받아 2가지 레이블 중에서 자동으로 classification을 할 수 있는 모델 설계 |
|  | 컴퓨터 비전 (CV) 기술, 딥러닝 기법, 신호처리 기법 등을 활용하여 최적의 모델 구축 |
| Label 의 불균형 | train dataset 내의 label들의 개수가 일관되지 않을 수 있음 |
|  | 모델이 특정 label에 쏠리지 않도록 평가할 수 있어야 함. |
| random resizing crop | 이미지 데이터에 random resizing crop 을 적용하였으므로 scale-invariance 를 고려할 수 있는 모델을 설계해야함. |

### 초기계획

- 전처리에 집중하자
- 다양한 CV 모델들에 적용해보자

---

## 2. Data Preprocessing

| 구분 | 적용 내용 |
| --- | --- |
| Digital Post Processing | Horizontal, Vertical, ShiftScaleRotate, ImageCompression, GaussNoise, Sharpen |
| Geometric Variations | RandomResizedCrop, RandomCrop |
| Label 개수의 일관성 | WeightedRandomSampler |
| Label Smoothing | 0, 1을 0.1, 0.9로 완화 |
| Normalization | ImageNet 기준 적용, 데이터셋 기준으로 직접 계산한 새로운 수치 적용 |

### Digital Post Processing

- RandomBrightnessContrast (밝기), HueSaturationValue (색조) 등 다양한 Albumentation 조합 실험
  - HueSaturation은 큰 효과가 없었음
- Noise, sharpen, blur 등 다양한 실험
  - ImageCompression으로 압축 과정의 뭉개짐 재현
  - GaussNoise로 노이즈 추가
  - Sharpen으로 경계선을 더 뚜렷하게 함

### Geometric Variations

- 이미지 비율 유지에 집중해 Resize 대신 Crop 사용
  - RandomResizedCrop으로 사물의 스케일 변화 대응
  - RandomCrop으로 해상도나 질감 손상 없이 국소적 특징 학습
- HorizontalFlip, VerticalFlip, ShiftScaleRotate

### Label 개수의 일관성

- WeightedRandomSampler: 샘플별 가중치 부여 (1 / 클래스별 개수)

### Label Smoothing

- Label Smoothing: 0, 1을 0.1, 0.9로 완화
- 논문 [When does label smoothing help?] "the clusters are much tighter, because label smoothing encourages that each example in training set to be equidistant from all the other class's templates."

### Normalization

- ImageNet 기준 적용
- 데이터셋 기준으로 직접 계산한 새로운 수치 적용

---

## 3. 모델

CV에서 널리 쓰이는 모델들을 사용함.

| 모델 | 설명 |
| --- | --- |
| ViT | google/vit-base-patch16-224 |
| SWIN Transformer | SWIN Transformer |
| EfficientNet-b0 | Transfer Learning |

| 설정 | 값 |
| --- | --- |
| Epochs | 30 |
| Optimizer | AdamW |
| LR scheduler | CosineAnnealing |
| Loss | BCEWithLogitsLoss |
| 저장 기준 | 과적합 방지를 위해 최고 AUC를 찍을 때마다 모델 저장 |

---

## 4. 실험 결과

기준: normalization, model

| Normalization | ViT | EfficientNet | SWIN |
| --- | --- | --- | --- |
| ImageNet normalization | 0.95 (kaggle submission: 0.87) | 0.9036 | 0.9648 (kaggle submission: 0.81) |
| dataset normalization | 0.9433 (kaggle submission: 0.86) | 0.8982 | 앞선 결과를 보고 굳이 할 필요 없다고 판단 |

### 첫 SOTA였던 ViT 모델

- Epoch 30 진행 후 추가로 30 진행
- 60번을 한 번에 이어서 돌리는 것보다 성능이 좋았음
- Warm Restart처럼 local minimum에서 벗어나는 효과
- Cosine annealing scheduling 사용

### Attention Rollout

- 대비되는 색상 경계에 집중하는 양상 확인
- 아이폰과 갤럭시의 이미지 튜닝 철학(ISP) 차이가 가장 극명하게 나타나는 구간
- slide 16 사진 추가

### Ensemble

- 정확도가 더 이상 오르지 않아 앙상블 기법 추가 진행
  - ViT + EfficientNet-b0 + SWIN 앙상블
  - 베이지안 최적화(Optuna): 머신러닝 및 딥러닝 모델의 하이퍼파라미터 최적화(HPO)를 자동화하는 파이썬 오픈소스 프레임워크
  - TTA (Test Time Augmentation)
- 리더보드상 0.89로 정확도 개선

---

## 5. Conclusion

| 구분 | 내용 |
| --- | --- |
| GOOD | 시각화 시 경계에 집중하는 특징이 보임 |
| BAD | 시각화 과정에서 노이즈, 특히 가장자리에도 집중하는 경향성이 보임 |
| 한계 | 시간 부족으로 앙상블을 많이 돌려보지 못해 최적의 앙상블 비율은 아니라고 판단 |
