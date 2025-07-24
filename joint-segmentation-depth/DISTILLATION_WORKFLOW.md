# Knowledge Distillation Workflow

## Overview
이 문서는 SafeTrip-Q 데이터셋에서 Knowledge Distillation을 사용하여 Hydranet 모델을 학습시키는 전체 워크플로우를 설명합니다.

## Step 1: Teacher Model 학습 (완료)
SegFormer 모델을 Cityscapes pre-trained weights로 시작하여 SafeTrip segmentation 데이터로 fine-tuning했습니다.

```bash
python train_teacher.py \
    --config configs/cfg_safetrip.yaml \
    --model segformer \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.0001 \
    --project safetrip-teacher
```

**Output**: `checkpoints/teacher/best_cityscapes_teacher.pth`

## Step 2: Pseudo Label 생성
학습된 teacher model을 사용하여 KITTI depth 데이터에 대한 pseudo segmentation label을 생성합니다.

```bash
python scripts/generate_pseudo_labels.py \
    --teacher-checkpoint checkpoints/teacher/best_cityscapes_teacher.pth \
    --kitti-root ../data/data_depth_annotated \
    --batch-size 8 \
    --confidence-threshold 0.7 \
    --output-path data/kitti_pseudo_labels.pkl
```

**Output**: `data/kitti_pseudo_labels.pkl`

### Confidence Threshold 선택 가이드
- `0.9`: 매우 높은 confidence만 사용 (정확하지만 적은 데이터)
- `0.7`: 균형잡힌 선택 (추천)
- `0.5`: 더 많은 데이터 활용 (노이즈 증가 가능)

## Step 3: Hydranet 학습 (with Distillation)
이제 KITTI 데이터도 pseudo segmentation label을 가지므로, 모든 데이터가 multi-task supervision을 받습니다.

```bash
python train.py \
    --config configs/cfg_safetrip.yaml \
    --project safetrip-distilled \
    --run-name hydranet-with-distillation
```

### 학습 데이터 구성
1. **SafeTrip Segmentation**: Surface + Polygon (실제 label)
2. **SafeTrip Depth**: 사용하지 않음
3. **KITTI Depth**: LiDAR depth (실제 label)
4. **KITTI Segmentation**: Teacher model의 pseudo label

### Expected Improvements
- **Multi-task Learning**: 모든 샘플이 두 task 모두 학습
- **Better Gradient Balance**: Segmentation과 Depth의 균형 개선
- **Domain Adaptation**: KITTI 이미지에 대한 segmentation 성능 향상

## Step 4: 성능 평가
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/cfg_safetrip.yaml
```

## Tips

### Teacher Model 성능이 낮은 경우
1. 더 많은 epoch 학습
2. Learning rate 조정
3. Data augmentation 강화
4. Batch size 증가

### Pseudo Label 품질 개선
1. Confidence threshold 조정
2. Test-time augmentation 적용
3. Multiple teacher ensemble

### Hydranet 학습 개선
1. Loss weight 조정 (현재 0.5:0.5)
2. KITTI ratio 조정 (현재 50%)
3. Depth preprocessing method 변경

## Monitoring

### WandB에서 확인할 메트릭
1. **Teacher Model**
   - `val/miou`: 0.5 이상이면 좋음
   - `val/acc`: 픽셀 정확도
   - Per-class IoU 분포

2. **Hydranet**
   - `val/seg_miou`: Segmentation 성능
   - `val/depth_rmse`: Depth 성능
   - 두 task의 loss 균형

## Troubleshooting

### "No pseudo labels found"
- Pseudo label 파일 경로 확인
- KITTI 데이터 경로 확인

### "Low confidence pseudo labels"
- Teacher model 성능 확인
- Confidence threshold 낮추기

### "NaN loss"
- Gradient clipping 확인
- Learning rate 낮추기
- Depth preprocessing 확인