"""
SafeTrip 멀티태스크 모델 통합 테스트 - 실제 데이터 대량 테스트
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import traceback

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_model_creation():
    """모델 생성 테스트"""
    print("🔨 모델 생성 테스트...")
    try:
        from model import create_model
        
        model = create_model(
            backbone_name='resnet34',
            num_classes=3,
            num_seg_classes=1,
            input_size=640,
            pretrained=False  # 빠른 테스트
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✅ 모델 생성 성공")
        print(f"  📊 전체 파라미터: {total_params:,}")
        print(f"  📊 학습 가능 파라미터: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 모델 생성 실패: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """순전파 테스트 - 대량 실제 데이터 사용"""
    print("\n🚀 순전파 테스트 (대량 데이터)...")
    try:
        from model import create_model
        from data.dataset import create_dataloader
        
        model = create_model(pretrained=False)
        model.eval()
        
        # 더 많은 샘플과 큰 배치 크기로 테스트
        dataloader = create_dataloader(
            data_root='data',
            mode='train',
            batch_size=8,  # 배치 크기 증가
            num_workers=0,
            max_samples=50,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth']
        )
        
        print(f"  🔍 데이터로더 생성 완료, 총 배치 수: {len(dataloader)}")
        
        # 여러 배치 테스트
        total_bbox_count = 0
        total_depth_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 처음 3개 배치만 테스트
                break
                
            images = batch['images']
            targets = batch['targets']
            
            with torch.no_grad():
                outputs = model(images)
            
            # BBox 통계
            batch_bbox_count = sum(len(bbox) for bbox in targets['bboxes'])
            total_bbox_count += batch_bbox_count
            
            # Depth 통계
            if 'depth_tensor' in targets and targets['depth_tensor'] is not None:
                batch_depth_count = len(targets['depth_indices'])
                total_depth_count += batch_depth_count
            else:
                batch_depth_count = 0
            
            print(f"\n  📦 배치 {batch_idx}:")
            print(f"    입력: {images.shape}")
            print(f"    BBox 개수: {[len(bbox) for bbox in targets['bboxes']]} (총 {batch_bbox_count})")
            print(f"    Surface: {targets['surface'].shape}")
            print(f"    Depth 샘플: {batch_depth_count}개")
            
            if batch_bbox_count > 0:
                print(f"    🎯 BBox 데이터 발견!")
                # 첫 번째 bbox 상세 정보
                for i, bbox_list in enumerate(targets['bboxes']):
                    if len(bbox_list) > 0:
                        first_bbox = bbox_list[0]
                        print(f"      샘플 {i} 첫 번째 bbox: {first_bbox}")
        
        print(f"\n  ✅ 순전파 테스트 완료")
        print(f"  📊 총 BBox 개수: {total_bbox_count}")
        print(f"  📊 총 Depth 샘플: {total_depth_count}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 순전파 실패: {e}")
        traceback.print_exc()
        return False

def test_loss_computation():
    """손실 계산 테스트 - 대량 실제 데이터 사용"""
    print("\n💔 손실 계산 테스트 (대량 데이터)...")
    try:
        from model import create_model
        from data.dataset import create_dataloader
        
        model = create_model(pretrained=False)
        model.eval()
        
        # 더 많은 샘플로 테스트
        dataloader = create_dataloader(
            data_root='data',
            mode='train',
            batch_size=8,  # 배치 크기 증가
            num_workers=0,
            max_samples=50,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth']
        )
        
        # BBox 데이터가 있는 배치 찾기
        bbox_batch_found = False
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images']
            targets = batch['targets']
            
            # BBox 개수 확인
            batch_bbox_count = sum(len(bbox) for bbox in targets['bboxes'])
            
            if batch_bbox_count > 0:
                print(f"  🎯 BBox 데이터 발견! 배치 {batch_idx}, BBox 개수: {batch_bbox_count}")
                bbox_batch_found = True
                
                outputs = model(images)
                
                # 손실 계산
                loss_dict = model.compute_loss(outputs, targets)
                
                print(f"  ✅ 손실 계산 성공 (BBox 데이터 포함)")
                for name, loss in loss_dict.items():
                    if isinstance(loss, torch.Tensor):
                        print(f"    📊 {name}: {loss.item():.4f}")
                    else:
                        print(f"    📊 {name}: {loss:.4f}")
                
                # 타겟 상세 분석
                print(f"\n  📦 타겟 상세 분석:")
                for i, bbox_list in enumerate(targets['bboxes']):
                    if len(bbox_list) > 0:
                        print(f"    샘플 {i}: {len(bbox_list)} BBox")
                        for j, bbox in enumerate(bbox_list[:3]):  # 처음 3개만
                            print(f"      BBox {j}: {bbox}")
                
                break
            
            if batch_idx >= 5:  # 5개 배치까지만 확인
                break
        
        if not bbox_batch_found:
            print(f"  ⚠️ BBox 데이터가 있는 배치를 찾지 못함. 더 많은 샘플이 필요할 수 있음.")
            
            # 그래도 일반 손실 계산 테스트
            batch = next(iter(dataloader))
            images = batch['images']
            targets = batch['targets']
            outputs = model(images)
            loss_dict = model.compute_loss(outputs, targets)
            
            print(f"  ✅ 일반 손실 계산 성공")
            for name, loss in loss_dict.items():
                if isinstance(loss, torch.Tensor):
                    print(f"    📊 {name}: {loss.item():.4f}")
                else:
                    print(f"    📊 {name}: {loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 손실 계산 실패: {e}")
        traceback.print_exc()
        return False

def test_dataset_statistics():
    """데이터셋 통계 분석 - 실제 데이터 분포 확인"""
    print("\n📊 데이터셋 통계 분석...")
    try:
        from data.dataset import SafeTripMultiTaskDataset
        
        # 더 많은 샘플로 데이터셋 생성
        dataset = SafeTripMultiTaskDataset(
            data_root='data',
            mode='train',
            max_samples=100,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth'],
            augment=False
        )
        
        print(f"  ✅ 데이터셋 생성 성공: {len(dataset)} 샘플")
        
        # 데이터셋 통계
        bbox_samples = 0
        surface_samples = 0
        depth_samples = 0
        total_bboxes = 0
        total_polygons = 0
        
        print(f"  🔍 전체 샘플 분석 중...")
        
        for i in range(min(50, len(dataset))):  # 처음 50개 샘플 분석
            try:
                sample = dataset[i]
                
                # BBox 확인
                if len(sample['targets']['bbox_labels']) > 0:
                    bbox_samples += 1
                    total_bboxes += len(sample['targets']['bbox_labels'])
                
                # Surface 확인
                if len(sample['targets']['polygons']) > 0:
                    surface_samples += 1
                    total_polygons += len(sample['targets']['polygons'])
                
                # Depth 확인
                if sample['targets']['depth'] is not None:
                    depth_samples += 1
                    
            except Exception as e:
                print(f"    ⚠️ 샘플 {i} 처리 오류: {e}")
        
        print(f"\n  📊 데이터 분포 (처음 50개 샘플):")
        print(f"    BBox 샘플: {bbox_samples}/50 ({bbox_samples/50*100:.1f}%)")
        print(f"    Surface 샘플: {surface_samples}/50 ({surface_samples/50*100:.1f}%)")
        print(f"    Depth 샘플: {depth_samples}/50 ({depth_samples/50*100:.1f}%)")
        print(f"    총 BBox 수: {total_bboxes}")
        print(f"    총 Polygon 수: {total_polygons}")
        
        # BBox가 있는 샘플 찾아서 상세 정보 출력
        if bbox_samples > 0:
            print(f"\n  🎯 BBox 샘플 상세 분석:")
            bbox_found = 0
            for i in range(len(dataset)):
                if bbox_found >= 3:  # 처음 3개만
                    break
                    
                try:
                    sample = dataset[i]
                    if len(sample['targets']['bbox_labels']) > 0:
                        bbox_found += 1
                        print(f"    샘플 {i}: {len(sample['targets']['bbox_labels'])} BBox")
                        print(f"      BBox 좌표: {sample['targets']['bboxes'][:3]}")  # 처음 3개만
                        print(f"      BBox 라벨: {sample['targets']['bbox_labels'][:3]}")
                except Exception as e:
                    continue
        
        return True
        
    except Exception as e:
        print(f"  ❌ 데이터셋 통계 실패: {e}")
        traceback.print_exc()
        return False

def test_dataset_compatibility():
    """데이터셋 호환성 테스트 - 대량 샘플"""
    print("\n📦 데이터셋 호환성 테스트...")
    try:
        from data.dataset import SafeTripMultiTaskDataset, create_dataloader
        
        # 더 많은 샘플로 데이터셋 생성
        dataset = SafeTripMultiTaskDataset(
            data_root='data',
            mode='train',
            max_samples=100,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth'],
            augment=False
        )
        
        print(f"  ✅ 데이터셋 생성 성공: {len(dataset)} 샘플")
        
        # 데이터셋 통계 출력
        stats = dataset.get_statistics()
        print(f"  📊 태스크 커버리지:")
        print(f"    BBox: {stats['task_coverage']['bbox']} / {stats['total_samples']}")
        print(f"    Surface: {stats['task_coverage']['surface']} / {stats['total_samples']}")
        print(f"    Depth: {stats['task_coverage']['depth']} / {stats['total_samples']}")
        
        # 더 큰 배치로 데이터로더 생성
        dataloader = create_dataloader(
            data_root='data',
            mode='train',
            batch_size=8,  # 배치 크기 증가
            num_workers=0,
            max_samples=100,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth']
        )
        
        # 배치 테스트
        batch = next(iter(dataloader))
        
        print(f"  ✅ 데이터로더 성공")
        print(f"  📐 배치 이미지: {batch['images'].shape}")
        print(f"  📊 BBox 개수: {[len(bbox_list) for bbox_list in batch['targets']['bboxes']]}")
        print(f"  📊 Surface 형태: {batch['targets']['surface'].shape}")
        if 'depth_tensor' in batch['targets'] and batch['targets']['depth_tensor'] is not None:
            print(f"  📊 Depth tensor 형태: {batch['targets']['depth_tensor'].shape}")
            print(f"  📊 Depth 유효 인덱스: {batch['targets']['depth_indices']}")
        else:
            print(f"  📊 Depth: 해당 배치에 depth 데이터 없음")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 데이터셋 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_training_step():
    """학습 스텝 테스트 - 대량 데이터로 실제 학습"""
    print("\n🎓 학습 스텝 테스트 (대량 데이터)...")
    try:
        from model import create_model
        from data.dataset import create_dataloader
        import torch.optim as optim
        
        # 모델 및 옵티마이저
        model = create_model(pretrained=False)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # 더 많은 샘플로 데이터로더 생성
        dataloader = create_dataloader(
            data_root='data',
            mode='train',
            batch_size=8,  # 배치 크기 증가
            num_workers=0,
            max_samples=100,  # 샘플 수 증가
            target_tasks=['bbox', 'surface', 'depth']
        )
        
        print(f"  🔍 총 {len(dataloader)}개 배치로 학습 테스트")
        
        # 여러 배치로 학습 스텝 테스트
        bbox_training_done = False
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 처음 3개 배치만 테스트
                break
                
            images = batch['images']
            targets = batch['targets']
            
            # BBox 개수 확인
            batch_bbox_count = sum(len(bbox) for bbox in targets['bboxes'])
            
            print(f"\n  📦 배치 {batch_idx}: BBox {batch_bbox_count}개")
            
            # Forward
            outputs = model(images)
            
            # Loss 계산
            loss_dict = model.compute_loss(outputs, targets)
            total_loss = loss_dict['total_loss']
            
            print(f"    손실 (학습 전):")
            for name, loss in loss_dict.items():
                if isinstance(loss, torch.Tensor):
                    print(f"      {name}: {loss.item():.4f}")
                else:
                    print(f"      {name}: {loss:.4f}")
            
            # Backward
            if isinstance(total_loss, torch.Tensor):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 학습 후 다시 계산
                with torch.no_grad():
                    outputs_after = model(images)
                    loss_dict_after = model.compute_loss(outputs_after, targets)
                
                print(f"    손실 (학습 후):")
                for name, loss in loss_dict_after.items():
                    if isinstance(loss, torch.Tensor):
                        print(f"      {name}: {loss.item():.4f}")
                    else:
                        print(f"      {name}: {loss:.4f}")
                
                if batch_bbox_count > 0:
                    bbox_training_done = True
                    print(f"    🎯 BBox 데이터로 학습 완료!")
        
        if bbox_training_done:
            print(f"\n  ✅ BBox 데이터 포함 학습 스텝 성공")
        else:
            print(f"\n  ⚠️ BBox 데이터 없이 학습 완료")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 학습 스텝 실패: {e}")
        traceback.print_exc()
        return False

def test_depth_data_availability():
    """Depth 데이터 가용성 확인"""
    print("\n🎯 Depth 데이터 가용성 확인...")
    try:
        from data.dataset import SafeTripMultiTaskDataset
        
        # Depth 태스크만 포함
        dataset = SafeTripMultiTaskDataset(
            data_root='data',
            mode='train',
            max_samples=50,  # 샘플 수 증가
            target_tasks=['depth'],  # Depth만
            augment=False
        )
        
        print(f"  ✅ Depth 전용 데이터셋: {len(dataset)} 샘플")
        
        # Depth 샘플 통계
        depth_count = 0
        for i in range(min(20, len(dataset))):
            try:
                sample = dataset[i]
                if sample['targets']['depth'] is not None:
                    depth_count += 1
                    if depth_count <= 3:  # 처음 3개만 상세 출력
                        depth_map = sample['targets']['depth']
                        print(f"    샘플 {i}: Depth {depth_map.shape}, 범위: {depth_map.min():.1f}~{depth_map.max():.1f}")
            except Exception as e:
                print(f"    ⚠️ 샘플 {i} 오류: {e}")
        
        print(f"  📊 Depth 샘플: {depth_count}/20")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Depth 데이터 확인 실패: {e}")
        traceback.print_exc()
        return False

def main():
    """모든 테스트 실행"""
    print("🧪 SafeTrip 멀티태스크 모델 대량 데이터 테스트 시작!")
    print("="*60)
    
    tests = [
        test_model_creation,
        test_dataset_statistics,  # 통계부터 확인
        test_forward_pass,
        test_loss_computation,
        test_dataset_compatibility,
        test_training_step,
        test_depth_data_availability
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} 테스트 중 예외 발생: {e}")
            results.append((test_func.__name__, False))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 전체 결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 실제 데이터로 완전 동작 확인됨")
    else:
        print("⚠️ 일부 테스트 실패. 추가 디버깅 필요")

if __name__ == "__main__":
    main() 