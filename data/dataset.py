"""
Sideguide 멀티태스크 데이터셋 로더

데이터 구조:
- data/bbox/Bbox_xxxx/ : 각 폴더마다 이미지들 + XML 파일 (Object Detection)
- data/surface/Surface_xxxx/ : 각 폴더마다 이미지들 + MASK/ + XML 파일 (Surface Segmentation)  
- data/depth/Depth_xxx/ : 각 폴더마다 스테레오 이미지들 + Depth_xxx.conf 파일 (Depth Estimation)

지원 기능:

"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Albumentations 완전 제거 - 기본 변환만 사용

# 자체 모듈 import
from .annotation_parser import SafeTripAnnotationParser, ImageAnnotation, BoundingBox, Polygon  
from .calibration_parser import SafeTripCalibrationParser, CalibrationData

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeTripMultiTaskDataset(Dataset):
    """
    Sideguide 멀티태스크 데이터셋
    
    지원 태스크:
    1. Object Detection (Bbox) - 29개 클래스
    2. Surface Segmentation (Polygon) - 6개 클래스  
    3. Depth Estimation (Stereo) - 연속값
    
    부분 라벨링 지원: 모든 이미지에 모든 태스크의 라벨이 있을 필요 없음
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        mode: str = "train",
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        cache_images: bool = False,
        max_samples: Optional[int] = None,
        target_tasks: List[str] = ["bbox", "surface", "depth"]
    ):
        """
        Args:
            data_root: 데이터 루트 경로 (data/ 폴더)
            mode: 모드 ('train', 'val', 'test')
            image_size: 목표 이미지 크기 (height, width)
            augment: 데이터 증강 사용 여부
            cache_images: 이미지 캐싱 여부 (메모리 사용량 증가)
            max_samples: 최대 샘플 수 (None이면 전체)
            target_tasks: 대상 태스크 리스트
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.image_size = image_size
        self.augment = augment and mode == "train"
        self.cache_images = cache_images
        self.max_samples = max_samples
        self.target_tasks = target_tasks
        
        # 파서 초기화
        self.annotation_parser = SafeTripAnnotationParser(data_root)
        self.calibration_parser = SafeTripCalibrationParser(data_root)
        
        # 클래스 정보
        self.bbox_classes = self.annotation_parser.BBOX_CLASSES
        self.surface_classes = self.annotation_parser.SURFACE_CLASSES
        self.num_bbox_classes = len(self.bbox_classes)
        self.num_surface_classes = len(self.surface_classes)
        
        # 데이터 로드
        self._load_data()
        
        # 변환 파이프라인 설정
        self._setup_transforms()
        
        # 이미지 캐시
        self.image_cache = {} if cache_images else None
        
        print(f"🔗 데이터셋 로드: {len(self)} 샘플, 태스크: {target_tasks}")  # 간소화된 로그
    
    def _load_data(self):
        """모든 데이터 로드 및 인덱스 생성"""
        # 어노테이션 데이터 로드
        if "bbox" in self.target_tasks or "surface" in self.target_tasks:
            all_annotations = self.annotation_parser.parse_all_folders()
            self.annotation_data = all_annotations
        else:
            self.annotation_data = {}
        
        # 캘리브레이션 데이터 로드  
        if "depth" in self.target_tasks:
            all_calibrations = self.calibration_parser.parse_all_folders()
            self.calibration_data = all_calibrations
        else:
            self.calibration_data = {}
        
        # 통합 샘플 인덱스 생성
        self.samples = self._create_sample_index()
        
        # 최대 샘플 수 제한
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
        
        # logger.info(f"총 {len(self.samples)}개 샘플 로드")  # 로그 간소화
    
    def _create_sample_index(self) -> List[Dict[str, Any]]:
        """통합 샘플 인덱스 생성"""
        samples = []
        
        # 어노테이션 기반 샘플들
        for folder_key, annotations in self.annotation_data.items():
            folder_type = "bbox" if folder_key.startswith("bbox_") else "surface"
            folder_name = folder_key.split("_", 1)[1]  # "bbox_Bbox_0640" -> "Bbox_0640"
            
            for annotation in annotations:
                # 이미지 경로 구성
                image_path = self.data_root / folder_type / folder_name / annotation.filename
                
                if not image_path.exists():
                    logger.warning(f"이미지 파일 없음: {image_path}")
                    continue
                
                sample = {
                    'image_path': image_path,
                    'annotation': annotation,
                    'folder_type': folder_type,
                    'folder_name': folder_name,
                    'has_bbox': folder_type == "bbox" and annotation.has_bboxes,
                    'has_surface': folder_type == "surface" and annotation.has_polygons,
                    'has_depth': False,  # 별도 처리
                    'depth_data': None
                }
                samples.append(sample)
        
        # Depth 기반 샘플들 (스테레오 이미지 쌍)
        for folder_name, calib_data in self.calibration_data.items():
            for left_img_path, disparity_path in calib_data.image_pairs:
                sample = {
                    'image_path': left_img_path,  # 좌측 이미지를 메인으로
                    'disparity_path': disparity_path,  # disparity 맵 경로 추가
                    'annotation': None,
                    'folder_type': 'depth',
                    'folder_name': folder_name,
                    'has_bbox': False,
                    'has_surface': False,
                    'has_depth': True,
                    'depth_data': calib_data
                }
                samples.append(sample)
        
        # 셔플 (train 모드에서만)
        if self.mode == "train":
            random.shuffle(samples)
        
        return samples
    
    def _setup_transforms(self):
        """데이터 변환 파이프라인 설정 - 기본 변환만 사용"""
        # Albumentations 완전 제거 - 기본 변환만 사용
        self.transform = self._basic_transform
    
    def _basic_transform(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """기본 이미지 변환 (Albumentations 없을 때)"""
        # 리사이즈
        h, w = image.shape[:2]
        target_h, target_w = self.image_size
        
        # 비율 유지하며 리사이즈
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # 패딩
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # 정규화 및 텐서 변환
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            'image': image_tensor,
            'scale': scale,
            'pad': (left, top, right, bottom)
        }
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """이미지 로드 (캐싱 지원)"""
        if self.image_cache is not None and str(image_path) in self.image_cache:
            return self.image_cache[str(image_path)]
        
        # 이미지 로드
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일 없음: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 캐싱
        if self.image_cache is not None:
            self.image_cache[str(image_path)] = image
        
        return image
    
    def _prepare_bbox_targets(self, annotation: ImageAnnotation, scale: float, pad: Tuple[int, int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """바운딩 박스 타겟 준비 (YOLO 형식)"""
        if not annotation.has_bboxes:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        
        left_pad, top_pad, right_pad, bottom_pad = pad
        img_h, img_w = annotation.height, annotation.width
        
        bboxes = []
        labels = []
        
        for bbox in annotation.bboxes:
            # 스케일링 및 패딩 적용
            x1 = (bbox.x1 * scale + left_pad) / self.image_size[1]
            y1 = (bbox.y1 * scale + top_pad) / self.image_size[0]
            x2 = (bbox.x2 * scale + left_pad) / self.image_size[1]
            y2 = (bbox.y2 * scale + top_pad) / self.image_size[0]
            
            # YOLO 형식으로 변환 (center_x, center_y, width, height)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # 유효성 검사
            if width > 0 and height > 0 and 0 <= center_x <= 1 and 0 <= center_y <= 1:
                bboxes.append([center_x, center_y, width, height])
                labels.append(self.annotation_parser.bbox_class_to_idx[bbox.label])
        
        if not bboxes:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        
        return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def _prepare_polygon_targets(self, annotation: ImageAnnotation, scale: float, pad: Tuple[int, int, int, int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """폴리곤 타겟 준비 (픽셀 마스크 + 폴리곤 좌표)"""
        if not annotation.has_polygons:
            return torch.zeros((1, *self.image_size), dtype=torch.float32), []
        
        left_pad, top_pad, right_pad, bottom_pad = pad
        
        # 픽셀 마스크 생성
        mask = np.zeros(self.image_size, dtype=np.uint8)
        polygon_coords = []
        
        for polygon in annotation.polygons:
            # 스케일링 및 패딩 적용하여 픽셀 좌표로 변환
            scaled_points = []
            pixel_points = []
            
            for x, y in polygon.points:
                # 정규화된 좌표로 변환
                scaled_x = (x * scale + left_pad) / self.image_size[1]
                scaled_y = (y * scale + top_pad) / self.image_size[0]
                
                # 범위 체크
                scaled_x = max(0, min(1, scaled_x))
                scaled_y = max(0, min(1, scaled_y))
                
                scaled_points.extend([scaled_x, scaled_y])
                
                # 픽셀 좌표로 변환 (마스크 생성용)
                pixel_x = int(scaled_x * self.image_size[1])
                pixel_y = int(scaled_y * self.image_size[0])
                pixel_points.append([pixel_x, pixel_y])
            
            if len(pixel_points) >= 3:  # 최소 3개 포인트
                # OpenCV로 폴리곤 마스크 생성
                polygon_array = np.array(pixel_points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_array], 1)  # 모든 surface를 1로 마킹
                
                # 정규화된 좌표 저장
                if len(scaled_points) >= 6:
                    polygon_coords.append(torch.tensor(scaled_points, dtype=torch.float32))
        
        # 마스크를 텐서로 변환 (1, H, W)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        
        return mask_tensor, polygon_coords
    
    def _prepare_depth_targets(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Depth 타겟 준비 (disparity 맵 로드)"""
        if not sample['has_depth']:
            return None
        
        try:
            # sample에서 직접 disparity 경로 가져오기 (calibration parser에서 이미 매칭됨)
            if 'disparity_path' in sample and sample['disparity_path'] != sample['image_path']:
                disparity_path = sample['disparity_path']
            else:
                # fallback: 기존 방식으로 찾기
                folder_name = sample['folder_name']  # "Depth_001"
                depth_folder = self.data_root / "depth" / folder_name
                
                # 좌측 이미지 파일명에서 disparity 파일명 추출
                left_img_path = sample['image_path']
                img_name = left_img_path.stem  # 확장자 제거
                
                # 파일명에서 식별자 추출 (예: ZED1_KSC_001032_L → ZED1_KSC_001032)
                if img_name.endswith('_L'):
                    base_name = img_name[:-2]
                elif img_name.endswith('_left'):
                    base_name = img_name[:-5]
                else:
                    base_name = img_name
                
                # Disparity 맵 경로 구성
                possible_disp_paths = [
                    depth_folder / f"{base_name}_disp.png",
                    depth_folder / f"{base_name}_disp16.png",
                ]
                
                disparity_path = None
                for disp_path in possible_disp_paths:
                    if disp_path.exists():
                        disparity_path = disp_path
                        break
                
                if disparity_path is None:
                    logger.warning(f"Disparity 맵 없음: {base_name}")
                    return torch.zeros(self.image_size, dtype=torch.float32)
            
            # Disparity 맵 로드 (grayscale)
            disparity_image = cv2.imread(str(disparity_path), cv2.IMREAD_UNCHANGED)
            
            if disparity_image is None:
                logger.warning(f"Disparity 맵 로드 실패: {disparity_path}")
                return torch.zeros(self.image_size, dtype=torch.float32)
            
            # 그레이스케일로 변환 (필요시)
            if len(disparity_image.shape) == 3:
                disparity_image = cv2.cvtColor(disparity_image, cv2.COLOR_BGR2GRAY)
            
            # 크기 조정
            disparity_resized = cv2.resize(disparity_image, (self.image_size[1], self.image_size[0]))
            
            # 정규화 (0-255 → 0-1)
            if disparity_resized.dtype == np.uint8:
                disparity_normalized = disparity_resized.astype(np.float32) / 255.0
            else:
                disparity_normalized = disparity_resized.astype(np.float32)
                # 16비트의 경우 적절히 정규화
                if disparity_normalized.max() > 1.0:
                    disparity_normalized = disparity_normalized / disparity_normalized.max()
            
            # 텐서로 변환 (1, H, W) - single channel depth map
            disparity_tensor = torch.from_numpy(disparity_normalized).unsqueeze(0)
            
            return disparity_tensor
            
        except Exception as e:
            logger.warning(f"Depth 타겟 준비 실패: {e}")
            return torch.zeros((1, *self.image_size), dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """샘플 반환"""
        sample = self.samples[idx]
        
        try:
            # 이미지 로드
            image = self._load_image(sample['image_path'])
            
            # 기본 변환 적용
            transform_result = self._basic_transform(image)
            transformed_image = transform_result['image']
            scale = transform_result['scale']
            pad = transform_result['pad']
            
            # 타겟 준비
            targets = {}
            
            if sample['annotation']:
                # BBox 타겟
                if "bbox" in self.target_tasks and sample['has_bbox']:
                    bbox_coords, bbox_labels = self._prepare_bbox_targets(
                        sample['annotation'], scale, pad
                    )
                    targets['bboxes'] = bbox_coords
                    targets['bbox_labels'] = bbox_labels
                else:
                    targets['bboxes'] = torch.zeros((0, 4))
                    targets['bbox_labels'] = torch.zeros((0,), dtype=torch.long)
                
                # Surface 타겟  
                if "surface" in self.target_tasks and sample['has_surface']:
                    surface_mask, polygon_coords = self._prepare_polygon_targets(
                        sample['annotation'], scale, pad
                    )
                    targets['surface'] = surface_mask
                    targets['polygons'] = polygon_coords
                else:
                    targets['surface'] = torch.zeros((1, *self.image_size), dtype=torch.float32)
                    targets['polygons'] = []
            else:
                # 어노테이션 없음
                targets['bboxes'] = torch.zeros((0, 4))
                targets['bbox_labels'] = torch.zeros((0,), dtype=torch.long)
                targets['surface'] = torch.zeros((1, *self.image_size), dtype=torch.float32)
                targets['polygons'] = []
            
            # Depth 타겟
            if "depth" in self.target_tasks:
                depth_target = self._prepare_depth_targets(sample)
                targets['depth'] = depth_target
            
            return {
                'image': transformed_image,
                'targets': targets,
                'metadata': {
                    'image_path': str(sample['image_path']),
                    'folder_type': sample['folder_type'],
                    'folder_name': sample['folder_name'],
                    'original_size': (image.shape[1], image.shape[0]),  # (width, height)
                    'scale': scale,
                    'pad': pad
                }
            }
            
        except Exception as e:
            logger.error(f"샘플 로드 오류 {idx}: {e}")
            # 빈 샘플 반환
            return {
                'image': torch.zeros((3, *self.image_size)),
                'targets': {
                    'bboxes': torch.zeros((0, 4)),
                    'bbox_labels': torch.zeros((0,), dtype=torch.long),
                    'surface': torch.zeros((1, *self.image_size), dtype=torch.float32),
                    'polygons': [],
                    'depth': None
                },
                'metadata': {
                    'image_path': str(sample['image_path']),
                    'folder_type': sample['folder_type'],
                    'folder_name': sample['folder_name'],
                    'original_size': (640, 640),
                    'scale': 1.0,
                    'pad': (0, 0, 0, 0)
                }
            }
    

    
    def get_class_info(self) -> Dict[str, Any]:
        """클래스 정보 반환"""
        return {
            'bbox_classes': self.bbox_classes,
            'surface_classes': self.surface_classes,
            'num_bbox_classes': self.num_bbox_classes,
            'num_surface_classes': self.num_surface_classes
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        stats = {
            'total_samples': len(self.samples),
            'task_coverage': {
                'bbox': sum(1 for s in self.samples if s['has_bbox']),
                'surface': sum(1 for s in self.samples if s['has_surface']),
                'depth': sum(1 for s in self.samples if s['has_depth'])
            },
            'folder_distribution': {}
        }
        
        # 폴더별 분포
        for sample in self.samples:
            folder_key = f"{sample['folder_type']}_{sample['folder_name']}"
            if folder_key not in stats['folder_distribution']:
                stats['folder_distribution'][folder_key] = 0
            stats['folder_distribution'][folder_key] += 1
        
        return stats


def multitask_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    멀티태스크 배치 콜레이션 함수
    
    가변 길이 타겟들을 처리하기 위한 커스텀 콜레이션
    """
    images = torch.stack([item['image'] for item in batch])
    
    # 타겟들 수집
    batch_targets: Dict[str, Any] = {
        'bboxes': [],
        'bbox_labels': [],
        'surface': [],
        'polygons': [],
        'depth': []
    }
    
    metadata_list = []
    
    for item in batch:
        targets = item['targets']
        
        # BBox 타겟
        batch_targets['bboxes'].append(targets['bboxes'])
        batch_targets['bbox_labels'].append(targets['bbox_labels'])
        
        # Surface 타겟 (픽셀 마스크)
        batch_targets['surface'].append(targets['surface'])
        
        # Polygon 타겟 (좌표)
        batch_targets['polygons'].append(targets['polygons'])
        
        # Depth 타겟
        depth = targets.get('depth')
        batch_targets['depth'].append(depth)
        
        metadata_list.append(item['metadata'])
    
    # Surface 마스크 스태킹
    batch_targets['surface'] = torch.stack(batch_targets['surface'])
    
    # Depth 타겟 스태킹 (None이 아닌 것들만)
    valid_depth_targets = [d for d in batch_targets['depth'] if d is not None]
    if valid_depth_targets:
        # 모든 depth 타겟이 같은 shape인지 확인
        shapes = [d.shape for d in valid_depth_targets]
        if len(set(shapes)) == 1:
            # 같은 shape이면 스택
            depth_indices = [i for i, d in enumerate(batch_targets['depth']) if d is not None]
            stacked_depth = torch.stack(valid_depth_targets)
            batch_targets['depth_tensor'] = stacked_depth
            batch_targets['depth_indices'] = depth_indices
        else:
            # 다른 shape이면 리스트로 유지
            batch_targets['depth_tensor'] = None
            batch_targets['depth_indices'] = []
    else:
        batch_targets['depth_tensor'] = None
        batch_targets['depth_indices'] = []
    
    return {
        'images': images,
        'targets': batch_targets,
        'metadata': metadata_list
    }


def create_dataloader(
    data_root: Union[str, Path],
    mode: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """데이터로더 생성 헬퍼 함수"""
    dataset = SafeTripMultiTaskDataset(
        data_root=data_root,
        mode=mode,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True,
        drop_last=(mode == "train")
    )


def main():
    """테스트 및 데모 실행"""
    # 데이터 루트 경로 설정
    data_root = Path("data")
    
    if not data_root.exists():
        logger.error(f"데이터 루트 폴더 없음: {data_root}")
        return
    
    # 데이터셋 생성
    logger.info("🚀 멀티태스크 데이터셋 생성 중...")
    dataset = SafeTripMultiTaskDataset(
        data_root=data_root,
        mode="train",
        image_size=(640, 640),
        augment=False,  # 테스트를 위해 비활성화
        max_samples=100,  # 테스트를 위해 제한
        target_tasks=["bbox", "surface"]  # depth는 제외 (이미지 매칭 이슈)
    )
    
    # 통계 출력
    stats = dataset.get_statistics()
    class_info = dataset.get_class_info()
    
    print("\n" + "="*80)
    print("📊 SafeTrip 멀티태스크 데이터셋 통계")
    print("="*80)
    
    print(f"📁 전체 샘플: {stats['total_samples']}개")
    print(f"📦 BBox 태스크: {stats['task_coverage']['bbox']}개")
    print(f"🔍 Surface 태스크: {stats['task_coverage']['surface']}개")
    print(f"🌊 Depth 태스크: {stats['task_coverage']['depth']}개")
    
    print(f"\n🎯 클래스 정보:")
    print(f"  BBox 클래스: {class_info['num_bbox_classes']}개")
    print(f"  Surface 클래스: {class_info['num_surface_classes']}개")
    
    # 샘플 데이터 테스트
    logger.info("🔍 샘플 데이터 테스트 중...")
    
    try:
        # 첫 번째 샘플 로드
        sample = dataset[0]
        
        print(f"\n📋 샘플 데이터 정보:")
        print(f"  이미지 형태: {sample['image'].shape}")
        print(f"  BBox 개수: {len(sample['targets']['bboxes'])}")
        print(f"  Polygon 개수: {len(sample['targets']['polygons'])}")
        print(f"  이미지 경로: {sample['metadata']['image_path']}")
        print(f"  폴더 타입: {sample['metadata']['folder_type']}")
        
        # 데이터로더 테스트
        logger.info("🔄 데이터로더 테스트 중...")
        dataloader = create_dataloader(
            data_root=data_root,
            mode="train",
            batch_size=4,
            num_workers=0,  # 윈도우에서 멀티프로세싱 이슈 방지
            max_samples=20,
            target_tasks=["bbox", "surface"]
        )
        
        batch = next(iter(dataloader))
        
        print(f"\n📦 배치 정보:")
        print(f"  이미지 배치 형태: {batch['images'].shape}")
        print(f"  배치 크기: {len(batch['metadata'])}")
        print(f"  첫 번째 BBox 개수: {len(batch['targets']['bboxes'][0])}")
        
        logger.info("✅ 데이터셋 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 테스트 실패: {e}")
        raise


if __name__ == "__main__":
    main() 