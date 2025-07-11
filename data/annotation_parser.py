"""
SafeTrip 멀티태스크 CVAT XML 어노테이션 파서

새로운 데이터 구조:
- data/bbox/Bbox_xxxx/ : 각 폴더마다 이미지들 + XML 파일
- data/surface/Surface_xxxx/ : 각 폴더마다 이미지들 + MASK/ + XML 파일

지원 기능:
- 멀티폴더 자동 스캔 및 파싱
- BBox Detection (29개 클래스)
- Surface Segmentation (6개 클래스 + 속성)
- YOLO 형식 변환
- 통계 생성 및 검증
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import logging
import numpy as np
from collections import defaultdict, Counter

# Pydantic 스타일 검증 (Context7 권장사항)
try:
    from pydantic import BaseModel, Field, validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# 로깅 설정
logging.basicConfig(level=logging.WARNING)  # INFO에서 WARNING으로 변경
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """바운딩 박스 어노테이션 데이터클래스"""
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    occluded: bool = False
    z_order: int = 0
    
    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """YOLO 형식으로 변환 (normalized center_x, center_y, width, height)"""
        center_x = (self.x1 + self.x2) / (2 * img_width)
        center_y = (self.y1 + self.y2) / (2 * img_height)
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        return center_x, center_y, width, height
    
    @property
    def area(self) -> float:
        """바운딩 박스 면적"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass  
class Polygon:
    """폴리곤 세그멘테이션 어노테이션 데이터클래스"""
    label: str
    points: List[Tuple[float, float]]
    occluded: bool = False
    z_order: int = 0
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_yolo_segmentation(self, img_width: int, img_height: int) -> List[float]:
        """YOLO segmentation 형식으로 변환 (normalized polygon points)"""
        normalized_points = []
        for x, y in self.points:
            normalized_points.extend([x / img_width, y / img_height])
        return normalized_points


@dataclass
class ImageAnnotation:
    """단일 이미지 어노테이션 데이터클래스"""
    image_id: int
    filename: str
    width: int
    height: int
    bboxes: List[BoundingBox] = field(default_factory=list)
    polygons: List[Polygon] = field(default_factory=list)
    source_folder: str = ""  # 어느 폴더에서 왔는지 추적
    
    @property
    def has_bboxes(self) -> bool:
        return len(self.bboxes) > 0
    
    @property  
    def has_polygons(self) -> bool:
        return len(self.polygons) > 0


class SafeTripAnnotationParser:
    """SafeTrip 프로젝트 멀티폴더 CVAT XML 어노테이션 파서"""
    
    # 32개 객체 탐지 클래스 (알 수 없는 클래스들 추가)
    BBOX_CLASSES = [
        'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'scooter', 'person', 
        'dog', 'cat', 'wheelchair', 'stroller', 'carrier',  # 이동체 (12개)
        'traffic_light', 'traffic_sign', 'stop', 'pole', 'tree_trunk', 
        'bench', 'chair', 'table', 'potted_plant', 'fire_hydrant', 
        'parking_meter', 'kiosk', 'movable_signage', 'bollard', 'barricade',  # 고정체 (17개)
        'traffic_light_controller', 'power_controller', 'sidewalk'  # 추가된 클래스들 (3개)
    ]
    
    # 6개 표면 세그멘테이션 클래스
    SURFACE_CLASSES = [
        'sidewalk', 'roadway', 'bike_lane', 'alley', 'caution_zone', 'braille_guide_blocks'
    ]
    
    # 표면 속성 매핑
    SURFACE_ATTRIBUTES = {
        'sidewalk': ['blocks', 'cement', 'urethane', 'asphalt', 'soil_stone', 'damaged', 'other'],
        'roadway': ['normal', 'crosswalk'],
        'alley': ['normal', 'crosswalk', 'speed_bump', 'damaged'],
        'caution_zone': ['stairs', 'manhole', 'tree_zone', 'grating', 'repair_zone'],
        'braille_guide_blocks': ['normal', 'damaged'],
        'bike_lane': []  # 속성 없음
    }
    
    def __init__(self, data_root: Union[str, Path]):
        """
        Args:
            data_root: 데이터 루트 경로 (data/ 폴더)
        """
        self.data_root = Path(data_root)
        self.bbox_folders = list((self.data_root / "bbox").glob("Bbox_*")) if (self.data_root / "bbox").exists() else []
        self.surface_folders = list((self.data_root / "surface").glob("Surface_*")) if (self.data_root / "surface").exists() else []
        
        # 클래스 인덱스 매핑
        self.bbox_class_to_idx = {cls: idx for idx, cls in enumerate(self.BBOX_CLASSES)}
        self.surface_class_to_idx = {cls: idx for idx, cls in enumerate(self.SURFACE_CLASSES)}
        
        # logger.info(f"BBox 폴더 {len(self.bbox_folders)}개, Surface 폴더 {len(self.surface_folders)}개 발견")  # 로그 간소화
    
    def parse_cvat_xml(self, xml_path: Path) -> List[ImageAnnotation]:
        """CVAT XML 파일을 파싱하여 어노테이션 리스트 반환"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotations = []
            
            # XML 내 모든 이미지 순회
            for image_elem in root.findall('.//image'):
                try:
                    annotation = self._parse_image_element(image_elem, xml_path.parent.name)
                    if annotation:
                        annotations.append(annotation)
                except Exception as e:
                    logger.warning(f"이미지 파싱 실패 {image_elem.get('name', 'unknown')}: {e}")
                    continue
            
            # logger.info(f"{xml_path.name}: {len(annotations)}개 이미지 파싱 완료")  # 로그 간소화
            return annotations
            
        except ET.ParseError as e:
            logger.error(f"XML 파싱 오류 {xml_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"예상치 못한 오류 {xml_path}: {e}")
            return []
    
    def _parse_image_element(self, image_elem: ET.Element, source_folder: str) -> Optional[ImageAnnotation]:
        """개별 image 엘리먼트 파싱"""
        image_id = int(image_elem.get('id', 0))
        filename = image_elem.get('name', '')
        width = int(image_elem.get('width', 0))
        height = int(image_elem.get('height', 0))
        
        if not filename or not width or not height:
            logger.warning(f"이미지 메타데이터 불완전: {filename}")
            return None
        
        annotation = ImageAnnotation(
            image_id=image_id,
            filename=filename,
            width=width,
            height=height,
            source_folder=source_folder
        )
        
        # 바운딩 박스 파싱
        for box_elem in image_elem.findall('box'):
            bbox = self._parse_bbox_element(box_elem)
            if bbox:
                annotation.bboxes.append(bbox)
        
        # 폴리곤 파싱  
        for polygon_elem in image_elem.findall('polygon'):
            polygon = self._parse_polygon_element(polygon_elem)
            if polygon:
                annotation.polygons.append(polygon)
        
        return annotation
    
    def _parse_bbox_element(self, box_elem: ET.Element) -> Optional[BoundingBox]:
        """바운딩 박스 엘리먼트 파싱"""
        try:
            label = box_elem.get('label', '')
            if label not in self.bbox_class_to_idx:
                logger.warning(f"알 수 없는 bbox 클래스: {label}")
                return None
            
            return BoundingBox(
                label=label,
                x1=float(box_elem.get('xtl', 0)),
                y1=float(box_elem.get('ytl', 0)),
                x2=float(box_elem.get('xbr', 0)),
                y2=float(box_elem.get('ybr', 0)),
                occluded=box_elem.get('occluded', '0') == '1',
                z_order=int(box_elem.get('z_order', 0))
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"바운딩 박스 파싱 오류: {e}")
            return None
    
    def _parse_polygon_element(self, polygon_elem: ET.Element) -> Optional[Polygon]:
        """폴리곤 엘리먼트 파싱"""
        try:
            label = polygon_elem.get('label', '')
            if label not in self.surface_class_to_idx:
                logger.warning(f"알 수 없는 surface 클래스: {label}")
                return None
            
            # 폴리곤 포인트 파싱
            points_str = polygon_elem.get('points', '')
            if not points_str:
                logger.warning(f"폴리곤 포인트 없음: {label}")
                return None
                
            points = []
            for point_str in points_str.split(';'):
                if point_str.strip():
                    x, y = map(float, point_str.split(','))
                    points.append((x, y))
            
            if len(points) < 3:
                logger.warning(f"폴리곤 포인트 부족 (최소 3개 필요): {label}")
                return None
            
            # 속성 파싱
            attributes = {}
            for attr_elem in polygon_elem.findall('attribute'):
                attr_name = attr_elem.get('name', '')
                attr_value = attr_elem.text or ''
                if attr_name and attr_value:
                    attributes[attr_name] = attr_value
            
            return Polygon(
                label=label,
                points=points,
                occluded=polygon_elem.get('occluded', '0') == '1',
                z_order=int(polygon_elem.get('z_order', 0)),
                attributes=attributes
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"폴리곤 파싱 오류: {e}")
            return None
    
    def parse_all_folders(self) -> Dict[str, List[ImageAnnotation]]:
        """모든 폴더의 XML 파일을 파싱"""
        all_annotations = {}
        
        # BBox 폴더들 파싱
        for folder in self.bbox_folders:
            xml_files = list(folder.glob("*.xml"))
            if not xml_files:
                logger.warning(f"XML 파일 없음: {folder}")
                continue
            
            folder_annotations = []
            for xml_file in xml_files:
                annotations = self.parse_cvat_xml(xml_file)
                folder_annotations.extend(annotations)
            
            if folder_annotations:
                all_annotations[f"bbox_{folder.name}"] = folder_annotations
        
        # Surface 폴더들 파싱
        for folder in self.surface_folders:
            xml_files = list(folder.glob("*.xml"))
            if not xml_files:
                logger.warning(f"XML 파일 없음: {folder}")
                continue
            
            folder_annotations = []
            for xml_file in xml_files:
                annotations = self.parse_cvat_xml(xml_file)
                folder_annotations.extend(annotations)
            
            if folder_annotations:
                all_annotations[f"surface_{folder.name}"] = folder_annotations
        
        return all_annotations
    
    def generate_statistics(self, all_annotations: Dict[str, List[ImageAnnotation]]) -> Dict[str, Any]:
        """어노테이션 통계 생성"""
        stats = {
            'total_folders': len(all_annotations),
            'total_images': 0,
            'total_bboxes': 0,
            'total_polygons': 0,
            'bbox_class_counts': Counter(),
            'surface_class_counts': Counter(),
            'surface_attribute_counts': defaultdict(Counter),
            'folder_stats': {}
        }
        
        for folder_key, annotations in all_annotations.items():
            folder_stats = {
                'images': len(annotations),
                'bboxes': 0,
                'polygons': 0,
                'bbox_classes': Counter(),
                'surface_classes': Counter()
            }
            
            for annotation in annotations:
                # BBox 통계
                folder_stats['bboxes'] += len(annotation.bboxes)
                for bbox in annotation.bboxes:
                    folder_stats['bbox_classes'][bbox.label] += 1
                    stats['bbox_class_counts'][bbox.label] += 1
                
                # Polygon 통계
                folder_stats['polygons'] += len(annotation.polygons)
                for polygon in annotation.polygons:
                    folder_stats['surface_classes'][polygon.label] += 1
                    stats['surface_class_counts'][polygon.label] += 1
                    
                    # 속성 통계
                    for attr_name, attr_value in polygon.attributes.items():
                        stats['surface_attribute_counts'][polygon.label][attr_value] += 1
            
            stats['folder_stats'][folder_key] = folder_stats
            stats['total_images'] += folder_stats['images']
            stats['total_bboxes'] += folder_stats['bboxes']
            stats['total_polygons'] += folder_stats['polygons']
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]) -> None:
        """통계 정보 출력"""
        print("\n" + "="*80)
        print("📊 SafeTrip 어노테이션 통계")
        print("="*80)
        
        print(f"📁 전체 폴더: {stats['total_folders']}개")
        print(f"🖼️  전체 이미지: {stats['total_images']}개")
        print(f"📦 전체 바운딩박스: {stats['total_bboxes']}개")
        print(f"🔍 전체 폴리곤: {stats['total_polygons']}개")
        
        # BBox 클래스 분포
        bbox_counts = cast(Counter, stats['bbox_class_counts'])
        if bbox_counts:
            print(f"\n📦 바운딩박스 클래스 분포 (총 {len(bbox_counts)}개 클래스):")
            for cls, count in bbox_counts.most_common():
                print(f"  {cls:20s}: {count:5d}개")
        
        # Surface 클래스 분포
        surface_counts = cast(Counter, stats['surface_class_counts'])
        surface_attr_counts = cast(Dict, stats['surface_attribute_counts'])
        if surface_counts:
            print(f"\n🔍 표면 클래스 분포 (총 {len(surface_counts)}개 클래스):")
            for cls, count in surface_counts.most_common():
                print(f"  {cls:20s}: {count:5d}개")
                
                # 속성 분포
                if cls in surface_attr_counts:
                    attr_counts = cast(Counter, surface_attr_counts[cls])
                    if attr_counts:
                        attr_str = ", ".join([f"{attr}({cnt})" for attr, cnt in attr_counts.most_common()])
                        print(f"    └─ 속성: {attr_str}")
        
        # 폴더별 상세 통계
        print(f"\n📁 폴더별 상세 통계:")
        for folder_key, folder_stats in stats['folder_stats'].items():
            print(f"  {folder_key}:")
            print(f"    이미지: {folder_stats['images']}개, " +
                  f"BBox: {folder_stats['bboxes']}개, " +
                  f"Polygon: {folder_stats['polygons']}개")


def main():
    """테스트 및 데모 실행"""
    # 데이터 루트 경로 설정
    data_root = Path("data")
    
    if not data_root.exists():
        logger.error(f"데이터 루트 폴더 없음: {data_root}")
        return
    
    # 파서 초기화
    parser = SafeTripAnnotationParser(data_root)
    
    # 모든 폴더 파싱
    logger.info("🚀 모든 폴더 파싱 시작...")
    all_annotations = parser.parse_all_folders()
    
    if not all_annotations:
        logger.warning("파싱된 어노테이션이 없습니다.")
        return
    
    # 통계 생성 및 출력
    stats = parser.generate_statistics(all_annotations)
    parser.print_statistics(stats)
    
    # 샘플 데이터 출력
    print("\n" + "="*80)
    print("📋 샘플 어노테이션 정보")
    print("="*80)
    
    for folder_key, annotations in list(all_annotations.items())[:2]:  # 처음 2개 폴더만
        if annotations:
            sample_annotation = annotations[0]
            print(f"\n📁 {folder_key} - {sample_annotation.filename}:")
            print(f"  크기: {sample_annotation.width}x{sample_annotation.height}")
            print(f"  BBox: {len(sample_annotation.bboxes)}개")
            print(f"  Polygon: {len(sample_annotation.polygons)}개")
            
            if sample_annotation.bboxes:
                bbox = sample_annotation.bboxes[0]
                print(f"  샘플 BBox: {bbox.label} ({bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f})")
            
            if sample_annotation.polygons:
                polygon = sample_annotation.polygons[0]
                print(f"  샘플 Polygon: {polygon.label} (포인트 {len(polygon.points)}개)")
                if polygon.attributes:
                    attrs = ", ".join([f"{k}={v}" for k, v in polygon.attributes.items()])
                    print(f"    속성: {attrs}")


if __name__ == "__main__":
    main() 