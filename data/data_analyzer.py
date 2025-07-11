"""
SafeTrip 데이터 분석 도구

각 태스크별로 데이터를 검증하고 시각화합니다.
- BBox 객체 탐지 데이터 분석
- Surface 인스턴스 세그멘테이션 데이터 분석
- Depth 스테레오 깊이 추정 데이터 분석
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import random
from typing import Dict, List

# 상위 디렉토리를 Python path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from data.annotation_parser import SafeTripAnnotationParser
from data.calibration_parser import SafeTripCalibrationParser


def get_bright_colors(num_colors):
    """밝고 눈에 띄는 색상들을 생성"""
    bright_colors = [
        '#FF3838',  # 밝은 빨강
        '#FFD700',  # 골드
        '#32CD32',  # 라임 그린
        '#FF6347',  # 토마토
        '#1E90FF',  # 도지블루
        '#FF69B4',  # 핫 핑크
        '#00CED1',  # 다크 터쿼이즈
        '#FFA500',  # 오렌지
        '#9370DB',  # 미디움 퍼플
        '#7FFF00',  # 차트리우스
        '#FF1493',  # 딥 핑크
        '#00FFFF',  # 시안
        '#FFB6C1',  # 라이트 핑크
        '#98FB98',  # 페일 그린
        '#F0E68C',  # 카키
    ]
    
    # 필요한 만큼 반복하거나 추가 생성
    if num_colors <= len(bright_colors):
        return bright_colors[:num_colors]
    
    # 더 많은 색상이 필요한 경우 HSV로 추가 생성
    additional_colors = []
    for i in range(num_colors - len(bright_colors)):
        hue = (i * 360 / (num_colors - len(bright_colors))) % 360
        # 밝고 채도 높은 색상 생성
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        additional_colors.append(hex_color)
    
    return bright_colors + additional_colors


def visualize_bbox_sample(folder: Path, annotations: List, sample_idx: int = 0):
    """BBox 샘플 시각화 (밝은 색상 적용)"""
    if sample_idx >= len(annotations):
        print(f"    ⚠️ 샘플 인덱스 {sample_idx}가 범위를 벗어남")
        return
    
    annotation = annotations[sample_idx]
    image_path = folder / annotation.filename
    
    if not image_path.exists():
        print(f"    ⚠️ 이미지 파일 없음: {image_path}")
        return
    
    try:
        # 이미지 로드
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"    ❌ 이미지 로드 실패: {image_path}")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 시각화 설정
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"BBox Sample: {annotation.filename}", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 해당 이미지의 어노테이션 찾기
        target_annotation = None
        for ann in annotations:
            if ann.filename == annotation.filename:
                target_annotation = ann
                break
        
        if target_annotation and target_annotation.bboxes:
            # 모든 클래스 수집
            all_classes = list(set(bbox.label for bbox in target_annotation.bboxes))
            colors = get_bright_colors(len(all_classes))
            class_to_color = {cls: colors[i] for i, cls in enumerate(all_classes)}
            
            # BBox 그리기
            for bbox in target_annotation.bboxes:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                width = x2 - x1
                height = y2 - y1
                
                color = class_to_color[bbox.label]
                
                # 사각형 그리기 (더 두꺼운 선)
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=3,  # 더 두꺼운 선
                    edgecolor=color,
                    facecolor='none'
                )
                plt.gca().add_patch(rect)
                
                # 라벨 텍스트 (배경과 함께)
                plt.text(
                    x1, y1 - 5,
                    bbox.label,
                    fontsize=12,  # 더 큰 폰트
                    fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
                )
        
        # 저장
        visualize_dir = Path("visualize/bbox")
        visualize_dir.mkdir(parents=True, exist_ok=True)
        save_path = visualize_dir / f"bbox_sample_{sample_idx}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    💾 시각화 저장: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"    ❌ BBox 시각화 오류: {e}")


def visualize_surface_sample(folder: Path, annotations: List, sample_idx: int = 0):
    """Surface 샘플 시각화"""
    if sample_idx >= len(annotations):
        print(f"    ⚠️ 샘플 인덱스 {sample_idx}가 범위를 벗어남")
        return
    
    annotation = annotations[sample_idx]
    image_path = folder / annotation.filename
    
    if not image_path.exists():
        print(f"    ⚠️ 이미지 파일 없음: {image_path}")
        return
    
    try:
        # 이미지 로드
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"    ❌ 이미지 로드 실패: {image_path}")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 시각화 설정
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"Surface Sample: {annotation.filename}", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 해당 이미지의 어노테이션 찾기
        target_annotation = None
        for ann in annotations:
            if ann.filename == annotation.filename:
                target_annotation = ann
                break
        
        if target_annotation and target_annotation.polygons:
            # Surface 클래스별 색상 매핑 (기존 색상 유지)
            surface_colors = {
                'sidewalk': 'red',
                'roadway': 'green', 
                'bike_lane': 'blue',
                'crosswalk': 'yellow',
                'parking': 'purple',
                'alley': 'orange'
            }
            
            # Polygon 그리기
            for poly in target_annotation.polygons:
                if poly.points and len(poly.points) >= 3:
                    # 좌표 배열 생성
                    points = np.array(poly.points, dtype=np.int32)
                    
                    # 색상 선택
                    color = surface_colors.get(poly.label, 'cyan')
                    
                    # Polygon 그리기 (반투명)
                    polygon = patches.Polygon(
                        points, 
                        closed=True, 
                        alpha=0.4, 
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2
                    )
                    plt.gca().add_patch(polygon)
                    
                    # 라벨 텍스트 (중심점에)
                    center_x = float(np.mean(points[:, 0]))
                    center_y = float(np.mean(points[:, 1]))
                    plt.text(
                        center_x, center_y,
                        poly.label,
                        fontsize=10,
                        fontweight='bold',
                        color='white',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
                    )
        
        # 저장
        visualize_dir = Path("visualize/surface")
        visualize_dir.mkdir(parents=True, exist_ok=True)
        save_path = visualize_dir / f"surface_sample_{sample_idx}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    💾 시각화 저장: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Surface 시각화 오류: {e}")


def visualize_depth_sample(folder: Path, sample_idx: int = 0):
    """Depth 샘플 시각화 (좌측 RGB 이미지 + GT disparity 맵)"""
    
    # 좌측 RGB 이미지 파일들 찾기 (_L.png 또는 _left.png)
    png_files = list(folder.glob("*.png"))
    left_images = [f for f in png_files if ('_L.png' in str(f) or '_left.png' in str(f)) and 'confidence' not in str(f) and 'disp' not in str(f)]
    
    if sample_idx >= len(left_images):
        print(f"    ⚠️ 좌측 이미지 샘플 인덱스 {sample_idx}가 범위를 벗어남 (총 {len(left_images)}개)")
        return
    
    if not left_images:
        print(f"    ❌ 좌측 RGB 이미지를 찾을 수 없음")
        return
    
    left_image_path = left_images[sample_idx]
    
    try:
        # 좌측 RGB 이미지 로드
        left_img = cv2.imread(str(left_image_path))
        if left_img is None:
            print(f"    ❌ 좌측 이미지 로드 실패: {left_image_path}")
            return
            
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        
        # 대응하는 disparity 맵 찾기
        img_name = left_image_path.stem  # 파일명 (확장자 제외)
        
        # 파일명에서 식별자 추출 (예: ZED1_KSC_001032_L → ZED1_KSC_001032)
        if img_name.endswith('_L'):
            base_name = img_name[:-2]
        elif img_name.endswith('_left'):
            base_name = img_name[:-5]
        else:
            base_name = img_name
        
        # 여러 가능한 disparity 파일명 패턴 시도
        possible_disp_names = [
            f"{base_name}_disp.png",
            f"{base_name}_disp16.png",
        ]
        
        disparity_path = None
        for disp_name in possible_disp_names:
            potential_path = folder / disp_name
            if potential_path.exists():
                disparity_path = potential_path
                break
        
        if disparity_path is None:
            print(f"    ❌ 대응하는 disparity 맵 없음: {base_name}")
            print(f"    🔍 시도한 파일명: {possible_disp_names}")
            return
        
        # Disparity 맵 로드
        disparity_img = cv2.imread(str(disparity_path), cv2.IMREAD_UNCHANGED)
        if disparity_img is None:
            print(f"    ❌ Disparity 맵 로드 실패: {disparity_path}")
            return
        
        # 시각화
        plt.figure(figsize=(15, 6))
        
        # 좌측 RGB 이미지
        plt.subplot(1, 3, 1)
        plt.imshow(left_img_rgb)
        plt.title(f"Left RGB Image: {left_image_path.name}", fontweight='bold')
        plt.axis('off')
        
        # Disparity 맵 (grayscale)
        plt.subplot(1, 3, 2)
        if len(disparity_img.shape) == 3:
            disparity_gray = cv2.cvtColor(disparity_img, cv2.COLOR_BGR2GRAY)
        else:
            disparity_gray = disparity_img
        plt.imshow(disparity_gray, cmap='gray')
        plt.title(f"GT Disparity: {disparity_path.name}", fontweight='bold')
        plt.colorbar()
        plt.axis('off')
        
        # Disparity 맵 (컬러맵)
        plt.subplot(1, 3, 3)
        plt.imshow(disparity_gray, cmap='viridis')
        plt.title("Disparity (Viridis)", fontweight='bold')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        
        # 저장
        visualize_dir = Path("visualize/depth")
        visualize_dir.mkdir(parents=True, exist_ok=True)
        save_path = visualize_dir / f"depth_sample_{sample_idx}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    💾 시각화 저장: {save_path}")
        print(f"    📸 학습 쌍: {left_image_path.name} → {disparity_path.name}")
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Depth 시각화 오류: {e}")


def check_bbox_data(data_root: Path, num_samples: int = 3):
    """BBox 객체 탐지 데이터 검증 및 시각화"""
    print("\n" + "="*60)
    print("📦 BBOX 객체 탐지 데이터 검증")
    print("="*60)
    
    parser = SafeTripAnnotationParser(data_root)
    
    # 폴더별 샘플 확인
    for i, folder in enumerate(parser.bbox_folders[:3]):
        print(f"\n📁 폴더: {folder.name}")
        
        # XML 파일 찾기
        xml_files = list(folder.glob("*.xml"))
        if not xml_files:
            print("  ❌ XML 파일 없음")
            continue
            
        # 첫 번째 XML 파일 파싱
        try:
            annotations = parser.parse_cvat_xml(xml_files[0])
            
            print(f"  ✅ 이미지 수: {len(annotations)}")
            
            # 총 bbox 수 계산
            total_bboxes = sum(len(ann.bboxes) for ann in annotations)
            print(f"  📝 총 bbox 수: {total_bboxes}")
            
            # 클래스 분포 확인
            all_classes = []
            for ann in annotations:
                for bbox in ann.bboxes:
                    all_classes.append(bbox.label)
            
            class_counts = {}
            for cls in all_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print(f"  🏷️ 클래스 분포: {dict(list(class_counts.items())[:5])}")
            
            # 시각화
            if annotations:
                print(f"  🎨 시각화 생성 중...")
                visualize_bbox_sample(folder, annotations, sample_idx=i)
            
        except Exception as e:
            print(f"  ❌ 파싱 오류: {e}")
    
    print(f"\n📊 전체 BBox 폴더 수: {len(parser.bbox_folders)}")


def check_surface_data(data_root: Path, num_samples: int = 3):
    """Surface 인스턴스 세그멘테이션 데이터 검증 및 시각화"""
    print("\n" + "="*60)  
    print("🛣️ SURFACE 인스턴스 세그멘테이션 데이터 검증")
    print("="*60)
    
    parser = SafeTripAnnotationParser(data_root)
    
    # 폴더별 샘플 확인
    for i, folder in enumerate(parser.surface_folders[:3]):
        print(f"\n📁 폴더: {folder.name}")
        
        # XML 파일 찾기
        xml_files = list(folder.glob("*.xml"))
        if not xml_files:
            print("  ❌ XML 파일 없음")
            continue
            
        try:
            annotations = parser.parse_cvat_xml(xml_files[0])
            
            print(f"  ✅ 이미지 수: {len(annotations)}")
            
            # 총 polygon 수 계산
            total_polygons = sum(len(ann.polygons) for ann in annotations)
            print(f"  🎯 총 polygon 수: {total_polygons}")
            
            # 클래스 분포 확인
            all_classes = []
            for ann in annotations:
                for poly in ann.polygons:
                    all_classes.append(poly.label)
            
            class_counts = {}
            for cls in all_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print(f"  🏷️ 클래스 분포: {dict(list(class_counts.items())[:5])}")
            
            # Polygon 포인트 수 확인
            if annotations and annotations[0].polygons:
                sample_poly = annotations[0].polygons[0]
                print(f"  📐 샘플 polygon 포인트 수: {len(sample_poly.points)}")
            
            # 시각화
            if annotations:
                print(f"  🎨 시각화 생성 중...")
                visualize_surface_sample(folder, annotations, sample_idx=i)
                    
        except Exception as e:
            print(f"  ❌ 파싱 오류: {e}")
    
    print(f"\n📊 전체 Surface 폴더 수: {len(parser.surface_folders)}")


def check_depth_data(data_root: Path, num_samples: int = 3):
    """Depth 스테레오 깊이 추정 데이터 검증 및 시각화"""
    print("\n" + "="*60)
    print("🎯 DEPTH 스테레오 깊이 추정 데이터 검증")  
    print("="*60)
    
    parser = SafeTripCalibrationParser(data_root)
    
    # 폴더별 샘플 확인
    for i, folder in enumerate(parser.depth_folders[:3]):
        print(f"\n📁 폴더: {folder.name}")
        
        # .conf 파일 찾기
        conf_files = list(folder.glob("*.conf"))
        if not conf_files:
            print("  ❌ .conf 파일 없음")
            continue
            
        try:
            calib_data = parser.parse_single_folder(folder)
            if calib_data:
                # 첫 번째 해상도의 파라미터로 baseline 확인
                if calib_data.stereo_params:
                    first_res = list(calib_data.stereo_params.keys())[0]
                    first_params = calib_data.stereo_params[first_res]
                    print(f"  📏 Baseline: {first_params.baseline:.3f}mm")
                
                print(f"  🔍 해상도: {len(calib_data.available_resolutions)}개")
                for res_name in calib_data.available_resolutions[:3]:
                    if res_name in parser.RESOLUTIONS:
                        width, height = parser.RESOLUTIONS[res_name]
                        print(f"    - {res_name}: {width}x{height}")
                
                # Depth 이미지 확인
                png_files = list(folder.glob("*.png"))
                print(f"  🖼️ Depth 이미지 수: {len(png_files)}")
                
                # 샘플 depth 이미지 통계
                if png_files:
                    sample_depth = cv2.imread(str(png_files[0]), cv2.IMREAD_UNCHANGED)
                    if sample_depth is not None:
                        print(f"  📊 Depth 통계: min={sample_depth.min()}, max={sample_depth.max()}, mean={sample_depth.mean():.1f}")
                
                # 시각화
                if png_files:
                    print(f"  🎨 시각화 생성 중...")
                    # 시각화
                    visualize_depth_sample(folder, sample_idx=i)
                        
        except Exception as e:
            print(f"  ❌ 캘리브레이션 파싱 오류: {e}")
    
    print(f"\n📊 전체 Depth 폴더 수: {len(parser.depth_folders)}")


def check_integrated_dataset(data_root: Path, num_samples: int = 5):
    """통합 멀티태스크 데이터셋 검증"""
    print("\n" + "="*60)
    print("🔗 통합 멀티태스크 데이터셋 검증")
    print("="*60)
    
    try:
        # 상위 디렉토리에서 dataset 모듈 import
        sys.path.append(str(data_root.parent))
        from data.dataset import SafeTripMultiTaskDataset
        
        dataset = SafeTripMultiTaskDataset(data_root, mode='train', max_samples=100)
        
        print(f"✅ 데이터셋 로드 성공")
        print(f"📊 총 샘플 수: {len(dataset)}")
        print(f"🏷️ BBox 클래스 수: {len(dataset.bbox_classes)}")
        print(f"🎯 Surface 클래스 수: {len(dataset.surface_classes)}")
        
        # 샘플 확인
        print(f"\n🔍 샘플 {num_samples}개 확인:")
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                print(f"\n  샘플 {i}:")
                print(f"    이미지 크기: {sample['image'].shape}")
                
                # BBox 라벨 확인
                if len(sample['targets']['bbox_labels']) > 0:
                    bbox_classes = [dataset.bbox_classes[idx] for idx in sample['targets']['bbox_labels']]
                    print(f"    BBox 클래스: {bbox_classes}")
                
                # Polygon 라벨 확인  
                if len(sample['targets']['polygon_labels']) > 0:
                    polygon_classes = [dataset.surface_classes[idx] for idx in sample['targets']['polygon_labels']]
                    print(f"    Polygon 클래스: {polygon_classes}")
                
                # Depth 확인
                if sample['targets']['depth'] is not None:
                    depth_map = sample['targets']['depth']
                    print(f"    Depth 맵: {depth_map.shape}, 범위: {depth_map.min():.1f}~{depth_map.max():.1f}")
                else:
                    print(f"    Depth 맵: 없음")
                    
            except Exception as e:
                print(f"    ❌ 샘플 {i} 로드 오류: {e}")
                
    except Exception as e:
        print(f"❌ 통합 데이터셋 로드 실패: {e}")


if __name__ == "__main__":
    # 데이터 루트 경로
    data_root = Path("C:/Users/user/Desktop/projects/safetrip/data")
    
    print("🎨 SafeTrip 데이터 분석 및 시각화 시작!")
    print("💾 시각화 결과는 visualize/bbox/, visualize/surface/, visualize/depth/ 폴더에 저장됩니다.")
    
    # 각 태스크별 데이터 확인
    check_bbox_data(data_root)
    check_surface_data(data_root)
    check_depth_data(data_root)
    check_integrated_dataset(data_root)
    
    print("\n�� 데이터 분석 및 시각화 완료!") 