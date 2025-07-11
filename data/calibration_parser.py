"""
SafeTrip 스테레오 캘리브레이션 파서

새로운 데이터 구조:
- data/depth/Depth_xxx/ : 각 폴더마다 스테레오 이미지들 + Depth_xxx.conf 파일

지원 기능:
- 멀티폴더 자동 스캔 및 파싱
- 스테레오 카메라 캘리브레이션 파라미터 추출
- 다양한 해상도 지원 (2K, FHD, HD, VGA)
- Disparity-to-depth 변환
- 스테레오 정류 파라미터
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
from collections import defaultdict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터 데이터클래스"""
    fx: float  # focal length in x
    fy: float  # focal length in y
    cx: float  # principal point x
    cy: float  # principal point y
    k1: float = 0.0  # radial distortion coefficient 1
    k2: float = 0.0  # radial distortion coefficient 2
    k3: float = 0.0  # radial distortion coefficient 3
    p1: float = 0.0  # tangential distortion coefficient 1
    p2: float = 0.0  # tangential distortion coefficient 2
    
    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 카메라 행렬"""
        return np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,   1.0]
        ])
    
    @property
    def distortion_coeffs(self) -> np.ndarray:
        """왜곡 계수 벡터"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class StereoParameters:
    """스테레오 카메라 파라미터 데이터클래스"""
    baseline: float  # 스테레오 베이스라인 (mm)
    left_camera: CameraIntrinsics
    right_camera: CameraIntrinsics
    
    # 스테레오 정류 파라미터 (선택적)
    rotation_matrix: Optional[np.ndarray] = None  # 3x3 회전 행렬
    translation_vector: Optional[np.ndarray] = None  # 3x1 변위 벡터
    essential_matrix: Optional[np.ndarray] = None  # 3x3 필수 행렬
    fundamental_matrix: Optional[np.ndarray] = None  # 3x3 기본 행렬
    
    def disparity_to_depth(self, disparity: float) -> float:
        """Disparity 값을 실제 거리(meter)로 변환"""
        if disparity <= 0:
            return float('inf')
        # depth = (fx * baseline) / disparity
        focal_length = self.left_camera.fx
        return (focal_length * self.baseline / 1000.0) / disparity  # mm to m
    
    def depth_to_disparity(self, depth_m: float) -> float:
        """실제 거리(meter)를 disparity 값으로 변환"""
        if depth_m <= 0:
            return 0.0
        focal_length = self.left_camera.fx
        return (focal_length * self.baseline / 1000.0) / depth_m


@dataclass
class CalibrationData:
    """캘리브레이션 데이터 컨테이너"""
    folder_name: str
    config_file: Path
    stereo_params: Dict[str, StereoParameters] = field(default_factory=dict)  # resolution -> params
    image_pairs: List[Tuple[Path, Path]] = field(default_factory=list)  # (left, right) paths
    
    @property
    def available_resolutions(self) -> List[str]:
        """사용 가능한 해상도 목록"""
        return list(self.stereo_params.keys())
    
    def get_stereo_params(self, resolution: str = "2K") -> Optional[StereoParameters]:
        """특정 해상도의 스테레오 파라미터 반환"""
        return self.stereo_params.get(resolution)


class SafeTripCalibrationParser:
    """SafeTrip 프로젝트 스테레오 캘리브레이션 파서"""
    
    # 지원하는 해상도 및 기본값
    RESOLUTIONS = {
        "2K": (2208, 1242),
        "FHD": (1920, 1080), 
        "HD": (1280, 720),
        "VGA": (672, 376)
    }
    
    def __init__(self, data_root: Union[str, Path]):
        """
        Args:
            data_root: 데이터 루트 경로 (data/ 폴더)
        """
        self.data_root = Path(data_root)
        self.depth_folders = list((self.data_root / "depth").glob("Depth_*")) if (self.data_root / "depth").exists() else []
        
        # logger.info(f"Depth 폴더 {len(self.depth_folders)}개 발견")  # 로그 간소화
    
    def parse_config_file(self, config_path: Path) -> Dict[str, StereoParameters]:
        """
        .conf 파일에서 스테레오 캘리브레이션 파라미터 파싱
        
        Args:
            config_path: .conf 파일 경로
            
        Returns:
            해상도별 스테레오 파라미터 딕셔너리
        """
        try:
            config = configparser.ConfigParser()
            config.read(config_path, encoding='utf-8')
            
            stereo_params = {}
            
            for resolution in self.RESOLUTIONS.keys():
                try:
                    params = self._parse_resolution_params(config, resolution)
                    if params:
                        stereo_params[resolution] = params
                        logger.debug(f"{config_path.name}: {resolution} 파라미터 파싱 완료")
                except Exception as e:
                    logger.warning(f"{config_path.name}: {resolution} 파라미터 파싱 실패 - {e}")
                    continue
            
            # logger.info(f"{config_path.name}: {len(stereo_params)}개 해상도 파라미터 파싱 완료")  # 로그 간소화
            return stereo_params
            
        except Exception as e:
            logger.error(f"Config 파일 파싱 오류 {config_path}: {e}")
            return {}
    
    def _parse_resolution_params(self, config: configparser.ConfigParser, resolution: str) -> Optional[StereoParameters]:
        """특정 해상도의 파라미터 파싱"""
        try:
            # 베이스라인 (공통)
            baseline = float(config.get('STEREO', 'Baseline'))
            
            # 좌측 카메라 파라미터
            left_section = f'LEFT_CAM_{resolution}'
            if not config.has_section(left_section):
                return None
                
            left_camera = CameraIntrinsics(
                fx=float(config.get(left_section, 'fx')),
                fy=float(config.get(left_section, 'fy')),
                cx=float(config.get(left_section, 'cx')),
                cy=float(config.get(left_section, 'cy')),
                k1=float(config.get(left_section, 'k1', fallback=0.0)),
                k2=float(config.get(left_section, 'k2', fallback=0.0)),
                k3=float(config.get(left_section, 'k3', fallback=0.0)),
                p1=float(config.get(left_section, 'p1', fallback=0.0)),
                p2=float(config.get(left_section, 'p2', fallback=0.0))
            )
            
            # 우측 카메라 파라미터
            right_section = f'RIGHT_CAM_{resolution}'
            if not config.has_section(right_section):
                return None
                
            right_camera = CameraIntrinsics(
                fx=float(config.get(right_section, 'fx')),
                fy=float(config.get(right_section, 'fy')),
                cx=float(config.get(right_section, 'cx')),
                cy=float(config.get(right_section, 'cy')),
                k1=float(config.get(right_section, 'k1', fallback=0.0)),
                k2=float(config.get(right_section, 'k2', fallback=0.0)),
                k3=float(config.get(right_section, 'k3', fallback=0.0)),
                p1=float(config.get(right_section, 'p1', fallback=0.0)),
                p2=float(config.get(right_section, 'p2', fallback=0.0))
            )
            
            return StereoParameters(
                baseline=baseline,
                left_camera=left_camera,
                right_camera=right_camera
            )
            
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            logger.warning(f"파라미터 파싱 오류 {resolution}: {e}")
            return None
    
    def parse_image_pairs(self, folder_path: Path) -> List[Tuple[Path, Path]]:
        """
        폴더에서 좌측 이미지들 찾아 depth 이미지 쌍 생성
        
        Args:
            folder_path: Depth_xxx 폴더 경로
            
        Returns:
            (left_image, disparity_path) 경로 쌍의 리스트
        """
        image_pairs = []
        
        # 모든 PNG 파일 찾기
        png_files = list(folder_path.glob("*.png"))
        
        # 좌측 RGB 이미지만 필터링 (_L.png 또는 _left.png이고 confidence/disp 제외)
        left_images = [
            f for f in png_files 
            if ('_L.png' in str(f) or '_left.png' in str(f)) 
            and 'confidence' not in str(f) 
            and 'disp' not in str(f)
        ]
        
        # 각 좌측 이미지에 대해 대응하는 disparity 맵 찾기
        for left_img in left_images:
            img_name = left_img.stem  # 확장자 제거
            
            # 파일명에서 식별자 추출 (예: ZED1_KSC_001032_L → ZED1_KSC_001032)
            if img_name.endswith('_L'):
                base_name = img_name[:-2]
            elif img_name.endswith('_left'):
                base_name = img_name[:-5]
            else:
                base_name = img_name
            
            # 대응하는 disparity 맵 찾기
            possible_disp_paths = [
                folder_path / f"{base_name}_disp.png",
                folder_path / f"{base_name}_disp16.png",
            ]
            
            disparity_path = None
            for disp_path in possible_disp_paths:
                if disp_path.exists():
                    disparity_path = disp_path
                    break
            
            if disparity_path:
                # 좌측 이미지와 disparity 맵 쌍으로 추가
                image_pairs.append((left_img, disparity_path))
            else:
                # disparity 맵이 없어도 좌측 이미지는 추가 (depth 없는 샘플로 처리)
                logger.debug(f"Disparity 맵 없음: {base_name}")
                image_pairs.append((left_img, left_img))  # fallback
        
        # logger.info(f"{folder_path.name}: {len(image_pairs)}개 스테레오 이미지 쌍 발견")  # 로그 간소화
        return image_pairs
    
    def parse_single_folder(self, folder_path: Path) -> Optional[CalibrationData]:
        """단일 Depth 폴더 파싱"""
        try:
            folder_name = folder_path.name
            
            # 설정 파일 찾기
            config_files = list(folder_path.glob(f"{folder_name}.conf"))
            if not config_files:
                logger.warning(f"설정 파일 없음: {folder_path}")
                return None
            
            config_file = config_files[0]
            
            # 캘리브레이션 파라미터 파싱
            stereo_params = self.parse_config_file(config_file)
            if not stereo_params:
                logger.warning(f"캘리브레이션 파라미터 파싱 실패: {folder_path}")
                return None
            
            # 이미지 쌍 파싱
            image_pairs = self.parse_image_pairs(folder_path)
            
            return CalibrationData(
                folder_name=folder_name,
                config_file=config_file,
                stereo_params=stereo_params,
                image_pairs=image_pairs
            )
            
        except Exception as e:
            logger.error(f"폴더 파싱 오류 {folder_path}: {e}")
            return None
    
    def parse_all_folders(self) -> Dict[str, CalibrationData]:
        """모든 Depth 폴더 파싱"""
        all_calibrations = {}
        
        for folder in self.depth_folders:
            calibration_data = self.parse_single_folder(folder)
            if calibration_data:
                all_calibrations[folder.name] = calibration_data
        
        # logger.info(f"총 {len(all_calibrations)}개 폴더 캘리브레이션 데이터 파싱 완료")  # 로그 간소화
        return all_calibrations
    
    def generate_statistics(self, all_calibrations: Dict[str, CalibrationData]) -> Dict[str, Any]:
        """캘리브레이션 통계 생성"""
        stats = {
            'total_folders': len(all_calibrations),
            'total_image_pairs': 0,
            'resolution_coverage': defaultdict(int),
            'baseline_stats': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0.0,
                'values': []
            },
            'folder_stats': {}
        }
        
        baseline_values = []
        
        for folder_name, calib_data in all_calibrations.items():
            folder_stats = {
                'image_pairs': len(calib_data.image_pairs),
                'resolutions': calib_data.available_resolutions,
                'baseline': None
            }
            
            # 해상도 커버리지
            for resolution in calib_data.available_resolutions:
                stats['resolution_coverage'][resolution] += 1
                
                # 베이스라인 통계 (첫 번째 해상도에서만)
                if folder_stats['baseline'] is None:
                    stereo_params = calib_data.get_stereo_params(resolution)
                    if stereo_params:
                        baseline = stereo_params.baseline
                        folder_stats['baseline'] = baseline
                        baseline_values.append(baseline)
                        stats['baseline_stats']['min'] = min(stats['baseline_stats']['min'], baseline)
                        stats['baseline_stats']['max'] = max(stats['baseline_stats']['max'], baseline)
            
            stats['folder_stats'][folder_name] = folder_stats
            stats['total_image_pairs'] += folder_stats['image_pairs']
        
        # 베이스라인 평균
        if baseline_values:
            stats['baseline_stats']['mean'] = np.mean(baseline_values)
            stats['baseline_stats']['values'] = baseline_values
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]) -> None:
        """통계 정보 출력"""
        print("\n" + "="*80)
        print("📊 SafeTrip 스테레오 캘리브레이션 통계")
        print("="*80)
        
        print(f"📁 전체 폴더: {stats['total_folders']}개")
        print(f"🖼️  전체 스테레오 이미지 쌍: {stats['total_image_pairs']}개")
        
        # 해상도 커버리지
        print(f"\n🔍 해상도 커버리지:")
        for resolution, count in stats['resolution_coverage'].items():
            width, height = self.RESOLUTIONS[resolution]
            print(f"  {resolution:4s} ({width}x{height}): {count:3d}개 폴더")
        
        # 베이스라인 통계
        baseline_stats = stats['baseline_stats']
        if baseline_stats['values']:
            print(f"\n📏 베이스라인 통계:")
            print(f"  최소: {baseline_stats['min']:.3f}mm")
            print(f"  최대: {baseline_stats['max']:.3f}mm")
            print(f"  평균: {baseline_stats['mean']:.3f}mm")
        
        # 폴더별 요약 (처음 10개만)
        print(f"\n📁 폴더별 요약 (처음 10개):")
        for folder_name, folder_stats in list(stats['folder_stats'].items())[:10]:
            resolutions_str = ", ".join(folder_stats['resolutions'])
            baseline_str = f"{folder_stats['baseline']:.1f}mm" if folder_stats['baseline'] else "N/A"
            print(f"  {folder_name}: {folder_stats['image_pairs']}쌍, "
                  f"{resolutions_str}, baseline={baseline_str}")


def main():
    """테스트 및 데모 실행"""
    # 데이터 루트 경로 설정
    data_root = Path("data")
    
    if not data_root.exists():
        logger.error(f"데이터 루트 폴더 없음: {data_root}")
        return
    
    # 파서 초기화
    parser = SafeTripCalibrationParser(data_root)
    
    if not parser.depth_folders:
        logger.warning("Depth 폴더가 없습니다.")
        return
    
    # 모든 폴더 파싱
    logger.info("🚀 모든 Depth 폴더 파싱 시작...")
    all_calibrations = parser.parse_all_folders()
    
    if not all_calibrations:
        logger.warning("파싱된 캘리브레이션 데이터가 없습니다.")
        return
    
    # 통계 생성 및 출력
    stats = parser.generate_statistics(all_calibrations)
    parser.print_statistics(stats)
    
    # 샘플 데이터 출력
    print("\n" + "="*80)
    print("📋 샘플 캘리브레이션 정보")
    print("="*80)
    
    # 첫 번째 폴더의 세부 정보
    if all_calibrations:
        first_folder, first_data = next(iter(all_calibrations.items()))
        print(f"\n📁 {first_folder}:")
        print(f"  이미지 쌍: {len(first_data.image_pairs)}개")
        print(f"  지원 해상도: {', '.join(first_data.available_resolutions)}")
        
        # 2K 해상도 파라미터 (가능한 경우)
        stereo_params = first_data.get_stereo_params("2K")
        if stereo_params:
            print(f"\n  📸 2K 스테레오 파라미터:")
            print(f"    베이스라인: {stereo_params.baseline:.3f}mm")
            print(f"    좌측 카메라 - fx: {stereo_params.left_camera.fx:.1f}, fy: {stereo_params.left_camera.fy:.1f}")
            print(f"    좌측 카메라 - cx: {stereo_params.left_camera.cx:.1f}, cy: {stereo_params.left_camera.cy:.1f}")
            
            # Depth 변환 예시
            test_disparities = [1.0, 10.0, 50.0, 100.0]
            print(f"\n  🔄 Disparity → Depth 변환 예시:")
            for disp in test_disparities:
                depth = stereo_params.disparity_to_depth(disp)
                print(f"    Disparity {disp:4.1f} → {depth:6.2f}m")
        
        # 첫 번째 이미지 쌍 정보
        if first_data.image_pairs:
            left_img, right_img = first_data.image_pairs[0]
            print(f"\n  📷 첫 번째 이미지 쌍:")
            print(f"    좌측: {left_img.name}")
            print(f"    우측: {right_img.name}")


if __name__ == "__main__":
    main() 