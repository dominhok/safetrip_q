import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class DepthConverter:
    """Convert disparity images to depth maps using calibration parameters."""
    
    def __init__(self):
        """Initialize depth converter."""
        pass
    
    def parse_calibration(self, calib_path: str) -> Dict[str, float]:
        """
        Parse calibration file (.conf) for depth conversion parameters.
        Each Depth_XXX folder has its own unique calibration.
        
        Args:
            calib_path: Path to calibration file (e.g., Depth_001/Depth_001.conf)
            
        Returns:
            Dictionary with calibration parameters
        """
        calib_data = {}
        
        try:
            with open(calib_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse numeric values
                        if key in ['BaseLine', 'fx', 'fy', 'cx', 'cy']:
                            calib_data[key] = float(value)
                            
            # Default calibration values from documentation if missing
            if 'BaseLine' not in calib_data:
                calib_data['BaseLine'] = 119.975  # mm
            if 'fx' not in calib_data:
                calib_data['fx'] = 1400.15
            if 'fy' not in calib_data:
                calib_data['fy'] = 1400.15
                
        except Exception as e:
            print(f"Error parsing calibration file {calib_path}: {e}")
            # Return default values
            calib_data = {
                'BaseLine': 119.975,
                'fx': 1400.15,
                'fy': 1400.15,
                'cx': 943.093,
                'cy': 559.187
            }
            
        return calib_data
    
    def disparity_to_depth(self, 
                          disparity_path: str,
                          calibration: Dict[str, float],
                          confidence_path: Optional[str] = None,
                          max_depth: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert disparity image to depth map using folder-specific calibration.
        
        Args:
            disparity_path: Path to disparity16 PNG file
            calibration: Calibration parameters dictionary (unique per folder)
            confidence_path: Optional path to confidence mask
            max_depth: Maximum depth value in meters (for clipping)
            
        Returns:
            Tuple of (depth_map, valid_mask) where:
            - depth_map: Depth in meters, shape (H, W)
            - valid_mask: Boolean mask of valid depth pixels
        """
        # Read disparity image (16-bit PNG)
        disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
        
        if disparity is None:
            raise ValueError(f"Could not read disparity image: {disparity_path}")
            
        # Convert to float for computation
        disparity = disparity.astype(np.float32)
        
        # GA-Net uses 16-bit PNG with scale factor
        # Typical scale factor is 256 (disparity = png_value / 256)
        disparity = disparity / 256.0
        
        # Get calibration parameters (unique for this folder)
        baseline = calibration['BaseLine'] / 1000.0  # Convert mm to meters
        focal_length = calibration['fx']  # Assume fx = fy for simplicity
        
        # Compute depth: depth = baseline * focal_length / disparity
        # Avoid division by zero
        valid_disp = disparity > 0.1  # Minimum disparity threshold
        depth = np.zeros_like(disparity)
        depth[valid_disp] = (baseline * focal_length) / disparity[valid_disp]
        
        # Clip to reasonable depth range
        depth = np.clip(depth, 0, max_depth)
        
        # Load confidence mask if provided
        if confidence_path and os.path.exists(confidence_path):
            confidence = cv2.imread(confidence_path, cv2.IMREAD_GRAYSCALE)
            if confidence is not None:
                # Binary confidence: 255 = confident, 0 = not confident
                valid_mask = (confidence > 128) & valid_disp
            else:
                valid_mask = valid_disp
        else:
            valid_mask = valid_disp
            
        return depth, valid_mask
    
    def get_depth_image_sets(self, depth_folder: str) -> Dict[str, Dict[str, str]]:
        """
        Get all image sets in a Depth folder.
        Each Depth_XXX folder contains multiple sets of 8 images.
        
        Args:
            depth_folder: Path to Depth_XXX folder
            
        Returns:
            Dictionary mapping base names to file paths for each set
        """
        image_sets = {}
        files = os.listdir(depth_folder)
        
        # Group files by their base name (before _L, _R, etc.)
        for f in files:
            if f.endswith('.conf'):
                continue
                
            # Extract base name (everything before the suffix)
            if '_L.png' in f:
                base_name = f.replace('_L.png', '')
                suffix = '_L.png'
            elif '_R.png' in f:
                base_name = f.replace('_R.png', '')
                suffix = '_R.png'
            elif '_left.png' in f:
                base_name = f.replace('_left.png', '')
                suffix = '_left.png'
            elif '_right.png' in f:
                base_name = f.replace('_right.png', '')
                suffix = '_right.png'
            elif '_disp16.png' in f:
                base_name = f.replace('_disp16.png', '')
                suffix = '_disp16.png'
            elif '_disp.png' in f:
                base_name = f.replace('_disp.png', '')
                suffix = '_disp.png'
            elif '_confidence_save.png' in f:
                base_name = f.replace('_confidence_save.png', '')
                suffix = '_confidence_save.png'
            elif '_confidence.png' in f:
                base_name = f.replace('_confidence.png', '')
                suffix = '_confidence.png'
            else:
                continue
                
            if base_name not in image_sets:
                image_sets[base_name] = {}
                
            image_sets[base_name][suffix] = os.path.join(depth_folder, f)
            
        return image_sets
    
    def load_depth_sample(self, 
                         depth_folder: str, 
                         base_name: str,
                         calibration: Dict[str, float]) -> Optional[Dict]:
        """
        Load a single depth sample with its associated files.
        
        Args:
            depth_folder: Path to Depth_XXX folder
            base_name: Base name of the image set
            calibration: Pre-loaded calibration for this folder
            
        Returns:
            Dictionary containing depth sample data
        """
        try:
            image_sets = self.get_depth_image_sets(depth_folder)
            
            if base_name not in image_sets:
                print(f"Base name {base_name} not found in {depth_folder}")
                return None
                
            image_set = image_sets[base_name]
            
            # Check for required files
            if '_L.png' not in image_set or '_disp16.png' not in image_set:
                print(f"Missing required files for {base_name}")
                return None
                
            # Load RGB image (Raw_Left)
            rgb_left = cv2.imread(image_set['_L.png'])
            if rgb_left is None:
                print(f"Could not read RGB image: {image_set['_L.png']}")
                return None
                
            # Convert BGR to RGB
            rgb_left = cv2.cvtColor(rgb_left, cv2.COLOR_BGR2RGB)
            
            # Get confidence path if exists
            confidence_path = image_set.get('_confidence.png', None)
            
            # Convert disparity to depth
            depth_map, valid_mask = self.disparity_to_depth(
                image_set['_disp16.png'], 
                calibration, 
                confidence_path
            )
            
            return {
                'rgb_image': rgb_left,
                'depth_map': depth_map,
                'valid_mask': valid_mask,
                'image_path': image_set['_L.png'],
                'base_name': base_name
            }
            
        except Exception as e:
            print(f"Error loading depth sample {base_name}: {e}")
            return None