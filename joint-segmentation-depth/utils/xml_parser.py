import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional
import os
import cv2


class CVATXMLParser:
    """Parse CVAT format XML annotations for SafeTrip-Q dataset."""
    
    def __init__(self, class_mapping: Dict[str, int]):
        """
        Initialize parser with class name to index mapping.
        
        Args:
            class_mapping: Dictionary mapping class names to indices
        """
        self.class_mapping = class_mapping
        
    def parse_xml(self, xml_path: str) -> Dict[str, np.ndarray]:
        """
        Parse CVAT XML file containing annotations for all images in a folder.
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            Dictionary mapping image names to segmentation masks
        """
        annotations = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find all images in the XML
            images = root.findall('.//image')
            if not images:
                return annotations
                
            # Process each image in the XML
            for image in images:
                # Get image name and dimensions
                img_name = image.get('name', '')
                if not img_name:
                    continue
                    
                xml_width = int(image.get('width', 1920))
                xml_height = int(image.get('height', 1080))
                
                # Create empty mask with ignore index (255)
                mask = np.full((xml_height, xml_width), 255, dtype=np.uint8)
                
                # Process all polygons in this image
                polygons = image.findall('.//polygon')
                
                # Sort polygons by z_order (higher z_order = on top)
                polygons_with_order = []
                for polygon in polygons:
                    z_order = int(polygon.get('z_order', 0))
                    polygons_with_order.append((z_order, polygon))
                
                # Sort by z_order (ascending, so we draw from bottom to top)
                polygons_with_order.sort(key=lambda x: x[0])
                
                # Draw polygons on mask
                for _, polygon in polygons_with_order:
                    label = polygon.get('label', '')
                    # Skip removed classes
                    if label in CVATXMLParser.get_removed_classes():
                        continue
                    if label not in self.class_mapping:
                        continue
                        
                    class_idx = self.class_mapping[label]
                    points_str = polygon.get('points', '')
                    
                    if not points_str:
                        continue
                        
                    # Parse points from "x1,y1;x2,y2;..." format
                    points = []
                    for point in points_str.split(';'):
                        if ',' in point:
                            x, y = point.split(',')
                            x = float(x)
                            y = float(y)
                            points.append((x, y))
                    
                    if len(points) >= 3:  # Need at least 3 points for a polygon
                        # Create PIL image for polygon drawing
                        pil_mask = Image.fromarray(mask)
                        draw = ImageDraw.Draw(pil_mask)
                        
                        # Draw filled polygon
                        draw.polygon(points, fill=class_idx)
                        
                        # Convert back to numpy
                        mask = np.array(pil_mask)
                
                # Store the mask for this image
                annotations[img_name] = mask
                        
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            
        return annotations
    
    def get_mask_for_image(self, 
                          annotations: Dict[str, np.ndarray], 
                          image_name: str,
                          target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Get segmentation mask for a specific image and optionally resize it.
        
        Args:
            annotations: Dictionary from parse_xml()
            image_name: Name of the image file
            target_shape: Optional (height, width) to resize mask to
            
        Returns:
            Segmentation mask or None if not found
        """
        if image_name not in annotations:
            # Try without path
            base_name = os.path.basename(image_name)
            if base_name not in annotations:
                return None
            image_name = base_name
            
        mask = annotations[image_name]
        
        # Resize if needed
        if target_shape and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
            
        return mask
    
    @staticmethod
    def get_surface_classes() -> List[str]:
        """Return list of Surface classes."""
        return [
            'sidewalk',
            'braille_guide_blocks', 
            'roadway',
            'alley',
            'bike_lane',
            'caution_zone'
        ]
    
    @staticmethod
    def get_polygon_classes() -> List[str]:
        """Return list of Polygon (obstacle) classes - filtered to actual obstacles only."""
        return [
            # Moving objects (8) - actual path blockers
            'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
            'stroller', 'scooter',
            # Fixed objects (10) - physical obstacles
            'tree_trunk', 'potted_plant', 'pole', 'bench', 
            'bollard', 'barricade', 'fire_hydrant', 'kiosk',
            'power_controller', 'traffic_light_controller'
        ]
        
    @staticmethod
    def get_removed_classes() -> List[str]:
        """Return list of removed classes that should be ignored."""
        return [
            # Rare classes (< 100 annotations)
            'parking_meter', 'cat', 'dog', 'wheelchair',
            # Non-obstacles (above head or not blocking path)
            'traffic_light', 'traffic_sign', 'stop', 'movable_signage',
            # Ambiguous
            'carrier', 'chair', 'table'
        ]
    
    @staticmethod
    def get_unified_class_mapping() -> Dict[str, int]:
        """
        Get unified class mapping for Surface + Polygon classes.
        
        Returns:
            Dictionary mapping class names to indices (0-34)
        """
        surface_classes = CVATXMLParser.get_surface_classes()
        polygon_classes = CVATXMLParser.get_polygon_classes()
        
        class_mapping = {}
        
        # Surface classes: 0-5
        for idx, cls in enumerate(surface_classes):
            class_mapping[cls] = idx
            
        # Polygon classes: 6-34
        for idx, cls in enumerate(polygon_classes):
            class_mapping[cls] = idx + len(surface_classes)
            
        return class_mapping