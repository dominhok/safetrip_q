import numpy as np
import os
import json
from utils.xml_parser import CVATXMLParser


def create_colormap(num_classes=24):
    """
    Generate colormap for visualization of 24 classes.
    Uses distinct colors for better visualization.
    """
    # Create a colormap with distinct colors
    np.random.seed(42)  # For reproducibility
    
    # Define some manually chosen distinct colors for important classes
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    
    # Surface classes (0-5) - Use earth tones
    colormap[0] = [128, 128, 128]  # sidewalk - gray
    colormap[1] = [255, 255, 0]    # braille_guide_blocks - yellow
    colormap[2] = [64, 64, 64]     # roadway - dark gray
    colormap[3] = [96, 96, 96]     # alley - medium gray
    colormap[4] = [0, 128, 0]      # bike_lane - green
    colormap[5] = [255, 128, 0]    # caution_zone - orange
    
    # Moving objects (6-13) - Warm colors
    colormap[6] = [255, 0, 0]      # person - red
    colormap[7] = [0, 0, 255]      # car - blue
    colormap[8] = [0, 128, 255]    # bus - light blue
    colormap[9] = [128, 0, 255]    # truck - purple
    colormap[10] = [255, 0, 128]   # bicycle - pink
    colormap[11] = [128, 255, 0]   # motorcycle - lime
    colormap[12] = [255, 128, 128] # stroller - light red
    colormap[13] = [128, 255, 255] # scooter - cyan
    
    # Fixed objects (14-23) - Cool colors
    colormap[14] = [0, 64, 0]      # tree_trunk - dark green
    colormap[15] = [0, 255, 128]   # potted_plant - mint
    colormap[16] = [192, 192, 192] # pole - silver
    colormap[17] = [139, 69, 19]   # bench - brown
    colormap[18] = [255, 165, 0]   # bollard - orange
    colormap[19] = [255, 0, 255]   # barricade - magenta
    colormap[20] = [255, 69, 0]    # fire_hydrant - red-orange
    colormap[21] = [75, 0, 130]    # kiosk - indigo
    colormap[22] = [128, 0, 128]   # power_controller - purple
    colormap[23] = [0, 255, 255]   # traffic_light_controller - cyan
        
    return colormap


def create_class_info():
    """Create class information including names and IDs."""
    class_names = []
    
    # Surface classes (0-5)
    surface_classes = CVATXMLParser.get_surface_classes()
    class_names.extend(surface_classes)
    
    # Polygon classes (6-34)
    polygon_classes = CVATXMLParser.get_polygon_classes()
    class_names.extend(polygon_classes)
    
    # Create class info dictionary
    class_info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'class_to_id': {name: idx for idx, name in enumerate(class_names)},
        'id_to_class': {idx: name for idx, name in enumerate(class_names)},
        'surface_classes': list(range(len(surface_classes))),
        'polygon_classes': list(range(len(surface_classes), len(class_names)))
    }
    
    return class_info


def calculate_class_weights(data_root):
    """
    Calculate class weights for 24 classes.
    Uses pre-calculated weights based on actual frequency.
    """
    # Import the weight calculation function
    from calculate_class_weights_24 import calculate_class_weights_from_data
    
    # Calculate weights from actual data
    weights, _ = calculate_class_weights_from_data(data_root)
    
    return weights


def main():
    """Prepare all necessary data files for SafeTrip-Q training."""
    
    # Create output directory
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create and save colormap
    print("Creating colormap...")
    colormap = create_colormap(24)  # 24 classes now
    np.save(os.path.join(output_dir, 'cmap_safetrip_24.npy'), colormap)
    print(f"Saved colormap to {output_dir}/cmap_safetrip_24.npy")
    
    # 2. Create and save class information
    print("\nCreating class information...")
    class_info = create_class_info()
    with open(os.path.join(output_dir, 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"Saved class info to {output_dir}/class_info.json")
    
    # 3. Calculate and save class weights
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(output_dir)
    np.save(os.path.join(output_dir, 'class_weights_24.npy'), class_weights)
    print(f"Saved class weights to {output_dir}/class_weights_24.npy")
    
    # 4. Print summary
    print("\n" + "="*50)
    print("SafeTrip-Q Data Preparation Complete!")
    print("="*50)
    print(f"Total classes: {class_info['num_classes']}")
    print(f"Surface classes: {len(class_info['surface_classes'])}")
    print(f"Polygon classes: {len(class_info['polygon_classes'])}")
    print("\nClass names:")
    for idx, name in enumerate(class_info['class_names']):
        print(f"  {idx:2d}: {name}")
        

if __name__ == "__main__":
    main()