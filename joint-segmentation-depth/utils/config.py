import yaml
import os


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    
    # Update paths in config
    if 'general' in config:
        if 'input_dir' in config['general']:
            config['general']['input_dir'] = os.path.join(base_dir, config['general']['input_dir'])
        if 'output_dir' in config['general']:
            config['general']['output_dir'] = os.path.join(base_dir, config['general']['output_dir'])
        if 'cmap' in config['general']:
            config['general']['cmap'] = os.path.join(base_dir, config['general']['cmap'])
    
    if 'dataset' in config:
        if 'root' in config['dataset']:
            config['dataset']['root'] = os.path.join(base_dir, config['dataset']['root'])
        if 'class_weights_path' in config['dataset']:
            config['dataset']['class_weights_path'] = os.path.join(base_dir, config['dataset']['class_weights_path'])
        if 'class_info_path' in config['dataset']:
            config['dataset']['class_info_path'] = os.path.join(base_dir, config['dataset']['class_info_path'])
    
    if 'training' in config:
        if 'checkpoint_dir' in config['training']:
            config['training']['checkpoint_dir'] = os.path.join(base_dir, config['training']['checkpoint_dir'])
        if 'tensorboard_dir' in config['training']:
            config['training']['tensorboard_dir'] = os.path.join(base_dir, config['training']['tensorboard_dir'])
    
    return config