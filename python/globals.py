# globals.py
import queue
import os

# Global queue for thread-safe logging
log_queue = queue.Queue()

# Global stop flag for training control
stop_training = False

# Global training parameters with updated defaults
current_batch_size = 12  # Changed from 8 to 12
default_epochs = 100     # Changed from previous value to 100
default_steps_per_epoch = 50
default_learning_rate = 0.01
default_image_size = 256

# Create only the Checkpoints directory
os.makedirs('./Checkpoints', exist_ok=True)

# Extension of supported image formats
supported_image_formats = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff')

# Available loss functions and their weights/parameters
loss_configs = {
    'mse': {},
    'perceptual': {},
    'combined': {
        'alpha': 0.8  # Weight between MSE and MAE (higher alpha means more MSE)
    }
}

# Advanced parameters
advanced_params = {
    'swa_start_pct': 0.75,     # SWA starts at this percentage of training
    'swa_freq': 1,             # SWA model update frequency (epochs)
    'swa_lr_factor': 0.1,      # SWA learning rate factor (multiplied by max_lr)
    'onecycle_warmup_pct': 0.3, # OneCycle warmup percentage
    'final_div_factor': 100,   # OneCycle final learning rate divisor
    'weight_decay': 0.01,      # L2 regularization strength - set default for AdamW
    'checkpoint_freq': 5       # Save checkpoint every N epochs
}

def get_parameter(name, default=None):
    """Get a parameter value with fallback to default"""
    if name in globals():
        return globals()[name]
    for param_dict in [advanced_params, loss_configs]:
        if name in param_dict:
            return param_dict[name]
    return default

def set_parameter(name, value):
    """Set a parameter value"""
    if name in globals():
        globals()[name] = value
        return True
            
    for param_dict in [advanced_params, loss_configs]:
        if name in param_dict:
            param_dict[name] = value
            return True
                
    return False