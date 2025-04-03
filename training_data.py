import os
import numpy as np
import tensorflow as tf
import time
import json
from PIL import Image, ImageEnhance, ImageOps
import uuid
import random

# Import the global variables
from globals import log_queue, default_image_size

def create_training_dataset(dataset_folder, batch_size, target_size=default_image_size):
    """
    Creates a TensorFlow dataset for training the fast perceptual model.
    
    Args:
        dataset_folder: Path to folder containing image files
        batch_size: Batch size for training
        target_size: Target image size (default from globals)
        
    Returns:
        A prefetched TensorFlow dataset
    """
    # Generate a unique name for the dataset
    unique_name = f"dataset_{uuid.uuid4().hex}"
    
    # Get all image files from the dataset folder
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) 
                  if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    
    if len(image_files) == 0:
        log_queue.put(f"Error: No image files found in {dataset_folder}")
        raise ValueError(f"No image files found in {dataset_folder}")
    
    log_queue.put(f"Found {len(image_files)} images for training")
    
    # Random image rotation using TF operations
    def random_rotation(image):
        # Random angle between -45 and 45 degrees
        angle = tf.random.uniform([], -45, 45, dtype=tf.float32)
        # Convert to radians
        angle_rad = angle * np.pi / 180
        # Perform rotation
        return tf.image.rot90(image, k=tf.cast(angle_rad / (np.pi/2), tf.int32))
    
    # Extract a random patch from an image
    def extract_random_patch(image):
        # Get image dimensions
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        
        # Size for extraction (larger to allow for rotation)
        extraction_size = tf.cast(target_size * 1.5, tf.int32)
        
        # If image is too small, resize it
        if tf.logical_or(tf.less(height, extraction_size), tf.less(width, extraction_size)):
            # Determine scale factor to make the smaller dimension match extraction_size
            scale = tf.cast(extraction_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
            new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
            # Resize the image
            image = tf.image.resize(image, [new_height, new_width])
            shape = tf.shape(image)
            height, width = shape[0], shape[1]
        
        # Calculate valid range for patch extraction
        h_start_max = height - extraction_size
        w_start_max = width - extraction_size
        
        # Get random start positions
        h_start = tf.random.uniform([], 0, h_start_max + 1, dtype=tf.int32)
        w_start = tf.random.uniform([], 0, w_start_max + 1, dtype=tf.int32)
        
        # Extract the patch
        patch = tf.image.crop_to_bounding_box(image, h_start, w_start, extraction_size, extraction_size)
        
        # Final center crop to target size
        return tf.image.resize_with_crop_or_pad(patch, target_size, target_size)
    
    # Apply augmentation to image
    def augment_image(image):
        # Random brightness adjustment
        image = tf.image.random_brightness(image, 0.2)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random saturation adjustment
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Random hue adjustment
        image = tf.image.random_hue(image, 0.1)
        
        # Random flips
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Ensure values are within [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=min(len(image_files), 1000), 
                              reshuffle_each_iteration=True)
    
    # Load and preprocess images
    def load_and_preprocess_image(file_path):
        try:
            # Read image
            image_data = tf.io.read_file(file_path)
            image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
            
            # Extract a random patch
            patch = extract_random_patch(image)
            
            # Apply augmentations
            augmented = augment_image(patch)
            
            return augmented
        except tf.errors.InvalidArgumentError:
            # Return a blank image if there's an error
            log_queue.put(f"Error loading image, returning blank")
            return tf.zeros([target_size, target_size, 3], dtype=tf.float32)
    
    # Map preprocessing function over the dataset
    dataset = dataset.map(load_and_preprocess_image, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add MixUp augmentation
    def apply_mixup(images):
        # Create pairs of images for mixup
        image1 = images
        # Shuffled version for the second image
        image2 = tf.random.shuffle(images)
        
        # Random mixup factor
        alpha = 0.2
        lam = tf.random.uniform([], alpha, 1.0 - alpha)
        
        # Apply mixup
        mixed_images = lam * image1 + (1.0 - lam) * image2
        
        return mixed_images
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Apply mixup occasionally
    dataset = dataset.map(lambda x: tf.cond(
        tf.random.uniform([], 0, 1) < 0.3,  # 30% chance
        lambda: apply_mixup(x),
        lambda: x
    ), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add repeat() to prevent "ran out of data" warning
    dataset = dataset.repeat()
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Function to process samples through VGG model
def process_samples_for_vgg(data_batch, vgg_model):
    """
    Process a batch of images through VGG model to get features.
    
    Args:
        data_batch: Batch of input images
        vgg_model: VGG model to extract features
        
    Returns:
        Tuple of (input_images, vgg_features)
    """
    # Pass the batch through VGG model to get features
    vgg_features = vgg_model(data_batch)
    # Return input images and their VGG features as targets
    return data_batch, vgg_features

# Dynamic batch size callback for dataset updates
class DynamicBatchSizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset_folder, vgg_model):
        super(DynamicBatchSizeCallback, self).__init__()
        self.dataset_folder = dataset_folder
        self.vgg_model = vgg_model
        self.last_batch_size = None
        
    def on_epoch_begin(self, epoch, logs=None):
        from globals import current_batch_size
        
        # Initialize last_batch_size on first call
        if self.last_batch_size is None:
            self.last_batch_size = current_batch_size
            return
            
        # Check if batch size has changed
        if self.last_batch_size != current_batch_size:
            log_queue.put(f"Batch size changed from {self.last_batch_size} to {current_batch_size}")
            
            try:
                # Create new dataset with updated batch size
                raw_dataset = create_training_dataset(self.dataset_folder, current_batch_size)
                
                # Map the VGG feature extraction
                new_dataset = raw_dataset.map(
                    lambda x: process_samples_for_vgg(x, self.vgg_model),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                
                # Store for future use
                self._new_dataset = new_dataset
                log_queue.put(f"New dataset created with batch size {current_batch_size}")
                
            except Exception as e:
                log_queue.put(f"Error updating dataset: {str(e)}")
            
            self.last_batch_size = current_batch_size
            
    def on_epoch_end(self, epoch, logs=None):
        # Update visualization callback if needed
        if hasattr(self, '_new_dataset') and hasattr(self.model, 'callbacks'):
            for callback in self.model.callbacks:
                if hasattr(callback, 'dataset'):
                    callback.dataset = self._new_dataset
                    log_queue.put("Updated visualization callback with new dataset")

# Create a more complete training module with continued training
class ContinuedTrainingHandler:
    """
    Handler for continued training with configuration management.
    Allows saving and loading training configuration and picking up training from where it left off.
    """
    def __init__(self, config_dir='./Logs'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
    def save_training_config(self, config):
        """Save training configuration to a JSON file"""
        # Generate unique filename with timestamp
        timestamp = int(time.time())
        config_file = os.path.join(self.config_dir, f'training_config_{timestamp}.json')
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
        return config_file
    
    def load_latest_config(self):
        """Load the latest training configuration"""
        config_files = [f for f in os.listdir(self.config_dir) if f.startswith('training_config_') and f.endswith('.json')]
        
        if not config_files:
            return None
            
        # Get the latest config file
        latest_config = max(config_files, key=lambda f: int(f.split('_')[2].split('.')[0]))
        config_path = os.path.join(self.config_dir, latest_config)
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            log_queue.put(f"Error loading config: {e}")
            return None
    
    def resume_training(self, model=None, checkpoint_dir='./Checkpoints'):
        """Resume training from the latest checkpoint if available"""
        # Get the latest config
        config = self.load_latest_config()
        
        if not config:
            log_queue.put("No previous training configuration found")
            return None, 0
            
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('fast_perceptual_loss_epoch_')] if os.path.exists(checkpoint_dir) else []
        
        if not checkpoint_files:
            log_queue.put("No checkpoints found for continued training")
            return None, 0
            
        # Extract epoch numbers
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                if '_' in f.split('epoch_')[1]:
                    epoch_num = int(f.split('epoch_')[1].split('_')[0])
                else:
                    epoch_num = int(f.split('epoch_')[1].split('.')[0])
                epoch_numbers.append(epoch_num)
            except (IndexError, ValueError):
                continue
                
        if not epoch_numbers:
            log_queue.put("No valid checkpoints found")
            return None, 0
            
        # Get the latest checkpoint
        latest_epoch = max(epoch_numbers)
        matching_files = [f for f in checkpoint_files if f'epoch_{latest_epoch:02d}' in f]
        
        if not matching_files:
            log_queue.put("No matching checkpoint file found")
            return None, 0
            
        latest_checkpoint = os.path.join(checkpoint_dir, matching_files[0])
        
        log_queue.put(f"Found checkpoint at epoch {latest_epoch}: {latest_checkpoint}")
        
        # Return the config and epoch number
        return config, latest_epoch

# Extended data augmentation with more sophisticated techniques
class AdvancedAugmentation:
    """
    Advanced data augmentation techniques implemented using TensorFlow operations.
    """
    @staticmethod
    def cutmix(images, alpha=1.0):
        """
        CutMix data augmentation: cuts a region from one image and pastes it onto another.
        
        Args:
            images: Batch of images [batch_size, height, width, channels]
            alpha: Hyperparameter controlling the strength of augmentation
            
        Returns:
            Mixed images
        """
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Generate random parameters for the cut rectangle
        lam = tf.random.beta(alpha, alpha, [])
        
        # Calculate cut size
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(height * cut_ratio, tf.int32)
        cut_w = tf.cast(width * cut_ratio, tf.int32)
        
        # Calculate center position
        cx = tf.random.uniform([], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, height, dtype=tf.int32)
        
        # Calculate box boundaries (clip to image edges)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        
        # Create a mask for the cut region
        mask = tf.pad(
            tf.ones((y2 - y1, x2 - x1), dtype=tf.float32),
            [[y1, height - y2], [x1, width - x2]]
        )
        mask = tf.reshape(mask, [height, width, 1])
        mask = tf.tile(mask, [1, 1, 3])  # Repeat for RGB channels
        
        # Generate shuffled indices
        shuffled_indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, shuffled_indices)
        
        # Apply CutMix
        cut_images = images * (1 - mask) + shuffled_images * mask
        
        return cut_images
    
    @staticmethod
    def random_erasing(images, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3):
        """
        Random Erasing: randomly erases rectangular regions from images.
        
        Args:
            images: Batch of images [batch_size, height, width, channels]
            probability: Probability of applying erasing to each image
            sl, sh: Min/max relative area of erased region
            r1, r2: Min/max aspect ratio of erased region
            
        Returns:
            Erased images
        """
        def erase_single_image(image):
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            channels = tf.shape(image)[2]
            
            # Calculate area
            area = tf.cast(height * width, tf.float32)
            
            # Decide whether to apply random erasing
            do_erase = tf.random.uniform([], 0, 1) < probability
            
            def erased_image():
                # Choose target area
                target_area = tf.random.uniform([], sl, sh) * area
                
                # Choose aspect ratio
                aspect_ratio = tf.random.uniform([], r1, r2)
                
                # Calculate height and width
                h = tf.cast(tf.sqrt(target_area / aspect_ratio), tf.int32)
                w = tf.cast(tf.sqrt(target_area * aspect_ratio), tf.int32)
                
                # Make sure h, w are not too large
                h = tf.minimum(h, height)
                w = tf.minimum(w, width)
                
                # Random position
                x = tf.random.uniform([], 0, width - w + 1, dtype=tf.int32)
                y = tf.random.uniform([], 0, height - h + 1, dtype=tf.int32)
                
                # Generate random noise
                noise = tf.random.uniform([h, w, channels], 0, 1)
                
                # Create update mask
                update_mask = tf.pad(
                    tf.ones([h, w], dtype=tf.float32),
                    [[y, height - y - h], [x, width - x - w]]
                )
                update_mask = tf.reshape(update_mask, [height, width, 1])
                update_mask = tf.tile(update_mask, [1, 1, channels])
                
                # Create noise padded to image size
                padded_noise = tf.pad(
                    noise,
                    [[y, height - y - h], [x, width - x - w], [0, 0]]
                )
                
                # Apply the erasing
                return image * (1 - update_mask) + padded_noise * update_mask
            
            return tf.cond(do_erase, erased_image, lambda: image)
        
        # Process batch of images
        return tf.map_fn(erase_single_image, images)