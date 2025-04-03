import time
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import uuid
import numpy as np
import threading
import os
import json

# Import the global variables
from globals import log_queue, stop_training, current_batch_size

# Add this flag to track if model statistics have been shown
model_stats_shown = False

# Base callback with error handling
class BaseCallback(Callback):
    """
    Base callback class with common functionality for all callbacks:
    - Thread-safe UI updates
    - Error handling and logging
    - Safe property access
    
    All custom callbacks should inherit from this class for consistent behavior.
    """
    def __init__(self, verbose=1):
        super(BaseCallback, self).__init__()
        self.verbose = verbose
        
    def _safe_get_property(self, obj, prop_name, default=None):
        """Safely get a property from an object with error handling"""
        try:
            if hasattr(obj, prop_name):
                return getattr(obj, prop_name)
        except Exception as e:
            self._log_error(f"Error accessing property '{prop_name}': {str(e)}")
        return default
        
    def _safe_ui_update(self, root, update_func, *args, **kwargs):
        """Safely update UI elements with error handling"""
        if root:
            try:
                root.after(0, lambda: update_func(*args, **kwargs))
            except Exception as e:
                self._log_error(f"Error updating UI: {str(e)}")
    
    def _log_message(self, message):
        """Log a message to the queue"""
        if self.verbose > 0:
            log_queue.put(message)
            
    def _log_error(self, error_message):
        """Log an error message to the queue"""
        log_queue.put(f"ERROR: {error_message}")
    
    def _get_current_lr(self):
        """Safely get current learning rate from optimizer"""
        try:
            if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                if hasattr(self.model.optimizer, 'lr'):
                    return float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception as e:
            self._log_error(f"Error getting learning rate: {str(e)}")
        return None
        
    def _check_stop_training(self):
        """Check if training should be stopped and update model"""
        if stop_training:
            self.model.stop_training = True
            return True
        return False

# Custom callback for batch size change
class BatchSizeChangeCallback(BaseCallback):
    def __init__(self, dataset_gen_func):
        super(BatchSizeChangeCallback, self).__init__()
        self.dataset_gen_func = dataset_gen_func
        self.last_batch_size = current_batch_size
        
    def on_epoch_end(self, epoch, logs=None):
        global current_batch_size
        
        # Check if batch size has changed
        if self.last_batch_size != current_batch_size:
            self._log_message(f"Batch size changed from {self.last_batch_size} to {current_batch_size}")
            
            # Create a new dataset with the updated batch size
            try:
                new_dataset = self.dataset_gen_func(current_batch_size)
                
                # Store the new dataset for the next fit call
                self._data = new_dataset
                
                # Notify user about the change
                self._log_message(f"Dataset updated with new batch size: {current_batch_size}")
            except Exception as e:
                self._log_error(f"Error creating dataset with new batch size: {str(e)}")
            
        # Save the current batch size for the next epoch
        self.last_batch_size = current_batch_size
        
    def on_epoch_begin(self, epoch, logs=None):
        # Apply the new dataset at the beginning of an epoch if it exists
        if hasattr(self, '_data'):
            try:
                self.model._dataset = self._data
                self._log_message(f"Applying new batch size {current_batch_size} for epoch {epoch+1}")
                delattr(self, '_data')  # Clear the stored dataset after applying
            except Exception as e:
                self._log_error(f"Error applying new dataset: {str(e)}")

# Custom logger callback
class LoggerCallback(BaseCallback):
    def __init__(self, log_frequency=10, verbose=1):
        super(LoggerCallback, self).__init__(verbose)
        self.log_frequency = log_frequency  # Log every N batches
        self.batch_times = []
        self.start_time = None
        self.last_loss = 0.0  # Add a field to store the last loss value
        
    def on_train_begin(self, logs=None):
        self._log_message("Training started")
        
    def on_batch_begin(self, batch, logs=None):
        self.start_time = time.time()
        
    def on_batch_end(self, batch, logs=None):
        if self._check_stop_training():
            return
            
        # Record batch time
        if self.start_time:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)
            
            # Keep only the last 50 batch times for moving average
            if len(self.batch_times) > 50:
                self.batch_times.pop(0)
        
        # Get loss value safely
        try:
            loss_val = logs.get('loss', None)
            if loss_val is not None:
                # Convert tensor to numpy if needed
                if isinstance(loss_val, tf.Tensor):
                    loss_val = loss_val.numpy()
                
                # Convert to float
                self.last_loss = float(loss_val)
        except Exception as e:
            # Just log the error but continue
            if self.verbose > 1:
                self._log_error(f"Error extracting loss: {str(e)}")
    
        if batch % self.log_frequency == 0:  # Log every N batches
            try:
                # Calculate average batch time
                avg_time = np.mean(self.batch_times) if self.batch_times else 0
                
                # Format message with loss and timing info
                message = f"Step {batch} - Loss: {self.last_loss:.6f}"
                if avg_time > 0:
                    message += f" ({avg_time:.3f}s/batch)"
                    
                self._log_message(message)
            except Exception as e:
                self._log_error(f"Error in logger callback: {str(e)}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Log a summary at the end of each epoch
        try:
            # Get loss value safely
            loss_val = self._safe_get_property(logs, 'loss', None)
            if isinstance(loss_val, tf.Tensor):
                loss_val = loss_val.numpy()
            
            if loss_val is not None:
                self.last_loss = float(loss_val)
                
            # Log epoch summary with current loss
            self._log_message(f"Epoch {epoch+1} completed - Final loss: {self.last_loss:.6f}")
        except Exception as e:
            self._log_error(f"Error in epoch end logging: {str(e)}")

# Summary stats callback for model monitoring
class ModelStatsCallback(BaseCallback):
    def __init__(self, log_frequency=10, verbose=1):
        super(ModelStatsCallback, self).__init__(verbose)
        self.last_log_epoch = -log_frequency  # Force first log at epoch 0
        self.log_frequency = log_frequency  # Log every N epochs
        
    def on_train_begin(self, logs=None):
        # Initial stats logging
        global model_stats_shown
        
        # Only show stats if they haven't been shown yet
        if not model_stats_shown:
            try:
                trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
                non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
                total_params = trainable_params + non_trainable_params
                
                self._log_message(f"Model summary:")
                self._log_message(f"  Trainable parameters: {trainable_params:,}")
                self._log_message(f"  Non-trainable parameters: {non_trainable_params:,}")
                self._log_message(f"  Total parameters: {total_params:,}")
                
                # Save initial model architecture to JSON if possible
                try:
                    model_config = self.model.to_json()
                    os.makedirs('./Logs', exist_ok=True)
                    
                    with open(f'./Logs/model_architecture_{int(time.time())}.json', 'w') as f:
                        f.write(model_config)
                except Exception as e:
                    self._log_error(f"Could not save model architecture: {str(e)}")
                
                # Mark that stats have been shown
                model_stats_shown = True
                    
            except Exception as e:
                self._log_error(f"Error in model stats initialization: {str(e)}")
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch - self.last_log_epoch >= self.log_frequency:  # Every N epochs
            try:
                # Calculate and log model metrics
                self._log_message(f"Model metrics at epoch {epoch}:")
                
                # Get current learning rate
                current_lr = self._get_current_lr()
                if current_lr:
                    self._log_message(f"  Current learning rate: {current_lr:.6f}")
                
                # Update last log epoch
                self.last_log_epoch = epoch
                    
            except Exception as e:
                self._log_error(f"Error in model stats callback: {str(e)}")
    
    # Removed on_train_end method to prevent final statistics

# Progress callback to update the progress bar, percentage and ETA
class ProgressCallback(BaseCallback):
    def __init__(self, root, total_epochs, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.root = root
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
        self.last_update_time = 0
        self.update_interval = 0.5  # Update UI at most every 0.5 seconds
        
    def on_epoch_begin(self, epoch, logs=None):
        if not self.root:
            return
            
        try:
            # Only update UI if enough time has passed since last update
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return
                
            self.last_update_time = current_time
            
            # Get progress value
            progress_value = (epoch) / self.total_epochs * 100
            
            # Direct use of progress_var if passed via root
            if hasattr(self.root, 'progress_var') and self.root.progress_var:
                self._safe_ui_update(self.root, self.root.progress_var.set, progress_value)
            
            # Direct use of UI elements if passed through
            if hasattr(self.root, 'percentage_var') and self.root.percentage_var:
                self._safe_ui_update(self.root, self.root.percentage_var.set, f"{progress_value:.1f}%")
                
            if epoch > 0 and hasattr(self.root, 'eta_var') and self.root.eta_var:
                # Calculate average epoch time
                if len(self.epoch_times) > 0:
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    remaining_epochs = self.total_epochs - (epoch)
                    estimated_seconds = avg_epoch_time * remaining_epochs
                    
                    # Format ETA as hours:minutes:seconds
                    eta_str = str(datetime.timedelta(seconds=int(estimated_seconds)))
                    self._safe_ui_update(self.root, self.root.eta_var.set, f"ETA: {eta_str}")
        except Exception as e:
            self._log_error(f"Error updating progress: {str(e)}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Record epoch time for ETA calculation
        if epoch == 0:
            self.start_time = time.time()  # Reset start time after first epoch
        else:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            
            # Keep only the last 5 epochs for more accurate recent timing
            if len(self.epoch_times) > 5:
                self.epoch_times.pop(0)
                
            self.start_time = time.time()  # Reset for next epoch
            
        # Update progress again at end of epoch
        if self.root:
            # Calculate progress percentage
            progress_value = (epoch + 1) / self.total_epochs * 100
            
            # Update UI elements
            if hasattr(self.root, 'progress_var') and self.root.progress_var:
                self._safe_ui_update(self.root, self.root.progress_var.set, progress_value)
            if hasattr(self.root, 'percentage_var') and self.root.percentage_var:
                self._safe_ui_update(self.root, self.root.percentage_var.set, f"{progress_value:.1f}%")
            
    def on_train_end(self, logs=None):
        # Ensure progress bar is at 100% when training completes
        if self.root and hasattr(self.root, 'progress_var') and self.root.progress_var:
            self._safe_ui_update(self.root, self.root.progress_var.set, 100.0)
        
        self._log_message("Training finished!")
