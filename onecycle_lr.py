import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import math
import numpy as np

# Import the global variables
from globals import log_queue


class OneCycleLR(Callback):
    """
    Enhanced Implementation of the One Cycle Learning Rate Policy.
    
    The 1cycle policy anneals the learning rate from an initial learning rate to a
    maximum learning rate and then back to an even lower final learning rate. This policy was
    proposed by Leslie Smith in the paper 'Super-Convergence: Very Fast Training of 
    Neural Networks Using Large Learning Rates'.
    
    This implementation adds:
    - Better cosine annealing with more controlled transitions
    - Weight decay decoupled from learning rate
    - Improved monitoring and logging
    
    Args:
        max_lr: Maximum learning rate.
        steps_per_epoch: Number of steps per epoch.
        epochs: Total number of epochs.
        min_lr: Minimum learning rate (starting). Default is max_lr/10.
        final_div_factor: Final LR will be max_lr/final_div_factor. Default is 100.
        warmup_pct: Percentage of total iterations for warmup phase. Default is 0.3.
        current_lr_var: Tkinter variable to update UI with current learning rate.
        momentum_range: Tuple of (max_momentum, min_momentum) to vary momentum inversely to lr.
        weight_decay: Weight decay factor. Set to None to disable. Default is None.
        verbose: Whether to print verbose output. Default is 1.
    """
    def __init__(self, max_lr, steps_per_epoch, epochs, min_lr=None, 
                 final_div_factor=100, warmup_pct=0.3, current_lr_var=None, 
                 momentum_range=(0.95, 0.85), weight_decay=None, verbose=1):
        super(OneCycleLR, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr / 10
        self.final_lr = max_lr / final_div_factor  # End with even lower LR
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs
        self.warmup_steps = int(self.total_steps * warmup_pct)
        self.annealing_steps = self.total_steps - self.warmup_steps
        self.current_step = 0
        self.current_lr_var = current_lr_var
        self.history = {'lr': [], 'momentum': [], 'weight_decay': []}
        self.initialized = False
        self.verbose = verbose
        
        # Momentum scheduling (inverse to learning rate)
        self.max_momentum, self.min_momentum = momentum_range
        
        # Weight decay scheduling
        self.weight_decay = weight_decay
        self.weight_decay_schedule = []
        if weight_decay is not None:
            # Create weight decay schedule - follow LR pattern but scaled
            self.initial_weight_decay = weight_decay
            self.max_weight_decay = weight_decay * 10  # Increase during warmup
            self.final_weight_decay = weight_decay / 10  # Lower at the end
        
    def _calculate_lr(self, step):
        """Calculate learning rate for current step according to schedule"""
        if step < self.warmup_steps:
            # Linear warmup phase
            progress = step / self.warmup_steps
            return self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            # Cosine annealing phase
            progress = (step - self.warmup_steps) / self.annealing_steps
            # Smoother cosine annealing with a slight offset to avoid sudden changes
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.final_lr + (self.max_lr - self.final_lr) * cosine_decay
    
    def _calculate_momentum(self, step):
        """Calculate momentum for current step - inverse to learning rate"""
        if step < self.warmup_steps:
            # Linear decrease during warmup
            progress = step / self.warmup_steps
            return self.max_momentum - (self.max_momentum - self.min_momentum) * progress
        else:
            # Linear increase during annealing
            progress = (step - self.warmup_steps) / self.annealing_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_momentum + (self.max_momentum - self.min_momentum) * cosine_decay
    
    def _calculate_weight_decay(self, step):
        """Calculate weight decay for current step if enabled"""
        if self.weight_decay is None:
            return None
            
        if step < self.warmup_steps:
            # Linear increase during warmup
            progress = step / self.warmup_steps
            return self.initial_weight_decay + (self.max_weight_decay - self.initial_weight_decay) * progress
        else:
            # Cosine annealing during cooldown
            progress = (step - self.warmup_steps) / self.annealing_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.final_weight_decay + (self.max_weight_decay - self.final_weight_decay) * cosine_decay
        
    def on_train_begin(self, logs=None):
        # Initialize with min_lr
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            log_queue.put("Warning: Optimizer not found in model. Cannot set learning rate.")
            return
            
        # Set initial learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
        
        # Initialize momentum
        if hasattr(self.model.optimizer, 'momentum'):
            tf.keras.backend.set_value(self.model.optimizer.momentum, self.max_momentum)
            
        # Setup weight decay for AdamW
        if self.weight_decay is not None:
            if hasattr(self.model.optimizer, 'decay'):
                self.supports_weight_decay = True
                tf.keras.backend.set_value(self.model.optimizer.decay, self.initial_weight_decay)
            elif hasattr(self.model.optimizer, 'weight_decay'):
                self.supports_weight_decay = True
                tf.keras.backend.set_value(self.model.optimizer.weight_decay, self.initial_weight_decay)
            else:
                self.supports_weight_decay = False
                if self.verbose > 0:
                    log_queue.put("Warning: Weight decay requested but optimizer doesn't support it.")
            
        # Update UI
        if self.current_lr_var:
            try:
                self.current_lr_var.set(f"Current LR: {self.min_lr:.6f}")
            except Exception as e:
                log_queue.put(f"Error updating LR display: {str(e)}")
                
        # Only log this information once on the first training run
        if not self.initialized and self.verbose > 0:
            log_queue.put(f"OneCycleLR: Warmup: {self.warmup_steps} steps, Annealing: {self.annealing_steps} steps")
            log_queue.put(f"OneCycleLR: LR range: [{self.min_lr:.6f}, {self.max_lr:.6f}, {self.final_lr:.6f}]")
            log_queue.put(f"OneCycleLR: Momentum range: [{self.min_momentum:.4f}, {self.max_momentum:.4f}]")
            if self.weight_decay is not None and self.supports_weight_decay:
                log_queue.put(f"OneCycleLR: Weight decay active: initial={self.initial_weight_decay}")
            self.initialized = True
    
    def on_epoch_begin(self, epoch, logs=None):
        # Log current settings at the beginning of each epoch
        if self.verbose > 0:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            
            momentum_str = ""
            if hasattr(self.model.optimizer, 'momentum'):
                current_momentum = float(tf.keras.backend.get_value(self.model.optimizer.momentum))
                momentum_str = f", momentum={current_momentum:.4f}"
                
            weight_decay_str = ""
            if self.weight_decay is not None and self.supports_weight_decay:
                if hasattr(self.model.optimizer, 'decay'):
                    current_wd = float(tf.keras.backend.get_value(self.model.optimizer.decay))
                else:
                    current_wd = float(tf.keras.backend.get_value(self.model.optimizer.weight_decay))
                weight_decay_str = f", weight_decay={current_wd:.6f}"
            
            log_queue.put(f"Epoch {epoch+1} starting with lr={current_lr:.6f}{momentum_str}{weight_decay_str}")
                
    def on_batch_begin(self, batch, logs=None):
        # Skip if we've gone beyond total steps
        if self.current_step >= self.total_steps:
            return
            
        # Calculate the current learning rate, momentum, and weight decay
        lr = self._calculate_lr(self.current_step)
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Set momentum if available
        if hasattr(self.model.optimizer, 'momentum'):
            momentum = self._calculate_momentum(self.current_step)
            tf.keras.backend.set_value(self.model.optimizer.momentum, momentum)
            self.history['momentum'].append(momentum)
        
        # Set weight decay if supported
        if self.weight_decay is not None and self.supports_weight_decay:
            weight_decay = self._calculate_weight_decay(self.current_step)
            if hasattr(self.model.optimizer, 'decay'):
                tf.keras.backend.set_value(self.model.optimizer.decay, weight_decay)
            else:
                tf.keras.backend.set_value(self.model.optimizer.weight_decay, weight_decay)
            self.history['weight_decay'].append(weight_decay)
            
        # Update UI variable if provided (every 5 batches to reduce overhead)
        if self.current_lr_var and (batch % 5 == 0 or batch == 0):
            try:
                self.current_lr_var.set(f"Current LR: {lr:.6f}")
            except Exception as e:
                log_queue.put(f"Error updating LR display: {str(e)}")
        
        # Store history for plotting
        self.history['lr'].append(lr)
        
        # Increment step counter
        self.current_step += 1
        
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            # Log current learning rate at the end of each epoch
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            
            # Create momentum message if applicable
            momentum_msg = ""
            if hasattr(self.model.optimizer, 'momentum'):
                current_momentum = float(tf.keras.backend.get_value(self.model.optimizer.momentum))
                momentum_msg = f" with momentum: {current_momentum:.4f}"
                
            # Create weight decay message if applicable
            weight_decay_msg = ""
            if self.weight_decay is not None and self.supports_weight_decay:
                if hasattr(self.model.optimizer, 'decay'):
                    current_wd = float(tf.keras.backend.get_value(self.model.optimizer.decay))
                else: 
                    current_wd = float(tf.keras.backend.get_value(self.model.optimizer.weight_decay))
                weight_decay_msg = f", weight_decay: {current_wd:.6f}"
                
            log_queue.put(f"Epoch {epoch+1} completed with lr: {current_lr:.6f}{momentum_msg}{weight_decay_msg}")
        
        # Update UI
        if self.current_lr_var:
            try:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                self.current_lr_var.set(f"Current LR: {current_lr:.6f}")
            except Exception as e:
                log_queue.put(f"Error updating LR display: {str(e)}")
                
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            log_queue.put("OneCycleLR schedule completed")
            
    def get_lr_schedule(self):
        """Returns the full learning rate schedule as a numpy array for plotting"""
        return np.array([self._calculate_lr(i) for i in range(self.total_steps)])
        
    def get_momentum_schedule(self):
        """Returns the full momentum schedule as a numpy array for plotting"""
        return np.array([self._calculate_momentum(i) for i in range(self.total_steps)])
        
    def get_weight_decay_schedule(self):
        """Returns the full weight decay schedule as a numpy array for plotting"""
        if self.weight_decay is None:
            return None
        return np.array([self._calculate_weight_decay(i) for i in range(self.total_steps)])


class SWA(Callback):
    """
    Stochastic Weight Averaging callback.
    
    Implements the Stochastic Weight Averaging (SWA) technique for improving
    generalization in deep learning models by averaging weights from multiple
    points in the training trajectory.
    
    Args:
        start_epoch: The epoch to start averaging from. Default is 10.
        swa_freq: Frequency of weight averaging in epochs. Default is 5.
        verbose: Verbosity level. Default is 1.
    """
    def __init__(self, start_epoch=10, swa_freq=5, verbose=1):
        super(SWA, self).__init__()
        self.start_epoch = start_epoch
        self.swa_freq = swa_freq
        self.verbose = verbose
        self.swa_weights = None
        self.swa_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        # Check if we should update SWA
        if epoch + 1 >= self.start_epoch and (epoch + 1 - self.start_epoch) % self.swa_freq == 0:
            # Get current model weights
            current_weights = self.model.get_weights()
            
            # Update SWA weights
            if self.swa_weights is None:
                # First time - just copy the weights
                self.swa_weights = current_weights
            else:
                # Update running average
                for i, w in enumerate(current_weights):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + w) / (self.swa_count + 1)
            
            # Update count
            self.swa_count += 1
            
            if self.verbose > 0:
                log_queue.put(f"SWA: Updated weights average at epoch {epoch+1} (total models: {self.swa_count})")
    
    def on_train_end(self, logs=None):
        # Apply SWA weights when training ends
        if self.swa_weights is not None:
            if self.verbose > 0:
                log_queue.put(f"SWA: Applying averaged weights from {self.swa_count} models")
            
            # Store original weights to allow recovery
            self.original_weights = self.model.get_weights()
            
            # Set SWA weights to model
            self.model.set_weights(self.swa_weights)
            
            if self.verbose > 0:
                log_queue.put("SWA: Weights successfully applied")
        else:
            if self.verbose > 0:
                log_queue.put("SWA: No weights were averaged, model remains unchanged")
                
    def get_original_weights(self):
        """Return the original (non-SWA) weights if available"""
        if hasattr(self, 'original_weights'):
            return self.original_weights
        return None
        
    def reset_original_weights(self):
        """Reset model to the original weights before SWA"""
        if hasattr(self, 'original_weights'):
            self.model.set_weights(self.original_weights)
            return True
        return False