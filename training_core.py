import os
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
# Import optimizers from tensorflow keras and tensorflow addons for AdamW
from tensorflow.keras.optimizers import Adam
# We need to import AdamW differently based on TensorFlow version
try:
    # For TensorFlow 2.11+
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    try:
        # For older TensorFlow versions with addons
        from tensorflow_addons.optimizers import AdamW
    except ImportError:
        # Fallback - we'll create a custom AdamW implementation
        from tensorflow.keras.optimizers import Adam
        
        class AdamW(Adam):
            """Adam optimizer with decoupled weight decay."""
            
            def __init__(self, weight_decay=0.01, **kwargs):
                super().__init__(**kwargs)
                self.weight_decay = weight_decay
                
            def _resource_apply_dense(self, grad, var, apply_state=None):
                # Apply weight decay
                if self.weight_decay > 0:
                    var.assign_sub(
                        self.weight_decay * self.learning_rate * var,
                        use_locking=self._use_locking
                    )
                # Then apply regular Adam update
                return super()._resource_apply_dense(grad, var, apply_state)
import matplotlib
# Force matplotlib to use a non-interactive backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import uuid
# Removed signal import as it only works in main thread

# Import the global variables from globals.py
from globals import log_queue, stop_training, current_batch_size, default_epochs, default_steps_per_epoch, default_learning_rate, default_image_size

# This is the fixed import line - importing only create_fast_perceptual_model
from model import create_fast_perceptual_model, WeightedAddLayer, MeanReduceLayer, MaxReduceLayer, StochasticDepth

# Import UI-related functionality
from training_ui import update_stats_chart, AdvancedVisualizationCallback
from training_helper import (
    LoggerCallback, 
    ProgressCallback, 
    ModelStatsCallback,
    BatchSizeChangeCallback
)
from onecycle_lr import OneCycleLR, SWA  # Import the OneCycleLR scheduler and SWA

# Import data processing functions
from training_data import create_training_dataset, process_samples_for_vgg

# Define L1 loss function instead of MSE
def l1_loss(y_true, y_pred):
    """
    Mean Absolute Error (L1) loss with proper scaling and explicit type casting.
    Scaled to be batch size independent.
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Use tf.reduce_mean with absolute difference
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def combined_loss(y_true, y_pred, alpha=0.8):
    """
    Combined loss that balances between MSE and MAE.
    MSE is good for high frequency details while MAE is robust to outliers.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        alpha: Weight between MSE and MAE (higher alpha means more MSE)
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return alpha * mse + (1.0 - alpha) * mae

def perceptual_loss(y_true, y_pred):
    """
    Custom loss for perceptual features that focuses on relative differences.
    Good for feature maps where patterns matter more than exact values.
    """
    # Compute normalized MSE (divide by feature magnitudes)
    epsilon = tf.keras.backend.epsilon()
    y_true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True) + epsilon)
    y_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1, keepdims=True) + epsilon)
    
    # Normalize the features
    y_true_normalized = y_true / (y_true_norm + epsilon)
    y_pred_normalized = y_pred / (y_pred_norm + epsilon)
    
    # Compute normalized difference
    feature_diff = tf.reduce_mean(tf.square(y_true_normalized - y_pred_normalized))
    
    # Add magnitude difference term (scaled down)
    magnitude_diff = tf.reduce_mean(tf.square(y_true_norm - y_pred_norm)) / tf.reduce_mean(tf.square(y_true_norm) + epsilon)
    
    # Combine with more weight on feature differences than magnitude
    return feature_diff + 0.1 * magnitude_diff

# Safer way to get entry widget values
def get_entry_value(master, widget_name, default_value=None):
    """Get an integer/float value from an entry widget with error handling"""
    try:
        # Try to find the widget
        widget = master.nametowidget(widget_name)
        if widget:
            # Try to get and convert its value
            value = widget.get()
            
            # Try as int first, then as float
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return default_value
    except (ValueError, KeyError, AttributeError, tf.errors.OpError):
        # Return default if anything goes wrong
        return default_value
    return default_value

# Compile the perceptual model with AdamW optimizer and L1 loss
def compile_model(model, initial_lr=default_learning_rate, 
                 loss_function="l1", weight_decay=None):
    """
    Compiles the model with AdamW optimizer and L1 loss.
    
    Args:
        model: The model to compile
        initial_lr: Initial learning rate
        loss_function: Loss function parameter is ignored - always using L1
        weight_decay: Weight decay factor (L2 regularization strength)
    """
    log_queue.put(f"Compiling model with AdamW optimizer and L1 loss")
    
    # Set default weight decay if not provided
    if weight_decay is None:
        weight_decay = 0.01
    
    # Configure the AdamW optimizer
    optimizer = AdamW(
        learning_rate=initial_lr,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,  # Add gradient clipping
        amsgrad=False
    )
        
    # Always use L1 loss regardless of the loss_function parameter
    loss = l1_loss
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)
    
    # Force optimizer initialization with a dummy batch
    try:
        # Create dummy data with dimensions that match the target size/4
        input_size = default_image_size
        output_size = input_size // 4  # Expected output size
        
        dummy_x = tf.zeros((1, input_size, input_size, 3))
        dummy_y = tf.zeros((1, output_size, output_size, 256))
        
        # Run one training step to initialize optimizer
        model.train_on_batch(dummy_x, dummy_y)
        log_queue.put("Optimizer initialized successfully")
    except Exception as e:
        log_queue.put(f"Error during optimizer initialization: {str(e)}")
    
    return model

# Instead of using signal handler, we'll rely on the existing stop_training flag
# which is already being checked in various callbacks and loops

# Train the fast perceptual loss model
def train_fast_perceptual_model(model, vgg_model, dataset_folder, 
                           batch_size_var, epochs=default_epochs, steps_per_epoch=default_steps_per_epoch, 
                           initial_lr=default_learning_rate,
                           canvas_frames=None, progress_var=None,
                           stats_canvas=None, stats_fig=None,
                           current_lr_var=None,
                           loss_function="l1",  # Using L1 loss as requested
                           use_swa=True,
                           weight_decay=None):
    global stop_training, current_batch_size
    stop_training = False
    current_batch_size = int(batch_size_var.get())
    
    # Signal handlers removed as they don't work in threads
    # We'll use the existing stop_training flag mechanism instead
    
    # Create only the Checkpoints directory
    checkpoint_dir = './Checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_queue.put(f"Training FastPerceptualLoss model with AdamW optimizer")
    log_queue.put(f"Using OneCycleLR (base lr: {initial_lr}, max lr: {initial_lr*10}) with weight decay: {weight_decay if weight_decay else 0.01}")
    log_queue.put(f"Using L1 loss as requested")

    # Create a sub-model of VGG19 up to block3_conv3
    vgg_submodel = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv3').output)
    vgg_submodel.trainable = False
    
    # Print VGG model dimensions to understand downsampling ratio
    log_queue.put(f"VGG19 Submodel: Input shape {vgg_model.input.shape}, Output shape {vgg_submodel.output.shape}")

    # Compile the model with L1 loss
    model = compile_model(model, initial_lr, "l1", weight_decay)
    
    # Define checkpoint logic with unique file naming to avoid HDF5 conflicts
    timestamp = int(time.time())
    checkpoint_filepath = os.path.join(
        checkpoint_dir, 
        f'fast_perceptual_loss_epoch_{{epoch:02d}}_{timestamp}.h5'
    )
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,  # Save only weights to avoid HDF5 issues
        save_freq='epoch',        # Save after every epoch
        verbose=1,                # Print messages when saving
        save_best_only=True,      # Only save when the model improves
        monitor='loss',           # Monitor the training loss
        mode='min'                # Explicitly set mode to min
    )

    # Check if there are existing checkpoints and load the latest one
    start_epoch = 0
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('fast_perceptual_loss_epoch_')] if os.path.exists(checkpoint_dir) else []
    if checkpoint_files:
        # Extract epoch numbers
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                # Handle both naming formats (with and without timestamp)
                if '_' in f.split('epoch_')[1]:
                    epoch_num = int(f.split('epoch_')[1].split('_')[0])
                else:
                    epoch_num = int(f.split('epoch_')[1].split('.')[0])
                epoch_numbers.append(epoch_num)
            except (IndexError, ValueError):
                continue
                
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)
            # Find the exact filename with this epoch number (could have different timestamps)
            matching_files = [f for f in checkpoint_files if f'epoch_{latest_epoch:02d}' in f]
            if matching_files:
                latest_checkpoint = matching_files[0]  # Use the first matching file
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                log_queue.put(f"Latest checkpoint found: {latest_checkpoint_path} (Epoch {latest_epoch})")
                
                # Register custom objects before loading the model
                custom_objects = {
                    'WeightedAddLayer': WeightedAddLayer,
                    'MeanReduceLayer': MeanReduceLayer,
                    'MaxReduceLayer': MaxReduceLayer,
                    'StochasticDepth': StochasticDepth,
                    'l1_loss': l1_loss,
                    'perceptual_loss': perceptual_loss,
                    'combined_loss': combined_loss,
                    'swish_activation': lambda x: x * tf.nn.sigmoid(x)
                }
                
                try:
                    # Load the model weights from the latest checkpoint
                    model.load_weights(latest_checkpoint_path)
                    start_epoch = latest_epoch
                    log_queue.put(f"Training will resume from epoch {start_epoch + 1}")
                    
                    # Re-compile the model after loading to ensure optimizer is initialized
                    model = compile_model(model, initial_lr, "l1", weight_decay)
                    
                except Exception as e:
                    log_queue.put(f"Error loading checkpoint: {str(e)}, starting from scratch")
        else:
            log_queue.put("No valid checkpoints found. Starting training from scratch.")
    else:
        log_queue.put("No checkpoints found. Starting training from scratch.")

    # Create the data processing pipeline
    try:
        # Create the dataset with the current_batch_size
        raw_dataset = create_training_dataset(dataset_folder, current_batch_size)
        
        # Map the VGG feature extraction to the dataset
        dataset = raw_dataset.map(
            lambda x: process_samples_for_vgg(x, vgg_submodel),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        log_queue.put(f"Dataset pipeline created with batch size {current_batch_size}")
        
    except Exception as e:
        log_queue.put(f"Error creating dataset: {str(e)}")
        return model
        
    # Use only the first frame for visualization
    viz_frame = canvas_frames[0] if canvas_frames else None
    
    # Setup OneCycleLR scheduler
    one_cycle_lr = OneCycleLR(
        max_lr=initial_lr * 10,  # The max LR is typically 10x the initial LR
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        min_lr=initial_lr,  # Start from the specified initial LR
        warmup_pct=0.3,     # 30% of training for warmup
        current_lr_var=current_lr_var,  # Pass UI variable
        weight_decay=weight_decay if weight_decay else None,
        verbose=1
    )
    
    # Setup SWA if enabled
    swa_callback = None
    if use_swa:
        # Start SWA from 75% of training
        swa_start = int(epochs * 0.75)
        swa_callback = SWA(start_epoch=swa_start, swa_freq=1, verbose=1)
        log_queue.put(f"Stochastic Weight Averaging (SWA) enabled starting from epoch {swa_start}")
        
    # Modified DataTrackingCallback to track and update charts
    class DataTrackingCallback(Callback):
        def __init__(self):
            super(DataTrackingCallback, self).__init__()
            self.epochs = []
            self.losses = []
            self.learning_rates = []
            self.last_update_time = 0
            self.update_interval = 1.0  # Update at most once per second
            
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Only update UI periodically to reduce overhead
                current_time = time.time()
                if current_time - self.last_update_time < self.update_interval:
                    # Just store data without updating UI
                    if hasattr(self.model.optimizer, 'lr'):
                        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                    else:
                        current_lr = initial_lr
                    
                    # Get loss value and ensure it's properly formatted as a float
                    current_loss = logs.get('loss', 0.0)
                    if isinstance(current_loss, tf.Tensor):
                        current_loss = current_loss.numpy()
                    current_loss = float(current_loss)
                    
                    self.epochs.append(epoch)
                    self.losses.append(current_loss)
                    self.learning_rates.append(current_lr)
                    return
                    
                self.last_update_time = current_time
                
                # Get current learning rate with error handling
                if hasattr(self.model.optimizer, 'lr'):
                    current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                else:
                    current_lr = initial_lr
                
                # Get loss value and ensure it's properly formatted
                current_loss = logs.get('loss', 0.0)
                if isinstance(current_loss, tf.Tensor):
                    current_loss = current_loss.numpy()
                current_loss = float(current_loss)
                
                # Update tracked data
                self.epochs.append(epoch)
                self.losses.append(current_loss)
                self.learning_rates.append(current_lr)
                
                # Explicitly log the current loss value
                log_queue.put(f"Current loss at epoch {epoch+1}: {current_loss:.6f}")
                
                # Only update chart at end of epoch
                if stats_fig and stats_canvas and root:
                    root.after(0, lambda: update_stats_chart(
                        stats_fig, 
                        stats_canvas, 
                        self.epochs,
                        self.losses,
                        self.learning_rates
                    ))
                if epoch % 3 == 0:  # Clear every 3 epochs
                    tf.keras.backend.clear_session()
                    gc.collect()  # Force garbage collection
            except Exception as e:
                log_queue.put(f"Error in data tracking callback: {str(e)}")
    
    # Get root window for thread-safe operations
    root = None
    if hasattr(batch_size_var, 'master'):
        # Get the root window through master attribute
        root = batch_size_var.master.winfo_toplevel()
    elif canvas_frames and len(canvas_frames) > 0:
        root = canvas_frames[0].winfo_toplevel()
        
    # Store UI variables directly on root for access by callbacks
    if root and progress_var:
        root.progress_var = progress_var
    
    # Setup the advanced visualization callback with the dataset
    visualization_callback = AdvancedVisualizationCallback(
        dataset, vgg_submodel, viz_frame, root, one_cycle_lr
    ) if viz_frame and root else None
    
    logger_callback = LoggerCallback()
    progress_callback = ProgressCallback(root, epochs)
    data_tracking_callback = DataTrackingCallback()
    
    # Add all callbacks - REMOVED any callbacks that might create log folders
    callbacks = [
        model_checkpoint_callback,    # Save checkpoints
        logger_callback,              # Log progress
        progress_callback,            # Update progress bar
        one_cycle_lr,                 # OneCycle learning rate scheduler
        data_tracking_callback,       # Data tracking for charts
    ]
    
    # Add SWA if enabled
    if swa_callback:
        callbacks.append(swa_callback)
    
    # Only add visualization if we have a frame
    if visualization_callback:
        callbacks.append(visualization_callback)
    
    # Create a dynamic batch size callback
    class DynamicBatchSizeCallback(Callback):
        def __init__(self, dataset_folder):
            super(DynamicBatchSizeCallback, self).__init__()
            self.dataset_folder = dataset_folder
            self.last_batch_size = current_batch_size
            
        def on_epoch_end(self, epoch, logs=None):
            global current_batch_size
            
            # Check if batch size has changed
            if self.last_batch_size != current_batch_size:
                log_queue.put(f"Batch size changed from {self.last_batch_size} to {current_batch_size}")
                
                try:
                    # Create new dataset with updated batch size
                    raw_dataset = create_training_dataset(self.dataset_folder, current_batch_size)
                    
                    # Map the VGG feature extraction
                    new_dataset = raw_dataset.map(
                        lambda x: process_samples_for_vgg(x, vgg_submodel),
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                    
                    # Store for future use
                    self._new_dataset = new_dataset
                    log_queue.put(f"New dataset created with batch size {current_batch_size}")
                    
                except Exception as e:
                    log_queue.put(f"Error updating dataset: {str(e)}")
                
                self.last_batch_size = current_batch_size
    
    dynamic_batch_size_callback = DynamicBatchSizeCallback(dataset_folder)
    callbacks.append(dynamic_batch_size_callback)
    
    # Train the model - ONLY ONE TRAINING SESSION for the entire training
    log_queue.put("Starting training...")
    try:
        # Single model.fit call for all epochs using initial_epoch to continue from start_epoch
        # This prevents creating a new training session for each epoch which resets learning rate
        history = model.fit(
            dataset,
            initial_epoch=start_epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=0  # Disable default progress bar
        )
        
        if stop_training:
            log_queue.put("Training stopped by user.")
    
    except KeyboardInterrupt:
        log_queue.put("Training interrupted by user.")
        
    except Exception as e:
        log_queue.put(f"Error in training: {str(e)}")
    
    finally:
        # Call on_train_end for all callbacks
        for callback in callbacks:
            try:
                callback.on_train_end({})
            except Exception as e:
                log_queue.put(f"Error in callback on_train_end: {str(e)}")

    # Save the final trained model with unique timestamp
    final_timestamp = int(time.time())
    final_model_path = os.path.join(checkpoint_dir, f'fast_perceptual_loss_final_{final_timestamp}.h5')
    try:
        model.save_weights(final_model_path)  # Save only weights to avoid HDF5 issues
        log_queue.put(f"FastPerceptualLoss model weights trained and saved to {final_model_path}.")
        
        # Save SWA model if used
        if swa_callback and hasattr(swa_callback, 'swa_weights') and swa_callback.swa_weights is not None:
            # Save original model weights
            original_weights = model.get_weights()
            
            # Apply SWA weights and save
            model.set_weights(swa_callback.swa_weights)
            swa_model_path = os.path.join(checkpoint_dir, f'fast_perceptual_loss_swa_{final_timestamp}.h5')
            model.save_weights(swa_model_path)  # Save only weights
            log_queue.put(f"SWA model weights saved to {swa_model_path}")
            
            # Restore original weights
            model.set_weights(original_weights)
    except Exception as e:
        log_queue.put(f"Error saving final model: {str(e)}")
        # Try alternate save methods if the first fails
        try:
            log_queue.put("Attempting to save with different method...")
            model.save_weights(os.path.join(checkpoint_dir, f'fast_perceptual_loss_weights_{final_timestamp}.h5'))
            log_queue.put(f"Model weights saved successfully.")
        except Exception as e2:
            log_queue.put(f"Failed to save weights as well: {str(e2)}")
    
    return model