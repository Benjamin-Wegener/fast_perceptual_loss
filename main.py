import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
import time
import datetime

# Import custom modules (with relative imports)
from model import create_fast_perceptual_model
from training_core import train_fast_perceptual_model
from training_ui import update_stats_chart
from globals import log_queue, stop_training, current_batch_size, default_epochs, default_steps_per_epoch, default_learning_rate

def create_gui():
    root = tk.Tk()
    root.title("Fast Perceptual Loss Model Trainer")
    root.geometry("1200x800")
    main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    left_panel = ttk.Frame(main_pane, width=300)
    main_pane.add(left_panel, weight=1)
    left_pane = ttk.PanedWindow(left_panel, orient=tk.VERTICAL)
    left_pane.pack(fill=tk.BOTH, expand=True)

    control_frame = ttk.Frame(left_pane)
    left_pane.add(control_frame, weight=2)

    chart_frame = ttk.LabelFrame(left_pane, text="Training Stats & Charts")
    left_pane.add(chart_frame, weight=3)

    right_panel = ttk.Frame(main_pane)
    main_pane.add(right_panel, weight=3)
    right_pane = ttk.PanedWindow(right_panel, orient=tk.VERTICAL)
    right_pane.pack(fill=tk.BOTH, expand=True)

    log_frame = ttk.LabelFrame(right_pane, text="Training Log")
    right_pane.add(log_frame, weight=1)

    viz_frame = ttk.LabelFrame(right_pane, text="Sample Patch Visualization")
    right_pane.add(viz_frame, weight=3)

    preview_frame = ttk.Frame(viz_frame, borderwidth=2, relief=tk.GROOVE)
    preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    viz_frames = [preview_frame]

    # Dataset folder selection
    ttk.Label(control_frame, text="Dataset Folder:").pack(anchor=tk.W, padx=5, pady=5)
    dataset_folder_var = tk.StringVar(value="Set14")
    dataset_entry = ttk.Entry(control_frame, textvariable=dataset_folder_var)
    dataset_entry.pack(fill=tk.X, padx=5, pady=2)
    ttk.Button(control_frame, text="Browse...", command=lambda: browse_folder(dataset_entry)).pack(anchor=tk.W, padx=5, pady=2)

    # Training parameters (more compact layout)
    params_frame = ttk.LabelFrame(control_frame, text="Training Parameters", padding=2)
    params_frame.pack(fill=tk.X, padx=5, pady=5, ipady=2)

    # Create a grid layout for parameters with 2 columns
    params_grid = ttk.Frame(params_frame)
    params_grid.pack(fill=tk.X, padx=2, pady=2)
    params_grid.columnconfigure(0, weight=1)
    params_grid.columnconfigure(1, weight=1)
    params_grid.columnconfigure(2, weight=1)
    params_grid.columnconfigure(3, weight=1)

    # Row 1: Batch Size & Epochs
    ttk.Label(params_grid, text="Batch:", font=('TkDefaultFont', 9)).grid(row=0, column=0, sticky=tk.W, padx=2, pady=1)
    batch_size_var = tk.StringVar(value=str(current_batch_size))
    
    # Batch size with +/- buttons in a single horizontal layout
    batch_frame = ttk.Frame(params_grid)
    batch_frame.grid(row=0, column=1, sticky=tk.W, padx=2, pady=1)
    
    ttk.Button(batch_frame, text="-", width=2, command=lambda: decrease_batch_size(batch_size_var)).pack(side=tk.LEFT, padx=0)
    batch_size_entry = ttk.Entry(batch_frame, textvariable=batch_size_var, width=3)
    batch_size_entry.pack(side=tk.LEFT, padx=1)
    ttk.Button(batch_frame, text="+", width=2, command=lambda: increase_batch_size(batch_size_var)).pack(side=tk.LEFT, padx=0)

    # Epochs in same row
    ttk.Label(params_grid, text="Epochs:", font=('TkDefaultFont', 9)).grid(row=0, column=2, sticky=tk.W, padx=2, pady=1)
    epochs_var = tk.IntVar(value=default_epochs)
    epochs_entry = ttk.Entry(params_grid, textvariable=epochs_var, width=5)
    epochs_entry.grid(row=0, column=3, sticky=tk.W, padx=2, pady=1)

    # Row 2: Steps per epoch and learning rate
    ttk.Label(params_grid, text="Steps/Epoch:", font=('TkDefaultFont', 9)).grid(row=1, column=0, sticky=tk.W, padx=2, pady=1)
    steps_var = tk.IntVar(value=default_steps_per_epoch)
    steps_entry = ttk.Entry(params_grid, textvariable=steps_var, width=5)
    steps_entry.grid(row=1, column=1, sticky=tk.W, padx=2, pady=1)
    
    ttk.Label(params_grid, text="Base LR:", font=('TkDefaultFont', 9)).grid(row=1, column=2, sticky=tk.W, padx=2, pady=1)
    init_lr_var = tk.DoubleVar(value=default_learning_rate)
    init_lr_entry = ttk.Entry(params_grid, textvariable=init_lr_var, width=6)
    init_lr_entry.grid(row=1, column=3, sticky=tk.W, padx=2, pady=1)

    # Row 3: Only SWA checkbox (loss function selection removed)
    use_swa_var = tk.BooleanVar(value=True)
    swa_check = ttk.Checkbutton(params_grid, text="Use SWA", variable=use_swa_var)
    swa_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=2, pady=1)

    # Progress bar with percentage and ETA
    progress_var = tk.DoubleVar(value=0)
    progress_frame = ttk.LabelFrame(control_frame, text="Training Progress")
    progress_frame.pack(fill=tk.X, padx=5, pady=10)
    progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate', variable=progress_var)
    progress_bar.pack(fill=tk.X, padx=5, pady=5)
    
    # Info variables
    percentage_var = tk.StringVar(value="0.0%")
    eta_var = tk.StringVar(value="ETA: --:--:--")
    current_lr_var = tk.StringVar(value=f"Current LR: {default_learning_rate}")
    
    # Store variables on root for thread-safe updates
    root.progress_var = progress_var
    root.percentage_var = percentage_var
    root.eta_var = eta_var
    root.current_lr_var = current_lr_var

    progress_info_frame = ttk.Frame(progress_frame)
    progress_info_frame.pack(fill=tk.X, padx=5, pady=2)
    percentage_label = ttk.Label(progress_info_frame, textvariable=percentage_var)
    percentage_label.pack(side=tk.LEFT, padx=5)
    eta_label = ttk.Label(progress_info_frame, textvariable=eta_var)
    eta_label.pack(side=tk.RIGHT, padx=5)

    # Current LR display
    current_lr_label = ttk.Label(progress_info_frame, textvariable=current_lr_var)
    current_lr_label.pack(side=tk.BOTTOM, padx=5, pady=2)

    # Model info
    model_info_frame = ttk.LabelFrame(control_frame, text="Model Information")
    model_info_frame.pack(fill=tk.X, padx=5, pady=5)
    model_info_text = tk.Text(model_info_frame, height=3, width=30, wrap=tk.WORD)
    model_info_text.pack(fill=tk.X, padx=5, pady=5)
    model_info_text.insert(tk.END, "FastPerceptualLoss Model: Lightweight model that mimics VGG19 perceptual features using AdamW and OneCycleLR scheduling.")
    model_info_text.config(state=tk.DISABLED)

    # Stats charts
    stats_fig = Figure(figsize=(4, 3), dpi=100)
    root.stats_fig = stats_fig
    stats_canvas = FigureCanvasTkAgg(stats_fig, master=chart_frame)
    stats_canvas_widget = stats_canvas.get_tk_widget()
    stats_canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    root.stats_canvas = stats_canvas

    # Setup initial plot
    setup_stats_plot(stats_fig)

    # Log area with copy button
    log_container = ttk.Frame(log_frame)
    log_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    log_text = scrolledtext.ScrolledText(log_container, wrap=tk.WORD, width=80, height=10)
    log_text.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
    log_text.insert(tk.END, "Welcome to the Fast Perceptual Loss Model Trainer\n")
    log_text.insert(tk.END, "Training uses AdamW optimizer with OneCycleLR scheduling for optimal convergence\n")
    log_text.insert(tk.END, "Set training parameters and click 'Start Training' to begin\n")
    log_text.config(state=tk.DISABLED)
    
    # Add copy button for log text
    def copy_log_to_clipboard():
        log_text.config(state=tk.NORMAL)
        log_content = log_text.get("1.0", tk.END)
        log_text.config(state=tk.DISABLED)
        root.clipboard_clear()
        root.clipboard_append(log_content)
        log_queue.put("Log content copied to clipboard")
        
    copy_button = ttk.Button(log_container, text="Copy Log", command=copy_log_to_clipboard)
    copy_button.pack(side=tk.BOTTOM, anchor=tk.E, padx=5, pady=5)

    # Buttons
    button_frame = ttk.Frame(control_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=10)
    
    # Define training functions first so they're available when referenced
    def training_thread(dataset_folder, epochs, steps_per_epoch, initial_lr, use_swa):
        try:
            # Create Checkpoints directory if it doesn't exist
            os.makedirs('Checkpoints', exist_ok=True)
            
            # Load VGG19
            log_queue.put("Loading VGG19 model...")
            vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
            vgg_model.trainable = False
            
            # Create the enhanced perceptual loss model
            log_queue.put("Creating FastPerceptualLoss model...")
            model = create_fast_perceptual_model(input_shape=(None, None, 3))
            log_queue.put(f"Total parameters: {model.count_params():,}")
            
            # Start the training process
            train_fast_perceptual_model(
                model,
                vgg_model,
                dataset_folder,
                batch_size_var=batch_size_var,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                initial_lr=initial_lr,
                canvas_frames=viz_frames,
                progress_var=progress_var,
                stats_canvas=stats_canvas,
                stats_fig=stats_fig,
                current_lr_var=current_lr_var,
                use_swa=use_swa
            )
            
            # Re-enable start button after training
            root.after(0, lambda: start_button.config(state=tk.NORMAL))
            root.after(0, lambda: stop_button.config(state=tk.DISABLED))
            
        except Exception as e:
            log_queue.put(f"Error during training: {str(e)}")
            # Re-enable start button
            root.after(0, lambda: start_button.config(state=tk.NORMAL))
            root.after(0, lambda: stop_button.config(state=tk.DISABLED))

    def start_training():
        global stop_training
        stop_training = False
        # Clear previous visualization if it exists
        for widget in viz_frames[0].winfo_children():
            widget.destroy()
            
        dataset_folder = dataset_folder_var.get()
        
        # Validate dataset folder
        if not os.path.exists(dataset_folder):
            log_queue.put(f"Error: Dataset folder '{dataset_folder}' does not exist!")
            return
            
        # Validate batch size
        try:
            global current_batch_size
            current_batch_size = int(batch_size_var.get())
            if current_batch_size <= 0:
                raise ValueError("Batch size must be positive")
            log_queue.put(f"Setting initial batch size to: {current_batch_size}")
        except ValueError as e:
            log_queue.put(f"Invalid batch size: {str(e)}. Using default value.")
            batch_size_var.set(str(current_batch_size))
            
        # Get training parameters
        epochs = epochs_var.get()
        steps_per_epoch = steps_var.get()
        initial_lr = init_lr_var.get()
        use_swa = use_swa_var.get()
        
        # Reset progress
        progress_var.set(0)
        percentage_var.set("0.0%")
        eta_var.set("ETA: --:--:--")
        
        # Update UI state
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        
        # Start training in a separate thread
        threading.Thread(
            target=training_thread, 
            args=(dataset_folder, epochs, steps_per_epoch, initial_lr, use_swa),
            daemon=True
        ).start()

    def stop_training_func():
        global stop_training
        stop_training = True
        log_queue.put("Stopping training after current epoch...")
        
    # Now create the buttons with the defined functions
    start_button = ttk.Button(button_frame, text="Start Training", command=start_training)
    start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(button_frame, text="Stop Training", command=stop_training_func, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=5)

    # Setup log update function
    def update_log():
        while not log_queue.empty():
            message = log_queue.get()
            log_text.config(state=tk.NORMAL)
            log_text.insert(tk.END, f"{message}\n")
            log_text.see(tk.END)
            log_text.config(state=tk.DISABLED)
        root.after(100, update_log)
    
    # Start log update loop
    update_log()

    return root

def update_chart_size(chart_frame, fig, canvas):
    """Update chart size based on parent frame size"""
    try:
        # Get parent frame size
        width = chart_frame.winfo_width()
        height = chart_frame.winfo_height()
        
        # Apply minimum size
        width = max(width, 200)
        height = max(height, 150)
        
        # Update figure size
        dpi = fig.dpi
        fig.set_size_inches(width/dpi, height/dpi)
        
        # Update layout and redraw
        fig.tight_layout()
        canvas.draw_idle()
    except Exception as e:
        log_queue.put(f"Error resizing chart: {str(e)}")

def setup_stats_plot(fig):
    """Initialize empty statistics plots"""
    fig.clear()
    ax1 = fig.add_subplot(211)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax2 = fig.add_subplot(212)
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True)
    ax2.set_yscale('log')  # Use log scale for LR
    fig.tight_layout()

def browse_folder(entry):
    """Open file dialog to select dataset folder"""
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def increase_batch_size(batch_size_var):
    """Increase batch size by 1"""
    global current_batch_size
    current_batch_size += 1
    batch_size_var.set(str(current_batch_size))
    log_queue.put(f"Batch size increased to {current_batch_size} for next epoch")

def decrease_batch_size(batch_size_var):
    """Decrease batch size by 1, with minimum of 1"""
    global current_batch_size
    if current_batch_size > 1:
        current_batch_size -= 1
        batch_size_var.set(str(current_batch_size))
        log_queue.put(f"Batch size decreased to {current_batch_size} for next epoch")
    else:
        log_queue.put("Batch size cannot be less than 1")

if __name__ == "__main__":
    # Configure TensorFlow for better performance
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Allow memory growth to avoid allocating all GPU memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            log_queue.put(f"GPU available: {len(physical_devices)} device(s)")
        except Exception as e:
            log_queue.put(f"Error configuring GPU: {str(e)}")
    else:
        log_queue.put("No GPU detected, using CPU for training (slower)")
    
    # Create and run the GUI
    root = create_gui()
    root.mainloop()