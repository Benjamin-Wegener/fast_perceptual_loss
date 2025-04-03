// training_ui.js
// UI-related functions for visualizing training progress

import { logQueue } from './globals.js';

// VGG19 preprocessing function
function preprocessForVGG19(imageTensor) {
  return tf.tidy(() => {
    // VGG19 preprocessing: RGB mean subtraction
    // Scale tensor from [0,1] to [0,255]
    const scaled = tf.mul(imageTensor, 255);
    
    // Create RGB means tensor with same shape as input
    const rgbMeans = tf.tensor([123.68, 116.779, 103.939]).reshape([1, 1, 3]);
    
    // Subtract means
    return tf.sub(scaled, rgbMeans);
  });
}

// Update statistics chart
function updateStatsChart(lossChart, lrChart, epochs, lossValues, lrValues) {
  try {
    // Make sure we have data to plot
    if (!epochs || epochs.length === 0 || !lossValues || lossValues.length === 0) {
      return;
    }
    
    // Update loss chart
    lossChart.data.labels = epochs;
    lossChart.data.datasets[0].data = lossValues;
    lossChart.update();
    
    // Update learning rate chart
    lrChart.data.labels = epochs;
    lrChart.data.datasets[0].data = lrValues;
    lrChart.update();
  } catch (e) {
    logQueue.put(`Error updating stats chart: ${e.message}`);
    console.error("Chart update error:", e);
  }
}

// Visualization helper - render tensor to canvas
async function renderTensorToCanvas(tensor, canvas, colormap = null) {
  try {
    // Make sure tensor is 3D (height, width, channels)
    let displayTensor = tensor;
    
    // Handle feature maps by summing across channels
    if (tensor.shape.length === 3 && tensor.shape[2] > 3) {
      displayTensor = tf.tidy(() => {
        // Sum across channels
        const summed = tf.sum(tensor, -1);
        
        // Normalize to [0, 1]
        const min = tf.min(summed);
        const max = tf.max(summed);
        const normalized = tf.div(
          tf.sub(summed, min),
          tf.add(tf.sub(max, min), tf.scalar(1e-7))
        );
        
        // Apply colormap if requested
        if (colormap === 'viridis') {
          // Simple approximation of viridis colormap
          const r = tf.sub(tf.scalar(0.5), tf.mul(normalized, tf.scalar(0.5)));
          const g = normalized;
          const b = tf.sqrt(normalized);
          return tf.stack([r, g, b], -1);
        } else if (colormap === 'hot') {
          // Simple approximation of hot colormap
          const r = tf.minimum(tf.mul(normalized, tf.scalar(3)), tf.scalar(1));
          const g = tf.maximum(tf.minimum(tf.sub(tf.mul(normalized, tf.scalar(3)), tf.scalar(1)), tf.scalar(1)), tf.scalar(0));
          const b = tf.maximum(tf.minimum(tf.sub(tf.mul(normalized, tf.scalar(3)), tf.scalar(2)), tf.scalar(1)), tf.scalar(0));
          return tf.stack([r, g, b], -1);
        } else {
          // Grayscale (repeat normalized to 3 channels)
          return tf.stack([normalized, normalized, normalized], -1);
        }
      });
    }
    
    // Ensure we have a properly shaped tensor for display
    const bytes = await tf.browser.toPixels(displayTensor);
    
    // Get canvas context
    const ctx = canvas.getContext('2d');
    
    // Create ImageData
    const imageData = new ImageData(
      new Uint8ClampedArray(bytes),
      displayTensor.shape[1],
      displayTensor.shape[0]
    );
    
    // Draw to canvas
    ctx.putImageData(imageData, 0, 0);
    
    // Dispose the display tensor if it's different from the input
    if (displayTensor !== tensor) {
      displayTensor.dispose();
    }
  } catch (e) {
    logQueue.put(`Error rendering tensor: ${e.message}`);
    console.error("Render error:", e);
    
    // Draw error pattern
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffcccc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ff0000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '14px Arial';
    ctx.fillText('Error', canvas.width/2, canvas.height/2);
  }
}

// Create a visualization of model predictions
async function createModelVisualization(model, featureExtractor, sampleImage, canvases, epoch) {
  try {
    const { inputCanvas, targetCanvas, predCanvas, diffCanvas } = canvases;
    
    // First render the original input image
    await renderTensorToCanvas(sampleImage, inputCanvas);
    
    // Expand dimensions to create a batch of size 1 and resize to 224x224 for VGG19
    const inputTensor = tf.tidy(() => {
      const expanded = tf.expandDims(sampleImage, 0);
      // Resize to VGG19 input size
      const resized = tf.image.resizeBilinear(expanded, [224, 224]);
      // Apply VGG19 preprocessing
      return preprocessForVGG19(resized);
    });
    
    // Get VGG features (target)
    const targetFeatures = tf.tidy(() => {
      return featureExtractor.predict(inputTensor);
    });
    
    // Get model prediction
    const predictedFeatures = tf.tidy(() => {
      return model.predict(inputTensor);
    });
    
    // Calculate absolute difference
    const diffFeatures = tf.tidy(() => {
      return tf.abs(tf.sub(predictedFeatures, targetFeatures));
    });
    
    // Render target features
    await renderTensorToCanvas(targetFeatures.squeeze(), targetCanvas, 'viridis');
    
    // Render predicted features
    await renderTensorToCanvas(predictedFeatures.squeeze(), predCanvas, 'viridis');
    
    // Render feature difference
    await renderTensorToCanvas(diffFeatures.squeeze(), diffCanvas, 'hot');
    
    // Dispose tensors
    inputTensor.dispose();
    targetFeatures.dispose();
    predictedFeatures.dispose();
    diffFeatures.dispose();
    
  } catch (e) {
    logQueue.put(`Error creating visualization: ${e.message}`);
    console.error("Visualization error:", e);
  }
}

// Advanced visualization callback with UI integration
class AdvancedVisualizationCallback {
  /**
   * Create a callback for visualizing model predictions
   * @param {Array} dataset Array of image tensors
   * @param {Object} featureExtractor Feature extractor model
   * @param {Object} canvases Object containing canvas elements
   * @param {Object} oneCycleLr OneCycleLR instance for learning rate tracking
   */
  constructor(dataset, featureExtractor, canvases, oneCycleLr = null) {
    this.dataset = dataset;
    this.featureExtractor = featureExtractor;
    this.canvases = canvases;
    this.oneCycleLr = oneCycleLr;
    this.sampleData = null;
    this.losses = [];
    this.learningRates = [];
    this.epochs = [];
    this.lastVisualizationUpdate = 0;
    this.visualizationUpdateInterval = 1000; // 1 second
  }
  
  // Initialize by getting sample data
  initialize() {
    // Get a sample image for visualization
    if (this.dataset && this.dataset.length > 0) {
      this.sampleData = this.dataset[0];
      logQueue.put("Sample data loaded for visualization");
    }
  }
  
  // Update visualization on epoch end
  async onEpochEnd(epoch, model) {
    try {
      // Skip if we don't have sample data
      if (!this.sampleData) {
        this.initialize();
        if (!this.sampleData) {
          return;
        }
      }
      
      // Only update visualization periodically
      const currentTime = Date.now();
      if (currentTime - this.lastVisualizationUpdate < this.visualizationUpdateInterval) {
        return;
      }
      
      this.lastVisualizationUpdate = currentTime;
      
      // Create visualization
      await createModelVisualization(
        model,
        this.featureExtractor,
        this.sampleData,
        this.canvases,
        epoch
      );
      
    } catch (e) {
      logQueue.put(`Error in visualization callback: ${e.message}`);
      console.error("Visualization callback error:", e);
    }
  }
}

// Progress tracking helper
function formatTime(seconds) {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  }
}

// Function to update ETA display
function updateETA(epoch, totalEpochs, epochTimes) {
  // Calculate average epoch time
  const avgEpochTime = epochTimes.reduce((a, b) => a + b, 0) / epochTimes.length;
  
  // Calculate remaining epochs
  const remainingEpochs = totalEpochs - epoch - 1;
  
  // Calculate estimated time remaining
  const estimatedSeconds = avgEpochTime * remainingEpochs;
  
  return formatTime(estimatedSeconds);
}

// Export visualization functions
export {
  updateStatsChart,
  renderTensorToCanvas,
  createModelVisualization,
  AdvancedVisualizationCallback,
  formatTime,
  updateETA,
  preprocessForVGG19
};