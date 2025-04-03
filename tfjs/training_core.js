import { logQueue, getStopTraining, setStopTraining, currentBatchSize, setCurrentBatchSize } from './globals.js';
import { OneCycleLR } from './onecycle_lr.js';
import { SWA } from './swa.js';

// VGG19 preprocessing function
function preprocessForVGG19(imageTensor) {
  return tf.tidy(() => {
    // VGG19 preprocessing: RGB mean subtraction
    // VGG19 was trained with RGB values in [0, 255] with mean RGB values subtracted
    // [123.68, 116.779, 103.939] are the RGB means for the ImageNet dataset
    
    // First, scale tensor from [0,1] to [0,255]
    const scaled = tf.mul(imageTensor, 255);
    
    // Create RGB means tensor with same shape as input
    const rgbMeans = tf.tensor([123.68, 116.779, 103.939]).reshape([1, 1, 3]);
    
    // Subtract means
    return tf.sub(scaled, rgbMeans);
  });
}

// L1 loss function instead of MSE
function l1Loss(yTrue, yPred) {
  /**
   * Mean Absolute Error (L1) loss with proper scaling.
   * Scaled to be batch size independent.
   */
  return tf.tidy(() => {
    const absDiff = tf.abs(tf.sub(yTrue, yPred));
    return tf.mean(absDiff);
  });
}

// Combined loss that balances between MSE and MAE
function combinedLoss(yTrue, yPred, alpha = 0.8) {
  return tf.tidy(() => {
    const mse = tf.mean(tf.square(tf.sub(yTrue, yPred)));
    const mae = tf.mean(tf.abs(tf.sub(yTrue, yPred)));
    return tf.add(tf.mul(mse, alpha), tf.mul(mae, tf.sub(tf.scalar(1.0), alpha)));
  });
}

// Perceptual loss for feature maps
function perceptualLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const epsilon = 1e-7;
    
    // Compute feature magnitudes
    const yTrueSquared = tf.square(yTrue);
    const yPredSquared = tf.square(yPred);
    
    // Sum across feature dimension and add epsilon for numerical stability
    const yTrueSumSquared = tf.add(tf.sum(yTrueSquared, -1, true), epsilon);
    const yPredSumSquared = tf.add(tf.sum(yPredSquared, -1, true), epsilon);
    
    // Square root to get magnitudes
    const yTrueNorm = tf.sqrt(yTrueSumSquared);
    const yPredNorm = tf.sqrt(yPredSumSquared);
    
    // Normalize the features
    const yTrueNormalized = tf.div(yTrue, tf.add(yTrueNorm, epsilon));
    const yPredNormalized = tf.div(yPred, tf.add(yPredNorm, epsilon));
    
    // Compute normalized difference
    const featureDiff = tf.mean(tf.square(tf.sub(yTrueNormalized, yPredNormalized)));
    
    // Add magnitude difference term (scaled down)
    const magnitudeDiff = tf.div(
      tf.mean(tf.square(tf.sub(yTrueNorm, yPredNorm))),
      tf.add(tf.mean(tf.square(yTrueNorm)), epsilon)
    );
    
    // Combine with more weight on feature differences than magnitude
    return tf.add(featureDiff, tf.mul(magnitudeDiff, 0.1));
  });
}

// Compile the perceptual model with Adam optimizer and L1 loss
async function compileModel(model, initialLr = 0.01, lossFunction = "l1", weightDecay = 0.01) {
  logQueue.put(`Compiling model with Adam optimizer and L1 loss`);
  
  // Set default weight decay if not provided
  if (weightDecay === null) {
    weightDecay = 0.01;
  }
  
  // TensorFlow.js doesn't have AdamW out of the box, we'll handle weight decay manually
  const optimizer = tf.train.adam(initialLr);
  
  // Always use L1 loss regardless of the loss_function parameter
  const loss = l1Loss;
  
  // Compile the model
  model.compile({
    optimizer: optimizer,
    loss: loss
  });
  
  // Force optimizer initialization with a dummy batch
  try {
    // Create dummy data with dimensions that match the expected size
    const inputSize = 224; // VGG19 expected input size
    const outputSize = inputSize / 4; // Expected output size
    
    const dummyX = tf.zeros([1, inputSize, inputSize, 3]);
    const dummyY = tf.zeros([1, outputSize, outputSize, 512]); // Changed to 512 for VGG19 feature maps
    
    // Run one training step to initialize optimizer
    await model.trainOnBatch(dummyX, dummyY);
    
    // Clean up tensors
    dummyX.dispose();
    dummyY.dispose();
    
    logQueue.put("Optimizer initialized successfully");
  } catch (e) {
    logQueue.put(`Error during optimizer initialization: ${e.message}`);
  }
  
  return model;
}

// Train the fast perceptual loss model
async function trainFastPerceptualModel({
  model,
  featureExtractor,
  dataset,
  batchSizeVar,
  epochs = 100,
  stepsPerEpoch = 50,
  initialLr = 0.01,
  visualizationCallbacks = {},
  progressCallbacks = {},
  useSwa = true,
  weightDecay = 0.01
}) {
  // Reset training state
  setStopTraining(false);
  
  // Get current batch size
  // Use setCurrentBatchSize instead of direct assignment
  const batchSize = parseInt(batchSizeVar.value);
  setCurrentBatchSize(batchSize);
  
  logQueue.put(`Training FastPerceptualLoss model with Adam optimizer`);
  logQueue.put(`Using OneCycleLR (base lr: ${initialLr}, max lr: ${initialLr*10}) with weight decay: ${weightDecay || 0.01}`);
  logQueue.put(`Using L1 loss as requested`);
  
  // Compile the model
  model = await compileModel(model, initialLr, "l1", weightDecay);
  
  // Setup OneCycleLR scheduler
  const oneCycleLr = new OneCycleLR({
    maxLr: initialLr * 10,
    stepsPerEpoch: stepsPerEpoch,
    epochs: epochs,
    minLr: initialLr,
    warmupPct: 0.3,
    onLrUpdate: progressCallbacks.updateLr || null,
    weightDecay: weightDecay || null,
    verbose: 1
  });
  
  // Setup SWA if enabled
  let swaCallback = null;
  if (useSwa) {
    // Start SWA from 75% of training
    const swaStart = Math.floor(epochs * 0.75);
    swaCallback = new SWA({
      startEpoch: swaStart,
      swaFreq: 1,
      verbose: 1
    });
    logQueue.put(`Stochastic Weight Averaging (SWA) enabled starting from epoch ${swaStart}`);
  }
  
  // Training loop
  let startEpoch = 0;
  
  // Storage for loss history and visualization
  const lossHistory = [];
  const lrHistory = [];
  
  try {
    // Store start time for ETA calculation
    let startTime = Date.now();
    let epochStartTime = startTime;
    
    // Initialize oneCycleLR
    oneCycleLr.onTrainBegin(model);
    
    // Main epoch loop
    for (let epoch = startEpoch; epoch < epochs; epoch++) {
      // Check if training should be stopped
      if (getStopTraining()) {
        logQueue.put("Training stopped by user");
        break;
      }
      
      // Notify epoch beginning
      oneCycleLr.onEpochBegin(epoch, model);
      if (swaCallback) swaCallback.onEpochEnd(epoch, model);
      
      // Reset loss accumulator for this epoch
      let epochLoss = 0;
      
      // Steps per epoch loop
      for (let step = 0; step < stepsPerEpoch; step++) {
        // Check if training should be stopped
        if (getStopTraining()) {
          break;
        }
        
        // Get scheduler updates for this batch
        const lrConfig = oneCycleLr.onBatchBegin(step, model);
        
        // Apply manual weight decay if needed
        if (lrConfig.weightDecay !== null) {
          applyWeightDecay(model, lrConfig.lr, lrConfig.weightDecay);
        }
        
        // Get batch of data
        const batch = await getBatchFromDataset(dataset, featureExtractor);
        if (!batch) {
          logQueue.put("Error getting batch from dataset");
          continue;
        }
        
        // Destructure batch
        const { xs, ys } = batch;
        
        // Train on batch
        const history = await model.trainOnBatch(xs, ys);
        
        // Update loss accumulator
        epochLoss += history;
        
        // Clean up tensors
        xs.dispose();
        ys.dispose();
        
        // Update progress within epoch
        if (progressCallbacks.updateProgress) {
          const progress = (epoch * stepsPerEpoch + step) / (epochs * stepsPerEpoch) * 100;
          progressCallbacks.updateProgress(progress);
        }
        
        // Visualize occasionally
        if (step === stepsPerEpoch - 1 && visualizationCallbacks.visualize) {
          await visualizationCallbacks.visualize(model, featureExtractor, epoch);
        }
      }
      
      // Compute average loss for this epoch
      const avgLoss = epochLoss / stepsPerEpoch;
      lossHistory.push(avgLoss);
      
      // Store current learning rate
      lrHistory.push(model.optimizer.learningRate);
      
      // Notify epoch end
      oneCycleLr.onEpochEnd(epoch, model);
      if (swaCallback) swaCallback.onEpochEnd(epoch, model);
      
      // Log epoch results
      logQueue.put(`Epoch ${epoch+1}/${epochs} completed with loss: ${avgLoss.toFixed(6)}`);
      
      // Update charts
      if (progressCallbacks.updateCharts) {
        progressCallbacks.updateCharts(
          Array.from({length: lossHistory.length}, (_, i) => i),
          lossHistory,
          lrHistory
        );
      }
      
      // Calculate ETA
      const epochTime = (Date.now() - epochStartTime) / 1000;
      epochStartTime = Date.now();
      
      if (progressCallbacks.updateETA) {
        progressCallbacks.updateETA(epoch, epochs, epochTime);
      }
      
      // Force garbage collection
      tf.disposeVariables();
      tf.engine().endScope();
      tf.engine().startScope();
    }
    
    // Notify training end
    oneCycleLr.onTrainEnd();
    if (swaCallback) swaCallback.onTrainEnd(model);
    
    // Apply final progress
    if (progressCallbacks.updateProgress) {
      progressCallbacks.updateProgress(100);
    }
    
    logQueue.put("Training completed successfully");
    
    // Save the model
    try {
      await saveModel(model, "fast-perceptual-loss-model");
      logQueue.put("Model saved to browser storage");
    } catch (e) {
      logQueue.put(`Error saving model: ${e.message}`);
    }
    
  } catch (e) {
    logQueue.put(`Error during training: ${e.message}`);
    console.error("Training error:", e);
  }
  
  return model;
}

// Apply weight decay manually (AdamW-like behavior)
function applyWeightDecay(model, lr, decay) {
  // Skip non-trainable weights and biases
  const trainableWeights = model.weights.filter(w => 
    w.trainable && !w.name.includes('bias'));
  
  // Apply weight decay
  trainableWeights.forEach(w => {
    // L2 regularization: w -= lr * decay * w
    const newValue = tf.tidy(() => {
      return tf.sub(
        w.val,
        tf.mul(
          tf.mul(tf.scalar(lr), tf.scalar(decay)),
          w.val
        )
      );
    });
    
    w.val.assign(newValue);
    newValue.dispose();
  });
}

// Get a batch of data from the dataset
async function getBatchFromDataset(dataset, featureExtractor) {
  try {
    // Get a batch of input images
    const xs = tf.tidy(() => {
      // Sample random images from dataset
      const indices = [];
      for (let i = 0; i < currentBatchSize; i++) {
        indices.push(Math.floor(Math.random() * dataset.length));
      }
      
      // Stack the selected images into a batch
      const batchImages = tf.stack(indices.map(idx => dataset[idx]));
      
      // Ensure the images are resized to 224x224 for VGG19
      const resized = tf.image.resizeBilinear(batchImages, [224, 224]);
      
      // Apply VGG19 preprocessing
      return preprocessForVGG19(resized);
    });
    
    // Generate target features using the feature extractor
    const ys = tf.tidy(() => {
      return featureExtractor.predict(xs);
    });
    
    return { xs, ys };
  } catch (e) {
    logQueue.put(`Error creating batch: ${e.message}`);
    return null;
  }
}

// Save model to IndexedDB
async function saveModel(model, name) {
  try {
    const timestamp = Date.now();
    const saveResult = await model.save(`indexeddb://${name}-${timestamp}`);
    return saveResult;
  } catch (e) {
    logQueue.put(`Error saving model: ${e.message}`);
    throw e;
  }
}

// Load model from IndexedDB
async function loadModel(name) {
  try {
    const models = await tf.io.listModels();
    const modelKeys = Object.keys(models).filter(key => key.includes(name));
    
    if (modelKeys.length === 0) {
      logQueue.put(`No saved models found with name: ${name}`);
      return null;
    }
    
    // Get the latest model (highest timestamp)
    const latestModelKey = modelKeys.sort().reverse()[0];
    logQueue.put(`Loading model: ${latestModelKey}`);
    
    return await tf.loadLayersModel(`indexeddb://${latestModelKey}`);
  } catch (e) {
    logQueue.put(`Error loading model: ${e.message}`);
    return null;
  }
}

// Export training functions
export {
  trainFastPerceptualModel,
  compileModel,
  l1Loss,
  combinedLoss,
  perceptualLoss,
  saveModel,
  loadModel,
  preprocessForVGG19
};