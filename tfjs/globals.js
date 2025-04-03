// globals.js
// Global variables and configuration for the Fast Perceptual Loss Trainer

// Queue for thread-safe logging
class LogQueue {
  constructor() {
    this.queue = [];
    this.listeners = [];
  }

  put(message) {
    this.queue.push(message);
    this.notifyListeners(message);
  }

  get() {
    return this.queue.shift();
  }

  empty() {
    return this.queue.length === 0;
  }

  addListener(listener) {
    this.listeners.push(listener);
  }

  notifyListeners(message) {
    this.listeners.forEach(listener => listener(message));
  }
}

// Global flag for training control - using let instead of const to allow modification
let stopTraining = false;

// Getter and setter functions for stopTraining
function getStopTraining() {
  return stopTraining;
}

function setStopTraining(value) {
  stopTraining = value;
}

// Global training parameters with defaults
let currentBatchSize = 12;  // Changed from 8 to 12

// Getter and setter functions for currentBatchSize
function getCurrentBatchSize() {
  return currentBatchSize;
}

function setCurrentBatchSize(value) {
  currentBatchSize = value;
}

const defaultEpochs = 100;
const defaultStepsPerEpoch = 50;
const defaultLearningRate = 0.01;
const defaultImageSize = 224;  // Changed from 256 to 224 for VGG19

// Create directories asynchronously
async function ensureDirectoriesExist() {
  // In browser environment, we'll use IndexedDB for storage instead of filesystem
  // This function is more of a placeholder for compatibility
  console.log("Storage API initialized");
  return true;
}

// Extension of supported image formats
const supportedImageFormats = ['.jpeg', '.jpg', '.png', '.bmp', '.tiff'];

// Available loss functions and their weights/parameters
const lossConfigs = {
  mse: {},
  perceptual: {},
  combined: {
    alpha: 0.8  // Weight between MSE and MAE (higher alpha means more MSE)
  }
};

// Advanced parameters
const advancedParams = {
  swaStartPct: 0.75,     // SWA starts at this percentage of training
  swaFreq: 1,            // SWA model update frequency (epochs)
  swaLrFactor: 0.1,      // SWA learning rate factor (multiplied by max_lr)
  onecycleWarmupPct: 0.3, // OneCycle warmup percentage
  finalDivFactor: 100,   // OneCycle final learning rate divisor
  weightDecay: 0.01,     // L2 regularization strength - set default for Adam
  checkpointFreq: 5      // Save checkpoint every N epochs
};

// Function to get parameter value with fallback
function getParameter(name, defaultValue = null) {
  if (name === 'currentBatchSize') {
    return currentBatchSize;
  }
  
  if (name === 'stopTraining') {
    return stopTraining;
  }
  
  if (window[name] !== undefined) {
    return window[name];
  }
  
  // Check in parameter dictionaries
  for (const paramDict of [advancedParams, lossConfigs]) {
    if (name in paramDict) {
      return paramDict[name];
    }
  }
  
  return defaultValue;
}

// Function to set parameter value
function setParameter(name, value) {
  if (name === 'currentBatchSize') {
    currentBatchSize = value;
    return true;
  }
  
  if (name === 'stopTraining') {
    stopTraining = value;
    return true;
  }
  
  if (window[name] !== undefined) {
    window[name] = value;
    return true;
  }
  
  // Check in parameter dictionaries
  for (const paramDict of [advancedParams, lossConfigs]) {
    if (name in paramDict) {
      paramDict[name] = value;
      return true;
    }
  }
  
  return false;
}

// Initialize log queue
const logQueue = new LogQueue();

// Export all variables and functions
export {
  LogQueue,
  logQueue,
  stopTraining,
  getStopTraining,
  setStopTraining,
  currentBatchSize,
  getCurrentBatchSize,
  setCurrentBatchSize,
  defaultEpochs,
  defaultStepsPerEpoch,
  defaultLearningRate,
  defaultImageSize,
  ensureDirectoriesExist,
  supportedImageFormats,
  lossConfigs,
  advancedParams,
  getParameter,
  setParameter
};