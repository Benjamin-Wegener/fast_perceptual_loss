// main.js
// Main application entry point for the Fast Perceptual Loss Trainer

import {
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
  ensureDirectoriesExist
} from './globals.js';

import { createFastPerceptualModel } from './model.js';
import { trainFastPerceptualModel } from './training_core.js';
import { createTrainingDataset } from './training_data.js';
import {
  updateStatsChart,
  renderTensorToCanvas,
  createModelVisualization,
  AdvancedVisualizationCallback,
  formatTime,
  updateETA
} from './training_ui.js';

// DOM Elements
let startButton;
let stopButton;
let batchSizeInput;
let batchDecreaseBtn;
let batchIncreaseBtn;
let datasetInput;
let datasetInfo;
let progressBar;
let logTextArea;
let copyLogBtn;
let currentLrText;
let etaText;
let inputCanvas;
let targetCanvas;
let predCanvas;
let diffCanvas;
let lossChart;
let lrChart;
let epochsInput;
let stepsInput;
let learningRateInput;
let useSwaCheckbox;
let toggleAdvancedBtn;
let advancedParamsDiv;
let statusIndicator;
let statusText;

// Global state
let model = null;
let featureExtractor = null;
let trainingDataset = [];
let isTraining = false;
let currentEpoch = 0;
let epochTimes = [];

// Initialize application
async function init() {
  // First, make sure all DOM elements are initialized
  initializeUI();
  
  // Then set up log queue listener
  logQueue.addListener((message) => {
    if (logTextArea) {  // Make sure the element exists before using it
      const timestamp = new Date().toLocaleTimeString();
      logTextArea.innerHTML += `\n[${timestamp}] ${message}`;
      logTextArea.scrollTop = logTextArea.scrollHeight;
    }
  });
  
  // Initialize TensorFlow.js
  await tf.ready();
  logQueue.put("TensorFlow.js ready");
  
  // Initialize the rest of the application
  setupEventListeners();
  initializeCharts();
  
  // Make sure storage is available
  ensureDirectoriesExist();
  
  updateStatus('ready', 'Ready to start');
}

// Initialize UI components
function initializeUI() {
  // Get DOM elements
  startButton = document.getElementById('start-training');
  stopButton = document.getElementById('stop-training');
  batchSizeInput = document.getElementById('batch-size');
  batchDecreaseBtn = document.getElementById('batch-decrease');
  batchIncreaseBtn = document.getElementById('batch-increase');
  datasetInput = document.getElementById('dataset-input');
  datasetInfo = document.getElementById('dataset-info');
  progressBar = document.getElementById('progress-bar');
  logTextArea = document.getElementById('log-text');
  copyLogBtn = document.getElementById('copy-log');
  currentLrText = document.getElementById('current-lr');
  etaText = document.getElementById('eta');
  inputCanvas = document.getElementById('input-canvas');
  targetCanvas = document.getElementById('target-canvas');
  predCanvas = document.getElementById('pred-canvas');
  diffCanvas = document.getElementById('diff-canvas');
  epochsInput = document.getElementById('epochs');
  stepsInput = document.getElementById('steps-per-epoch');
  learningRateInput = document.getElementById('learning-rate');
  useSwaCheckbox = document.getElementById('use-swa');
  toggleAdvancedBtn = document.getElementById('toggle-advanced');
  advancedParamsDiv = document.getElementById('advanced-params');
  statusIndicator = document.querySelector('.status-indicator');
  statusText = document.getElementById('status-text');
  
  // Set default values
  if (batchSizeInput) batchSizeInput.value = getCurrentBatchSize();
  if (epochsInput) epochsInput.value = defaultEpochs;
  if (stepsInput) stepsInput.value = defaultStepsPerEpoch;
  if (learningRateInput) learningRateInput.value = defaultLearningRate;
}

// Set up event listeners
function setupEventListeners() {
  // Start training button
  if (startButton) {
    startButton.addEventListener('click', startTraining);
  }
  
  // Stop training button
  if (stopButton) {
    stopButton.addEventListener('click', stopTrainingHandler);
  }
  
  // Batch size controls
  if (batchDecreaseBtn && batchSizeInput) {
    batchDecreaseBtn.addEventListener('click', () => {
      const currentValue = parseInt(batchSizeInput.value);
      if (currentValue > 1) {
        batchSizeInput.value = currentValue - 1;
        setCurrentBatchSize(currentValue - 1);
        logQueue.put(`Batch size decreased to ${currentValue - 1}`);
      }
    });
  }
  
  if (batchIncreaseBtn && batchSizeInput) {
    batchIncreaseBtn.addEventListener('click', () => {
      const currentValue = parseInt(batchSizeInput.value);
      batchSizeInput.value = currentValue + 1;
      setCurrentBatchSize(currentValue + 1);
      logQueue.put(`Batch size increased to ${currentValue + 1}`);
    });
  }
  
  // Dataset input change
  if (datasetInput) {
    datasetInput.addEventListener('change', handleDatasetSelection);
  }
  
  // Copy log button
  if (copyLogBtn && logTextArea) {
    copyLogBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(logTextArea.textContent);
      logQueue.put("Log content copied to clipboard");
    });
  }
  
  // Toggle advanced parameters
  if (toggleAdvancedBtn && advancedParamsDiv) {
    toggleAdvancedBtn.addEventListener('click', () => {
      if (advancedParamsDiv.classList.contains('hidden')) {
        advancedParamsDiv.classList.remove('hidden');
        toggleAdvancedBtn.textContent = '▼ Hide Advanced Parameters';
      } else {
        advancedParamsDiv.classList.add('hidden');
        toggleAdvancedBtn.textContent = '▶ Show Advanced Parameters';
      }
    });
  }
}

// Initialize charts
function initializeCharts() {
  if (!document.getElementById('loss-chart') || !document.getElementById('lr-chart')) {
    console.error('Chart elements not found');
    return;
  }

  const lossCtx = document.getElementById('loss-chart').getContext('2d');
  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Loss',
        data: [],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
  });
  
  const lrCtx = document.getElementById('lr-chart').getContext('2d');
  lrChart = new Chart(lrCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Learning Rate',
        data: [],
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          type: 'logarithmic'
        }
      }
    }
  });
}

// Handle dataset selection
async function handleDatasetSelection(event) {
  const files = event.target.files;
  
  if (files.length === 0) {
    datasetInfo.textContent = "No files selected";
    return;
  }
  
  datasetInfo.textContent = `Selected ${files.length} files`;
  updateStatus('loading', 'Loading dataset...');
  
  try {
    // Clear previous dataset
    disposeTensorDataset();
    
    // Create new dataset
    trainingDataset = await createTrainingDataset(Array.from(files), defaultImageSize);
    
    logQueue.put(`Dataset created with ${trainingDataset.length} images`);
    updateStatus('ready', `Dataset loaded with ${trainingDataset.length} images`);
    
    // Initialize visualization with a sample
    if (trainingDataset.length > 0) {
      const sampleImage = trainingDataset[0];
      await renderTensorToCanvas(sampleImage, inputCanvas);
    }
  } catch (error) {
    logQueue.put(`Error loading dataset: ${error.message}`);
    updateStatus('error', 'Error loading dataset');
  }
}

// Update status indicator
function updateStatus(status, message) {
  if (statusText) statusText.textContent = message;
  if (statusIndicator) {
    statusIndicator.className = 'status-indicator';
    
    switch(status) {
      case 'ready':
        statusIndicator.classList.add('status-ready');
        break;
      case 'loading':
        statusIndicator.classList.add('status-loading');
        break;
      case 'error':
        statusIndicator.classList.add('status-error');
        break;
    }
  }
}

// Update progress bar
function updateProgress(percentage) {
  if (progressBar) {
    progressBar.style.width = `${percentage}%`;
    progressBar.textContent = `${percentage.toFixed(1)}%`;
  }
}

// Update learning rate display
function updateLrDisplay(lr) {
  if (currentLrText) {
    currentLrText.textContent = `Current LR: ${lr.toFixed(6)}`;
  }
}

// Update ETA display
function updateEtaDisplay(epoch, totalEpochs, epochTime) {
  if (!etaText) return;
  
  // Add to epoch times array
  epochTimes.push(epochTime);
  
  // Keep only last 5 epoch times for better estimate
  if (epochTimes.length > 5) {
    epochTimes.shift();
  }
  
  const etaString = updateETA(epoch, totalEpochs, epochTimes);
  etaText.textContent = `ETA: ${etaString}`;
}

// Initialize models
async function initializeModels() {
  updateStatus('loading', 'Initializing models...');
  
  try {
    // Create the fast perceptual model
    model = createFastPerceptualModel([null, null, 3]);
    logQueue.put("FastPerceptualLoss model created");
    
    // Load VGG19 as feature extractor (instead of MobileNet)
    logQueue.put("Loading VGG19 as feature extractor...");
    
    // VGG19 URL - TensorFlow.js pre-trained model
    featureExtractor = await tf.loadLayersModel('VGG19/model/model.json');
    
    // Create a sub-model with just the feature extraction layers
    // VGG19 typically uses block4_conv2 or block5_conv2 for perceptual features
    const layer = featureExtractor.getLayer('block4_conv2');
    featureExtractor = tf.model({
      inputs: featureExtractor.inputs,
      outputs: layer.output
    });
    
    logQueue.put("VGG19 feature extractor loaded successfully");
    updateStatus('ready', 'Models initialized');
    
    return true;
  } catch (error) {
    logQueue.put(`Error initializing models: ${error.message}`);
    updateStatus('error', 'Error initializing models');
    return false;
  }
}

// Start training process
async function startTraining() {
  // Check if dataset is loaded
  if (!trainingDataset || trainingDataset.length === 0) {
    logQueue.put("Please load a dataset first");
    return;
  }
  
  // Check if already training
  if (isTraining) {
    logQueue.put("Training already in progress");
    return;
  }
  
  isTraining = true;
  setStopTraining(false);
  
  // Update UI
  if (startButton) startButton.disabled = true;
  if (stopButton) stopButton.disabled = false;
  
  // Reset progress
  updateProgress(0);
  if (etaText) etaText.textContent = "ETA: --:--:--";
  epochTimes = [];
  
  // Initialize models if not already done
  if (!model || !featureExtractor) {
    const success = await initializeModels();
    if (!success) {
      isTraining = false;
      if (startButton) startButton.disabled = false;
      if (stopButton) stopButton.disabled = true;
      return;
    }
  }
  
  // Get training parameters
  const epochs = parseInt(epochsInput.value) || defaultEpochs;
  const stepsPerEpoch = parseInt(stepsInput.value) || defaultStepsPerEpoch;
  const initialLr = parseFloat(learningRateInput.value) || defaultLearningRate;
  const useSwa = useSwaCheckbox.checked;
  
  // Clear charts
  if (lossChart) {
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();
  }
  
  if (lrChart) {
    lrChart.data.labels = [];
    lrChart.data.datasets[0].data = [];
    lrChart.update();
  }
  
  // Setup callbacks for visualization
  const visualizationCallback = new AdvancedVisualizationCallback(
    trainingDataset,
    featureExtractor,
    {
      inputCanvas,
      targetCanvas,
      predCanvas,
      diffCanvas
    }
  );
  
  // Initialize visualization
  visualizationCallback.initialize();
  
  try {
    // Start training
    await trainFastPerceptualModel({
      model,
      featureExtractor,
      dataset: trainingDataset,
      batchSizeVar: batchSizeInput,
      epochs,
      stepsPerEpoch,
      initialLr,
      visualizationCallbacks: {
        visualize: (model, featureExtractor, epoch) => 
          visualizationCallback.onEpochEnd(epoch, model)
      },
      progressCallbacks: {
        updateProgress,
        updateLr: updateLrDisplay,
        updateETA: updateEtaDisplay,
        updateCharts: (epochs, lossValues, lrValues) => 
          updateStatsChart(lossChart, lrChart, epochs, lossValues, lrValues)
      },
      useSwa,
      weightDecay: parseFloat(document.getElementById('weight-decay').value) || 0.01
    });
    
    logQueue.put("Training completed");
  } catch (error) {
    logQueue.put(`Training error: ${error.message}`);
    console.error("Training error:", error);
  } finally {
    // Reset UI
    isTraining = false;
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    updateStatus('ready', 'Training completed');
  }
}

// Stop training handler
function stopTrainingHandler() {
  if (!isTraining) {
    return;
  }
  
  logQueue.put("Stopping training...");
  setStopTraining(true);
  if (stopButton) stopButton.disabled = true;
}

// Clean up tensors when dataset changes
function disposeTensorDataset() {
  if (trainingDataset && trainingDataset.length > 0) {
    trainingDataset.forEach(tensor => {
      if (tensor && tensor.dispose) {
        tensor.dispose();
      }
    });
    trainingDataset = [];
  }
}

// Clean up all tensors and resources
function cleanup() {
  disposeTensorDataset();
  
  if (model) {
    model.dispose();
    model = null;
  }
  
  if (featureExtractor) {
    featureExtractor.dispose();
    featureExtractor = null;
  }
  
  // Clear memory
  tf.disposeVariables();
  tf.engine().purgeUnusedTensors();
}

// Handle application exit
window.addEventListener('beforeunload', cleanup);

// Wait for DOM to be fully loaded before initializing
document.addEventListener('DOMContentLoaded', init);

// Export key functions for potential external use
export {
  startTraining,
  stopTrainingHandler,
  updateProgress,
  updateLrDisplay,
  updateEtaDisplay,
  cleanup
};