// main.js (for browser)
// Make sure you have the TensorFlow.js CDN script in your index.html
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
// And no 'import * as tf from '@tensorflow/tfjs';' line here.

// Import utility functions from utils.js
import { 
    getTimestamp, 
    logMemoryUsage, 
    getConvLayer, // getConvLayer is now directly imported and used
    featureMSELoss,
    augmentImage, 
    extractRandomPatches, 
    loadImageAsTensor, 
    loadPatchesFromSingleImage, 
    displayTensorAsImage,
    visualizeFeatureMap,
    visualizeDifferenceHeatmap,
    saveSideBySideImage
} from './utils.js';


// Configuration
const EPOCHS = 3; 
const BATCH_SIZE = 4; 
const LEARNING_RATE = 1e-4;
const WEIGHT_DECAY = 1e-4; // Weight decay for L2 regularization

// Image size for both input and output of the lightweight feature extractor
// Doubled from 128 to 256
const IMAGE_SIZE = 256; 

// Paths for browser-based operations
const IMAGE_DATA_URL_PREFIX = './Set14/'; // You will need to provide images in a 'dataset' folder
const MODEL_NAME = 'lightweight_feature_extractor_model'; // Name for IndexedDB storage
const MODEL_LOAD_PATH = `indexeddb://${MODEL_NAME}`; // Path for loading model from IndexedDB
const MODEL_SAVE_PATH = `indexeddb://${MODEL_NAME}`; // Path for saving to IndexedDB

// UI elements
let statusElement;
let epochStatusElement;
let lossStatusElement;
let sampleContainer; // This will now point to the .sample-grid div
let saveModelBtn;
let loadModelInput; 
let startTrainingBtn;
let deleteModelBtn;
let stopTrainingBtn; // Added stop training button reference
let trainingTimeElement; 
let epochTimingElement; 

// Chart.js related elements
let lossChartCanvas;
let lossChart;
let lossData = [];
let epochLabels = [];

// Flag to control training interruption
let stopTrainingFlag = false;


// Backend selection is now handled internally
let currentBackend = ''; 

// Global flags for model configuration - now always true as requested
const ENABLE_DEPTHWISE_CONVS = true; 
const ENABLE_UNET_SKIP_CONNECTIONS = true; 

// Global models
let lightweightFeatureExtractor; // This is the model we will train
let fullVggFeatureExtractor;    // This is the pre-trained VGG19 acting as our target/teacher

// Helper to update UI status (kept in main.js as it directly manipulates UI)
function updateStatus(message) {
    if (statusElement) {
        statusElement.textContent = `Status: ${message}`;
    }
    console.log(`[${getTimestamp()}] ${message}`);
}


/**
 * Loads the pre-trained VGG19 model from local files and sets up a feature extractor.
 * It will consistently use 'block2_pool' as the feature extraction layer. This model acts as the ground truth for training.
 * With an IMAGE_SIZE of 256, 'block2_pool' should now output 64x64x128 features.
 * @returns {Promise<tf.LayersModel>} A TensorFlow.js LayersModel representing the VGG19 feature extractor.
 */
async function createFullVggFeatureExtractor() { // Renamed for clarity
    updateStatus('Loading pre-trained VGG19 model from local files (teacher model)...');
    const vgg19URL = './tfjs_vgg19_imagenet-master/model/model.json';
    // Layer name 'block2_pool' is correct for 4x spatial downsampling (256 -> 64)
    const layerName = 'block2_pool'; 

    try {
        const vgg = await tf.loadLayersModel(vgg19URL);
        console.log("Full VGG19 model loaded. Summary:");
        vgg.summary(); 

        const featureExtractorLayer = vgg.getLayer(layerName); 
        if (!featureExtractorLayer) {
            throw new Error(`Layer "${layerName}" not found in VGG19 model. Please check the layer name.`);
        }

        const model = tf.model({
            inputs: vgg.input,
            outputs: featureExtractorLayer.output
        });
        console.log(`VGG19 Feature Extractor model (output from ${layerName}). Summary:`);
        model.summary(); 
        model.trainable = false; // VGG model is not trained
        updateStatus(`VGG19 feature extractor for layer "${layerName}" loaded successfully.`);
        return model;
    } catch (error) {
        console.error('Error loading VGG19 model from local URL:', error);
        updateStatus(`Error loading VGG19 model: ${error.message}. Please ensure 'tfjs_vgg19_imagenet-master/model/model.json' is correctly placed.`);
        throw error; // Re-throw the error to stop training if VGG fails to load
    }
}


/**
 * Creates the lightweight feature extractor model.
 * This model is trained to mimic the output features of a specific VGG19 layer.
 * Its architecture includes depthwise convolutions and U-Net style skip connections as configured.
 * With IMAGE_SIZE = 256, the output shape will now be 64x64x128 to match VGG19's block2_pool.
 * @returns {tf.LayersModel} The compiled Keras-style model.
 */
function createLightweightFeatureExtractor() { // Renamed from createGeneratorModel
    updateStatus('Creating lightweight feature extractor model (student model)...');
    const input = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3] }); // 256x256x3

    // Encoder Path (Downsampling)
    // Pass WEIGHT_DECAY to getConvLayer
    let x = getConvLayer(tf, 32, 3, ENABLE_DEPTHWISE_CONVS, WEIGHT_DECAY).apply(input); // 256x256x32
    x = tf.layers.batchNormalization().apply(x); 
    
    // First pooling layer: 256 -> 128
    x = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(x); // 128x128x32

    // Second convolutional block
    x = getConvLayer(tf, 64, 3, ENABLE_DEPTHWISE_CONVS, WEIGHT_DECAY).apply(x); // 128x128x64
    x = tf.layers.batchNormalization().apply(x); 

    // Second pooling layer: 128 -> 64. This ensures the output matches the 64x64 target from VGG's block2_pool.
    x = tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }).apply(x); // 64x64x64

    // The output layer's filters should match the target VGG layer's filters (128)
    x = getConvLayer(tf, 128, 3, ENABLE_DEPTHWISE_CONVS, WEIGHT_DECAY).apply(x); // 64x64x128 (matches VGG's block2_pool output for 256x256 input)

    const model = tf.model({ inputs: input, outputs: x }); // Corrected output to be 'x'
    console.log("Lightweight Feature Extractor Model Summary:");
    model.summary(); 
    return model;
}


/**
 * Initializes the TensorFlow.js backend.
 * Prioritizes WebGPU, then WebGL, then CPU.
 */
async function initializeTfBackend() {
    // Try WebGPU first
    updateStatus(`Attempting to set backend to WEBGPU...`);
    try {
        const success = await tf.setBackend('webgpu');
        if (success) {
            currentBackend = 'webgpu';
            updateStatus(`Backend: WEBGPU.`);
            return;
        }
    } catch (error) {
        console.warn(`WebGPU initialization failed: ${error.message}. Falling back to WebGL.`);
    }

    // Try WebGL next
    updateStatus(`Attempting to set backend to WEBGL...`);
    try {
        const success = await tf.setBackend('webgl');
        if (success) {
            currentBackend = 'webgl';
            updateStatus(`Backend: WEBGL.`);
            return;
        }
    } catch (error) {
        console.warn(`WebGL initialization failed: ${error.message}. Falling back to CPU.`);
    }

    // Finally, try CPU
    updateStatus(`Attempting to set backend to CPU...`);
    try {
        const success = await tf.setBackend('cpu');
        if (success) {
            currentBackend = 'cpu';
            updateStatus(`Backend: CPU.`);
            return;
        }
    } catch (error) {
        console.error(`CPU backend initialization failed: ${error.message}. No suitable backend found.`);
        updateStatus(`Failed to set any suitable backend.`);
        throw new Error("No suitable TensorFlow.js backend found.");
    } finally {
        // Dispose and re-create models if a backend change actually happened during the process
        // This is primarily for a clean slate if backend changes mid-run, or if models were partially created.
        if (lightweightFeatureExtractor) {
            lightweightFeatureExtractor.dispose();
            lightweightFeatureExtractor = null;
        }
        if (fullVggFeatureExtractor) { 
            fullVggFeatureExtractor.dispose();
            fullVggFeatureExtractor = null;
        }
        if (epochStatusElement) epochStatusElement.textContent = 'Epoch: N/A';
        if (lossStatusElement) lossStatusElement.textContent = 'Loss: N/A';
        if (trainingTimeElement) trainingTimeElement.textContent = 'N/A';
        if (epochTimingElement) epochTimingElement.textContent = 'N/A'; 
    }
}

/**
 * Initializes the Chart.js loss curve chart.
 */
function initializeLossChart() {
    if (lossChart) {
        lossChart.destroy(); // Destroy existing chart if it exists
    }
    const ctx = lossChartCanvas.getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochLabels,
            datasets: [{
                label: 'Training Loss',
                data: lossData,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Updates the loss chart with new data.
 * @param {number} epoch - The current epoch number.
 * @param {number} loss - The loss value for the current epoch.
 */
function updateLossChart(epoch, loss) {
    epochLabels.push(`Epoch ${epoch}`);
    lossData.push(loss);
    lossChart.update(); // Update the chart
}

/**
 * Resets the loss chart data.
 */
function resetLossChart() {
    lossData = [];
    epochLabels = [];
    if (lossChart) {
        lossChart.destroy(); // Destroy existing chart
        initializeLossChart(); // Re-initialize with empty data
    }
}


/**
 * Runs the training process using batch-by-batch training.
 */
async function runTraining() {
    // Reset chart data before new training run
    resetLossChart();
    stopTrainingFlag = false; // Reset stop flag at the start of training

    // Disable relevant buttons and enable stop button
    startTrainingBtn.disabled = true;
    stopTrainingBtn.disabled = false;
    deleteModelBtn.disabled = true;
    saveModelBtn.disabled = true;
    loadModelInput.disabled = true;

    // Initialize the full VGG feature extractor (teacher model)
    if (!fullVggFeatureExtractor) {
        try {
            fullVggFeatureExtractor = await createFullVggFeatureExtractor(); 
        } catch (error) {
            updateStatus(`Failed to initialize Full VGG feature extractor. Training aborted.`);
            console.error('Failed to initialize Full VGG feature extractor:', error);
            startTrainingBtn.disabled = false; // Re-enable start button on failure
            stopTrainingBtn.disabled = true; // Disable stop button
            deleteModelBtn.disabled = false; // Re-enable delete button
            saveModelBtn.disabled = false; // Re-enable save button
            loadModelInput.disabled = false; // Re-enable load input
            return; 
        }
    }

    // Initialize or load the lightweight feature extractor (student model)
    if (!lightweightFeatureExtractor) { 
        updateStatus('Attempting to load lightweight feature extractor...');
        let loadedSuccessfully = false;

        // 1. Try to load from IndexedDB
        try {
            const loadedModel = await tf.loadLayersModel(MODEL_LOAD_PATH);
            if (loadedModel.input.shape[1] === IMAGE_SIZE && loadedModel.input.shape[2] === IMAGE_SIZE) {
                lightweightFeatureExtractor = loadedModel;
                updateStatus('Lightweight feature extractor loaded successfully from IndexedDB.');
                loadedSuccessfully = true;
            } else {
                updateStatus(`Loaded lightweight model input shape mismatch from IndexedDB. Will try local files or create new.`);
                loadedModel.dispose(); // Dispose mismatching model
            }
        } catch (e) {
            console.warn(`[${getTimestamp()}] No existing lightweight feature extractor found in IndexedDB or error loading: ${e.message}`);
            updateStatus('No existing model found in IndexedDB.');
        }

        // 2. If not loaded from IndexedDB, try to load from local files (downloads location, i.e., same folder as index.html)
        if (!loadedSuccessfully) {
            updateStatus('Attempting to load lightweight feature extractor from local files (model.json in root)...');
            try {
                // Assume the downloaded model files are in the same directory (e.g., 'model.json')
                // This path refers to the 'model.json' file in the root of the serving directory
                const loadedModel = await tf.loadLayersModel('./model.json'); 
                if (loadedModel.input.shape[1] === IMAGE_SIZE && loadedModel.input.shape[2] === IMAGE_SIZE) {
                    lightweightFeatureExtractor = loadedModel;
                    updateStatus('Lightweight feature extractor loaded successfully from local files.');
                    loadedSuccessfully = true;
                } else {
                    updateStatus(`Loaded lightweight model input shape mismatch from local files. Creating new model.`);
                    loadedModel.dispose(); // Dispose mismatching model
                }
            } catch (e) {
                console.warn(`[${getTimestamp()}] No lightweight feature extractor found in local files or error loading: ${e.message}`);
                updateStatus('No existing model found locally.');
            }
        }

        // 3. If still not loaded, create a new model
        if (!loadedSuccessfully) {
            updateStatus('Creating a new lightweight feature extractor model.');
            lightweightFeatureExtractor = createLightweightFeatureExtractor();
        }
    }

    lightweightFeatureExtractor.summary(); 
    logMemoryUsage(tf); 

    // Compile the lightweight feature extractor with Adam optimizer and L2 regularization
    lightweightFeatureExtractor.compile({ 
        optimizer: tf.train.adam(LEARNING_RATE), // Reverted to Adam
        loss: (yTrue, yPred) => featureMSELoss(tf, yTrue, yPred) 
    });
    updateStatus('Starting training for lightweight feature extractor...');

    // Example image files (replace with your actual dataset)
    const imageFiles = [
        'baboon.jpeg', 'barbara.jpeg', 'bridge.jpeg', 'coastguard.jpeg', 'comic.jpeg',
        'face.jpeg', 'flowers.jpeg', 'foreman.jpeg', 'lenna.jpeg', 'man.jpeg',
        'monarch.jpeg', 'pepper.jpeg', 'ppt3.jpeg', 'zebra.jpeg'
    ];

    const trainingStartTime = performance.now();
    let totalLoss = 0; 
    let totalBatchesProcessed = 0;

    try {
        // Training loop - epoch by epoch
        for (let epoch = 0; epoch < EPOCHS; epoch++) {
            if (stopTrainingFlag) {
                updateStatus(`Training stopped by user at epoch ${epoch + 1}.`);
                break; // Exit the epoch loop
            }

            updateStatus(`Starting epoch ${epoch + 1}/${EPOCHS}`);
            const epochStartTime = performance.now();
            let epochLoss = 0; 
            let epochBatches = 0; 

            imageFiles.sort(() => Math.random() - 0.5); 

            for (const file of imageFiles) {
                if (stopTrainingFlag) { // Check flag before processing each file
                    updateStatus(`Training stopped by user during epoch ${epoch + 1}.`);
                    break; 
                }

                const imageUrl = `${IMAGE_DATA_URL_PREFIX}${file}`;
                updateStatus(`Epoch ${epoch + 1}/${EPOCHS} - Processing: ${file}`);
                
                tf.engine().startScope(); 
                let imagePatches = null;
                try {
                    // Load multiple patches from one image
                    imagePatches = await loadPatchesFromSingleImage(tf, imageUrl, BATCH_SIZE * 5, IMAGE_SIZE, updateStatus, getTimestamp); 

                    if (imagePatches.shape[0] > 0) {
                        const numPatches = imagePatches.shape[0];
                        const numBatches = Math.ceil(numPatches / BATCH_SIZE);
                        
                        for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
                            if (stopTrainingFlag) { // Check flag before processing each batch
                                break; 
                            }
                            const startIdx = batchIdx * BATCH_SIZE;
                            const endIdx = Math.min(startIdx + BATCH_SIZE, numPatches);
                            
                            const inputBatch = imagePatches.slice([startIdx, 0, 0, 0], [endIdx - startIdx, -1, -1, -1]);
                            
                            try {
                                // Get target features from the full VGG model
                                const targetFeatures = fullVggFeatureExtractor.predict(inputBatch);
                                
                                // Train the lightweight model
                                const result = await lightweightFeatureExtractor.trainOnBatch(inputBatch, targetFeatures);
                                const loss = Array.isArray(result) ? result[0] : result;
                                
                                epochLoss += loss;
                                epochBatches++;
                                totalBatchesProcessed++;
                                
                                inputBatch.dispose();
                                targetFeatures.dispose();
                                
                            } catch (error) {
                                console.error(`Error training on batch for ${file}, batch ${batchIdx}:`, error);
                                if (inputBatch instanceof tf.Tensor) inputBatch.dispose();
                            }
                        }
                    } else {
                        console.warn(`[${getTimestamp()}] No valid patches from ${file} for training in epoch ${epoch + 1}.`);
                    }
                } catch (imageProcessingError) {
                    console.error(`Error processing image ${file} for epoch ${epoch + 1}:`, imageProcessingError);
                } finally {
                    if (imagePatches instanceof tf.Tensor) imagePatches.dispose();
                    tf.engine().endScope(); 
                    logMemoryUsage(tf); 
                }
            }

            // End of epoch
            const epochEndTime = performance.now();
            const epochDuration = ((epochEndTime - epochStartTime) / 1000).toFixed(2);
            const avgEpochLoss = epochBatches > 0 ? epochLoss / epochBatches : 0;
            
            epochStatusElement.textContent = `Epoch: ${epoch + 1}/${EPOCHS}`;
            lossStatusElement.textContent = `Loss: ${avgEpochLoss.toFixed(6)}`;
            epochTimingElement.textContent = `${epochDuration} seconds`;
            updateStatus(`Epoch ${epoch + 1}/${EPOCHS} completed: Avg Loss = ${avgEpochLoss.toFixed(6)}, Time = ${epochDuration}s`);
            
            totalLoss += avgEpochLoss; 
            
            // Update the loss chart after each epoch
            updateLossChart(epoch + 1, avgEpochLoss);

            // Save the model every 10 epochs
            if (!stopTrainingFlag && (epoch + 1) % 10 === 0) { // Only save if training is not stopped
                updateStatus(`Saving model at epoch ${epoch + 1}...`);
                try {
                    // Ensure TensorFlow.js is ready and backend is initialized before saving
                    await tf.ready(); 
                    let originalBackend = tf.getBackend(); // Store current backend
                    try {
                        // Temporarily set backend to CPU for saving, as WebGPU might have issues
                        await tf.setBackend('cpu');
                        console.log(`[${getTimestamp()}] Switched backend to CPU for saving model checkpoint.`);
                        await lightweightFeatureExtractor.save(MODEL_SAVE_PATH);
                        updateStatus(`Model saved to IndexedDB at epoch ${epoch + 1}.`);
                    } finally {
                        // Switch back to the original backend if it was different
                        if (tf.getBackend() !== originalBackend) {
                            await tf.setBackend(originalBackend);
                            console.log(`[${getTimestamp()}] Switched backend back to ${originalBackend} after saving checkpoint.`);
                        }
                    }
                } catch (error) {
                    updateStatus(`Error saving model at epoch ${epoch + 1}: ${error.message}`);
                    console.error('Error saving model during training:', error);
                }
            }


            // --- Sample Visualization Logic ---
            tf.engine().startScope(); 
            try {
                const sampleImageFile = 'lenna.jpeg'; 
                const sampleImageUrl = `${IMAGE_DATA_URL_PREFIX}${sampleImageFile}`;

                let originalSampleInput = null; 
                let lightweightModelSampleOutput = null; 

                try {
                    originalSampleInput = await loadImageAsTensor(tf, sampleImageUrl, IMAGE_SIZE, updateStatus, getTimestamp); 
                    if (originalSampleInput) {
                        // Clear previous samples before adding new ones
                        sampleContainer.innerHTML = ''; 
                        // Get the features from the lightweight model for visualization
                        lightweightModelSampleOutput = lightweightFeatureExtractor.predict(originalSampleInput.expandDims(0)); 

                        await saveSideBySideImage(tf, originalSampleInput.expandDims(0), lightweightModelSampleOutput, `Epoch ${epoch + 1} Sample`, sampleContainer, fullVggFeatureExtractor, getTimestamp);
                    } else {
                        console.warn(`[${getTimestamp()}] Could not load sample image "${sampleImageFile}". Skipping sample visualization for this epoch.`);
                    }
                } catch (sampleError) {
                    console.error(`[${getTimestamp()}] Error during sample visualization for epoch ${epoch + 1}:`, sampleError);
                } finally {
                    if (originalSampleInput instanceof tf.Tensor) originalSampleInput.dispose();
                    if (lightweightModelSampleOutput instanceof tf.Tensor) lightweightModelSampleOutput.dispose(); 
                }

            } finally {
                tf.engine().endScope(); 
                logMemoryUsage(tf); 
            }
            // --- End of Sample Visualization Logic ---
        }

        // Training completed (or stopped by user)
        const trainingEndTime = performance.now();
        const totalTrainingDuration = ((trainingEndTime - trainingStartTime) / 1000).toFixed(2);
        trainingTimeElement.textContent = `${totalTrainingDuration} seconds`;
        
        const avgTotalLoss = totalBatchesProcessed > 0 ? totalLoss / EPOCHS : 0; // Average over epochs
        updateStatus(`Training completed! Avg Loss: ${avgTotalLoss.toFixed(6)}, Total Time: ${totalTrainingDuration}s`);

        // Final save after training completes, if not already saved on a 10th epoch AND not stopped by user
        if (!stopTrainingFlag && EPOCHS % 10 !== 0) {
            updateStatus(`Saving final model...`);
            try {
                // Ensure TensorFlow.js is ready and backend is initialized before saving
                await tf.ready(); 
                let originalBackend = tf.getBackend(); // Store current backend
                try {
                    // Temporarily set backend to CPU for saving, as WebGPU might have issues
                    await tf.setBackend('cpu');
                    console.log(`[${getTimestamp()}] Switched backend to CPU for final model save.`);
                    await lightweightFeatureExtractor.save(MODEL_SAVE_PATH);
                    updateStatus(`Final model saved to IndexedDB: ${MODEL_NAME}.`);
                } finally {
                    // Switch back to the original backend if it was different
                    if (tf.getBackend() !== originalBackend) {
                        await tf.setBackend(originalBackend);
                        console.log(`[${getTimestamp()}] Switched backend back to ${originalBackend} after final save.`);
                    }
                }
            } catch (error) {
                updateStatus(`Error saving final model: ${error.message}`);
                console.error('Error saving final model:', error);
            }
        }

    } catch (error) {
        updateStatus(`Training aborted due to error: ${error.message}`);
        console.error('Training error:', error);
    } finally {
        // Always re-enable/disable buttons appropriately when training finishes or stops
        startTrainingBtn.disabled = false;
        stopTrainingBtn.disabled = true;
        deleteModelBtn.disabled = false;
        saveModelBtn.disabled = false;
        loadModelInput.disabled = false;
        // Dispose all variables in the global tf.engine
        tf.disposeVariables(); 
        logMemoryUsage(tf);
    }
}

// Function to handle manual model saving (e.g., to file for download)
async function saveModelToFile() {
    if (lightweightFeatureExtractor) {
        updateStatus('Saving lightweight model to file...');
        let originalBackend = tf.getBackend(); // Store current backend
        let modelToSave = lightweightFeatureExtractor; // Assume we save the existing model

        try {
            // Ensure TensorFlow.js is ready
            await tf.ready(); 

            // If the current backend is not CPU, switch to CPU for saving
            // Also clone the model to ensure all tensors are on CPU and ready for serialization
            if (originalBackend !== 'cpu') {
                await tf.setBackend('cpu');
                console.log(`[${getTimestamp()}] Switched backend to CPU for manual save.`);
                
                // Clone the model to ensure all weights are on CPU.
                // This is a more robust way to handle potential WebGPU tensor access issues during saving.
                modelToSave = tf.model({
                    inputs: lightweightFeatureExtractor.inputs,
                    outputs: lightweightFeatureExtractor.outputs
                });
                modelToSave.setWeights(lightweightFeatureExtractor.getWeights());
                console.log(`[${getTimestamp()}] Cloned model to CPU for saving.`);
            }

            // Perform the save operation with the CPU-ready model
            await modelToSave.save('downloads://lightweight-feature-extractor'); 
            updateStatus('Lightweight model downloaded.');

        } catch (error) {
            updateStatus(`Error downloading lightweight model: ${error.message}.`);
            console.error('Error downloading lightweight model:', error);
        } finally {
            // Dispose the temporary cloned model if it was created
            if (modelToSave !== lightweightFeatureExtractor) {
                modelToSave.dispose();
                console.log(`[${getTimestamp()}] Disposed temporary CPU model clone.`);
            }
            // Always switch back to the original backend if it was different
            if (tf.getBackend() !== originalBackend) {
                await tf.setBackend(originalBackend);
                console.log(`[${getTimestamp()}] Switched backend back to ${originalBackend} after manual save.`);
            }
        }
    } else {
        updateStatus('No lightweight model to save. Train a model first.');
    }
}

// Function to handle manual model loading from file input
async function loadModelFromFile(event) {
    const files = event.target.files;
    if (files.length > 0) {
        updateStatus('Loading lightweight model from files...');
        try {
            // Ensure TF.js is ready and backend is set before loading
            await tf.ready();
            let originalBackend = tf.getBackend();
            try {
                // It might be safer to load on CPU first to prevent backend-specific issues
                await tf.setBackend('cpu');
                if (lightweightFeatureExtractor) lightweightFeatureExtractor.dispose(); 
                lightweightFeatureExtractor = await tf.loadLayersModel(tf.io.browserFiles(files));
                updateStatus('Lightweight model loaded from files successfully.');
                lightweightFeatureExtractor.summary();
                logMemoryUsage(tf); 
            } finally {
                 // Switch back to the original backend if it was different
                if (tf.getBackend() !== originalBackend) {
                    await tf.setBackend(originalBackend);
                }
            }
        } catch (error) {
            updateStatus(`Error loading lightweight model from files: ${error.message}`);
            console.error('Error loading lightweight model from files:', error);
        }
    }
}

// Function to delete the model from IndexedDB
async function deleteModel() {
    updateStatus('Attempting to delete lightweight model from IndexedDB...');
    try {
        await tf.ready(); 
        let originalBackend = tf.getBackend();
        try {
            await tf.setBackend('cpu'); // Perform delete on CPU backend for stability
            await tf.io.removeModel(MODEL_SAVE_PATH); 
            
            if (lightweightFeatureExtractor) {
                lightweightFeatureExtractor.dispose(); 
                lightweightFeatureExtractor = null;
            }
            if (fullVggFeatureExtractor) { 
                fullVggFeatureExtractor.dispose();
                fullVggFeatureExtractor = null;
            }
            updateStatus('Lightweight model deleted successfully from IndexedDB. Ready for a new start.');
            epochStatusElement.textContent = 'Epoch: N/A';
            lossStatusElement.textContent = 'N/A'; 
            trainingTimeElement.textContent = 'N/A'; 
            epochTimingElement.textContent = 'N/A'; 
            logMemoryUsage(tf); 
        } finally {
            if (tf.getBackend() !== originalBackend) {
                await tf.setBackend(originalBackend);
            }
        }
    } catch (error) {
        if (error.message.includes('Cannot find model with path')) {
            updateStatus('No existing lightweight model found in IndexedDB to delete.');
            console.warn('No existing lightweight model found in IndexedDB to delete:', error.message);
        } else {
            updateStatus(`Error deleting lightweight model: ${error.message}`);
            console.error('Error deleting lightweight model:', error);
        }
    }
}

/**
 * Recreates the models (lightweight feature extractor and full VGG if needed)
 * when a configuration change happens.
 */
function recreateModelsOnConfigChange() {
    if (lightweightFeatureExtractor) {
        lightweightFeatureExtractor.dispose();
        lightweightFeatureExtractor = null; 
    }
    // Full VGG is reloaded as needed, so no explicit recreation here unless its source changes
    updateStatus('Configuration changed. Models will be recreated on next training run.');
    logMemoryUsage(tf); 
}


// Event Listeners for UI buttons
document.addEventListener('DOMContentLoaded', async () => { 
    statusElement = document.getElementById('status');
    epochStatusElement = document.getElementById('epoch-status');
    lossStatusElement = document.getElementById('loss-status');
    // Ensure sampleContainer targets the .sample-grid div inside #sample-images
    sampleContainer = document.getElementById('sample-images').querySelector('.sample-grid'); 
    saveModelBtn = document.getElementById('save-model-btn');
    loadModelInput = document.getElementById('load-model-input'); 
    startTrainingBtn = document.getElementById('start-training-btn');
    deleteModelBtn = document.getElementById('delete-model-btn');
    stopTrainingBtn = document.getElementById('stop-training-btn'); // Get reference to the new button
    trainingTimeElement = document.getElementById('training-time'); 
    epochTimingElement = document.getElementById('epoch-time'); 

    // Get the chart canvas element
    lossChartCanvas = document.getElementById('lossChart');
    // Initialize the chart
    initializeLossChart();

    // Initialize backend to WebGPU on start, with fallbacks
    await initializeTfBackend(); 

    // Add event listeners
    if (startTrainingBtn) {
        startTrainingBtn.addEventListener('click', runTraining);
    } else {
        console.error("Element with ID 'start-training-btn' not found.");
    }
    if (saveModelBtn) {
        saveModelBtn.addEventListener('click', saveModelToFile);
    } else {
        console.error("Element with ID 'save-model-btn' not found.");
    }
    if (loadModelInput) {
        loadModelInput.addEventListener('change', loadModelFromFile);
    } else {
        console.error("Element with ID 'load-model-input' not found.");
    }
    if (deleteModelBtn) { 
        deleteModelBtn.addEventListener('click', deleteModel);
    } else {
        console.error("Element with ID 'delete-model-btn' not found.");
    }
    if (stopTrainingBtn) { // Add listener for the new stop button
        stopTrainingBtn.addEventListener('click', () => {
            stopTrainingFlag = true;
            updateStatus('Stopping training requested by user...');
        });
        stopTrainingBtn.disabled = true; // Initially disabled
    } else {
        console.error("Element with ID 'stop-training-btn' not found.");
    }

    updateStatus('Ready to start training the lightweight feature extractor.');
    logMemoryUsage(tf); 
});
