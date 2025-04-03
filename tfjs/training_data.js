// training_data.js
// Data loading and processing functions for the Fast Perceptual Loss model

import { logQueue, supportedImageFormats, defaultImageSize } from './globals.js';

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

// Load a single image as a tensor
async function loadImageAsTensor(imageFile) {
  return new Promise((resolve, reject) => {
    try {
      const img = new Image();
      const imageUrl = URL.createObjectURL(imageFile);
      
      img.onload = () => {
        // Convert image to tensor
        const tensor = tf.tidy(() => {
          // Create a tensor from the image
          const imageTensor = tf.browser.fromPixels(img);
          
          // Normalize to [0, 1]
          const normalized = tf.div(imageTensor, 255);
          
          // Clean up URL
          URL.revokeObjectURL(imageUrl);
          
          return normalized;
        });
        
        resolve(tensor);
      };
      
      img.onerror = (error) => {
        URL.revokeObjectURL(imageUrl);
        reject(new Error(`Failed to load image: ${error.message}`));
      };
      
      img.src = imageUrl;
    } catch (error) {
      reject(error);
    }
  });
}

// Create training dataset from uploaded files
async function createTrainingDataset(imageFiles, targetSize = defaultImageSize) {
  try {
    // Filter only supported image formats
    const validFiles = imageFiles.filter(file => {
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return supportedImageFormats.includes(extension);
    });
    
    if (validFiles.length === 0) {
      logQueue.put("Error: No valid image files found");
      throw new Error("No valid image files found");
    }
    
    logQueue.put(`Found ${validFiles.length} images for training`);
    
    // Load images as tensors
    const tensors = [];
    for (const file of validFiles) {
      try {
        const tensor = await loadImageAsTensor(file);
        
        // Extract random patch and resize to target size
        const processedTensor = tf.tidy(() => {
          // Get dimensions
          const [height, width] = tensor.shape.slice(0, 2);
          
          // If image is too small, resize it
          let resizedTensor = tensor;
          const extractionSize = Math.floor(targetSize * 1.5);
          
          if (height < extractionSize || width < extractionSize) {
            // Calculate scale factor
            const scale = extractionSize / Math.min(height, width);
            const newHeight = Math.floor(height * scale);
            const newWidth = Math.floor(width * scale);
            
            // Resize the image
            resizedTensor = tf.image.resizeBilinear(tensor, [newHeight, newWidth]);
          }
          
          // Get random patch
          const [newHeight, newWidth] = resizedTensor.shape.slice(0, 2);
          const hStartMax = newHeight - extractionSize;
          const wStartMax = newWidth - extractionSize;
          
          const hStart = Math.floor(Math.random() * (hStartMax + 1));
          const wStart = Math.floor(Math.random() * (wStartMax + 1));
          
          // Extract patch
          const patch = tf.slice(
            resizedTensor,
            [hStart, wStart, 0],
            [extractionSize, extractionSize, 3]
          );
          
          // Final resize to target size
          return tf.image.resizeBilinear(patch, [targetSize, targetSize]);
        });
        
        tensors.push(processedTensor);
        
        // Dispose the original tensor
        tensor.dispose();
      } catch (error) {
        logQueue.put(`Error processing image ${file.name}: ${error.message}`);
      }
    }
    
    logQueue.put(`Successfully processed ${tensors.length} images`);
    return tensors;
  } catch (error) {
    logQueue.put(`Error creating dataset: ${error.message}`);
    throw error;
  }
}

// Augment an image tensor with random transformations
function augmentImage(imageTensor) {
  return tf.tidy(() => {
    // Make a copy to avoid modifying the original
    let augmented = imageTensor.clone();
    
    // Random brightness adjustment
    const brightnessAdjustment = tf.randomUniform([], -0.2, 0.2);
    augmented = tf.add(augmented, brightnessAdjustment);
    
    // Random contrast adjustment
    const contrastFactor = tf.randomUniform([], 0.8, 1.2);
    const mean = tf.mean(augmented, [0, 1], true);
    augmented = tf.add(tf.mul(augmented.sub(mean), contrastFactor), mean);
    
    // Random left-right flip with 50% probability
    if (Math.random() > 0.5) {
      augmented = tf.reverse(augmented, 1);
    }
    
    // Random up-down flip with 50% probability
    if (Math.random() > 0.5) {
      augmented = tf.reverse(augmented, 0);
    }
    
    // Ensure values are clipped to [0, 1]
    augmented = tf.clipByValue(augmented, 0, 1);
    
    return augmented;
  });
}

// Apply MixUp augmentation to a batch of images
function applyMixup(images, alpha = 0.2) {
  return tf.tidy(() => {
    // Create a shuffled version for the second batch
    const indices = Array.from(Array(images.shape[0]).keys());
    const shuffledIndices = tf.util.shuffle(indices);
    const shuffledImages = tf.gather(images, shuffledIndices);
    
    // Random mixup factor between alpha and 1-alpha
    const lambda = tf.randomUniform([], alpha, 1.0 - alpha);
    
    // Apply mixup
    return tf.add(
      tf.mul(images, lambda),
      tf.mul(shuffledImages, tf.sub(tf.scalar(1.0), lambda))
    );
  });
}

// Advanced data augmentation with CutMix
function applyCutMix(images, alpha = 1.0) {
  return tf.tidy(() => {
    const batchSize = images.shape[0];
    const height = images.shape[1];
    const width = images.shape[2];
    
    // Generate random parameters for the cut rectangle
    const lam = tf.randomUniform([], 0, 1, 'float32').dataSync()[0];
    
    // Calculate cut size
    const cutRatio = Math.sqrt(1.0 - lam);
    const cutH = Math.floor(height * cutRatio);
    const cutW = Math.floor(width * cutRatio);
    
    // Calculate center position
    const cx = Math.floor(Math.random() * width);
    const cy = Math.floor(Math.random() * height);
    
    // Calculate box boundaries (clip to image edges)
    const x1 = Math.max(0, cx - Math.floor(cutW / 2));
    const y1 = Math.max(0, cy - Math.floor(cutH / 2));
    const x2 = Math.min(width, cx + Math.floor(cutW / 2));
    const y2 = Math.min(height, cy + Math.floor(cutH / 2));
    
    // Create mask for the cut region
    const mask = tf.buffer([height, width, 1], 'float32');
    
    // Fill mask with 1s in the cut region
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) {
        mask.set(1, y, x, 0);
      }
    }
    
    const maskTensor = mask.toTensor();
    const expandedMask = tf.tile(maskTensor, [1, 1, 3]);
    
    // Generate shuffled indices
    const shuffledIndices = tf.util.shuffle([...Array(batchSize).keys()]);
    const shuffledImages = tf.gather(images, shuffledIndices);
    
    // Apply CutMix
    return tf.add(
      tf.mul(images, tf.sub(tf.scalar(1), expandedMask)),
      tf.mul(shuffledImages, expandedMask)
    );
  });
}

// Process samples through feature extractor
async function processSamplesForFeatureExtractor(dataBatch, featureExtractor) {
  return tf.tidy(() => {
    // VGG19 preprocessing for the data batch
    const preprocessedBatch = preprocessForVGG19(dataBatch);
    
    // Pass the batch through feature extractor to get features
    const features = featureExtractor.predict(preprocessedBatch);
    
    // Return input images and their extracted features as targets
    return { xs: preprocessedBatch, ys: features };
  });
}

// Create a batch from dataset with augmentation
function createAugmentedBatch(dataset, batchSize) {
  return tf.tidy(() => {
    const indices = [];
    const datasetSize = dataset.length;
    
    // Select random indices
    for (let i = 0; i < batchSize; i++) {
      indices.push(Math.floor(Math.random() * datasetSize));
    }
    
    // Gather selected images
    const selectedImages = indices.map(i => dataset[i]);
    
    // Apply individual augmentations
    const augmentedImages = selectedImages.map(img => augmentImage(img));
    
    // Stack into a batch
    const batch = tf.stack(augmentedImages);
    
    // Apply batch-level augmentation (mixup or cutmix) occasionally
    const rand = Math.random();
    if (rand < 0.3) {
      // 30% chance of applying mixup
      return applyMixup(batch, 0.2);
    } else if (rand < 0.5) {
      // 20% chance of applying cutmix
      return applyCutMix(batch, 1.0);
    }
    
    return batch;
  });
}

// Export data functions
export {
  createTrainingDataset,
  augmentImage,
  applyMixup,
  applyCutMix,
  processSamplesForFeatureExtractor,
  createAugmentedBatch,
  preprocessForVGG19
};