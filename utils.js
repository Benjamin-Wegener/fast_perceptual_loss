// utils.js

/**
 * Generates a timestamp string for logging.
 * @returns {string} The current time in HH:MM:SS format.
 */
export function getTimestamp() {
    return new Date().toTimeString().split(' ')[0];
}

/**
 * Logs current TensorFlow.js memory usage.
 * @param {object} tf - The TensorFlow.js library object.
 */
export function logMemoryUsage(tf) {
    // Only log memory if a backend is currently active and tf.memory() is available.
    // This prevents errors if called before backend initialization or after disposal.
    if (tf.getBackend() && tf.memory) {
        const mem = tf.memory();
        console.log(`[${getTimestamp()}] TF Memory: ${mem.numBytes / 1024 / 1024} MB (${mem.numTensors} tensors)`);
    } else {
        console.log(`[${getTimestamp()}] TF Memory: Backend not yet initialized or memory info not available.`);
    }
}

/**
 * Helper function to create either a Conv2D or SeparableConv2D layer based on a flag.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {number} filters - The number of filters for the convolution.
 * @param {number} kernelSize - The size of the convolution kernel.
 * @param {boolean} enableDepthwiseConvs - Flag to enable depthwise convolutions.
 * @param {number} weightDecay - The L2 regularization strength.
 * @returns {tf.layers.Layer} A configured convolutional layer.
 */
export const getConvLayer = (tf, filters, kernelSize, enableDepthwiseConvs, weightDecay) => {
    const commonParams = {
        filters: filters,
        kernelSize: kernelSize,
        activation: 'relu',
        padding: 'same',
    };

    if (enableDepthwiseConvs) {
        return tf.layers.separableConv2d({
            ...commonParams,
            depthwiseInitializer: 'heUniform',
            pointwiseInitializer: 'heUniform',
            // Apply L2 regularization to both depthwise and pointwise kernels for separable convolutions
            depthwiseRegularizer: tf.regularizers.l2({l2: weightDecay}),
            pointwiseRegularizer: tf.regularizers.l2({l2: weightDecay})
        });
    } else {
        return tf.layers.conv2d({
            ...commonParams,
            kernelInitializer: 'heUniform',
            kernelRegularizer: tf.regularizers.l2({l2: weightDecay}) // Apply L2 regularization to kernel for standard conv2d
        });
    }
};

/**
 * Implements the Mean Squared Error loss between two feature tensors.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} yTrue - The ground truth feature tensor.
 * @param {tf.Tensor} yPred - The predicted feature tensor.
 * @returns {tf.Tensor} The calculated Mean Squared Error loss.
 */
export function featureMSELoss(tf, yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred);
}

/**
 * Augments an image tensor by applying random left-right flips.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The input image tensor (e.g., [H, W, C]).
 * @returns {tf.Tensor} The augmented image tensor.
 */
export function augmentImage(tf, tensor) {
    return tf.tidy(() => {
        let aug = tensor.expandDims(0); // Add batch dimension
        if (Math.random() > 0.5) {
            aug = tf.image.flipLeftRight(aug);
        }
        const result = aug.squeeze(0); // Remove batch dimension
        return result;
    });
}

/**
 * Extracts random patches of IMAGE_SIZE from a given image tensor.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} imgTensor - The full image tensor.
 * @param {number} patchCount - The number of random patches to extract.
 * @param {number} imageSize - The size of the patches to extract (IMAGE_SIZE).
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor>} A stacked tensor of patches.
 */
export async function extractRandomPatches(tf, imgTensor, patchCount, imageSize, getTimestampFn) {
    const [h, w, c] = imgTensor.shape;
    const patches = [];

    if (h < imageSize || w < imageSize) {
        console.warn(`[${getTimestampFn()}] Skipping image due to small size for patch extraction: ${h}x${w}. Required: ${imageSize}x${imageSize}`);
        return tf.zeros([0, imageSize, imageSize, 3]);
    }

    for (let i = 0; i < patchCount; i++) {
        const patch = tf.tidy(() => {
            const top = Math.floor(Math.random() * (h - imageSize + 1));
            const left = Math.floor(Math.random() * (w - imageSize + 1));

            const patchSlice = imgTensor.slice([top, left, 0], [imageSize, imageSize, c]);
            const augPatch = augmentImage(tf, patchSlice);
            patchSlice.dispose();
            return augPatch;
        });
        patches.push(patch);
    }

    let stackedPatches;
    if (patches.length > 0) {
        stackedPatches = tf.stack(patches);
    } else {
        stackedPatches = tf.zeros([0, imageSize, imageSize, 3]);
        console.warn(`[${getTimestampFn()}] No patches were extracted from this image. Returning empty tensor.`);
    }

    patches.forEach(t => t.dispose());

    return stackedPatches;
}

/**
 * Loads and preprocesses an image from a URL into a TensorFlow tensor,
 * resizing it to IMAGE_SIZE x IMAGE_SIZE.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {string} url - The URL of the image.
 * @param {number} imageSize - The target size for the image.
 * @param {function} updateStatusFn - Function to update UI status.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor3D>} The image tensor normalized to [0, 1].
 */
export async function loadImageAsTensor(tf, url, imageSize, updateStatusFn, getTimestampFn) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = imageSize;
            canvas.height = imageSize;

            ctx.drawImage(img, 0, 0, imageSize, imageSize);

            const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
            const tensor = tf.browser.fromPixels(imageData).div(255);
            console.log(`[${getTimestampFn()}] Tensor created from ${url} with shape: ${tensor.shape}`);
            resolve(tensor);
        };
        img.onerror = (err) => {
            console.error(`Failed to load image from URL: ${url}`, err);
            reject(err);
        };
        img.src = url;
    });
}

/**
 * Loads training patches from a single image and returns them immediately
 * @param {object} tf - The TensorFlow.js library object.
 * @param {string} imageUrl - URL of the image to process
 * @param {number} patchCount - Number of patches to extract
 * @param {number} imageSize - The target size for the image.
 * @param {function} updateStatusFn - Function to update UI status.
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 * @returns {Promise<tf.Tensor>} Patches for this image
 */
export async function loadPatchesFromSingleImage(tf, imageUrl, patchCount, imageSize, updateStatusFn, getTimestampFn) {
    let fullImageTensor = null;
    try {
        fullImageTensor = await loadImageAsTensor(tf, imageUrl, imageSize, updateStatusFn, getTimestampFn);

        if (fullImageTensor && fullImageTensor.shape && fullImageTensor.shape.length === 3) {
            const patches = await extractRandomPatches(tf, fullImageTensor, patchCount, imageSize, getTimestampFn);
            return patches;
        } else {
            console.warn(`[${getTimestampFn()}] loadImageAsTensor for ${imageUrl} returned an invalid tensor.`);
            return tf.zeros([0, imageSize, imageSize, 3]);
        }
    } catch (error) {
        console.error(`Error processing image ${imageUrl}:`, error);
        return tf.zeros([0, imageSize, imageSize, 3]);
    } finally {
        if (fullImageTensor instanceof tf.Tensor) fullImageTensor.dispose();
    }
}

/**
 * Displays a tensor as an image on the webpage using a canvas.
 * This function expects the input tensor to already be scaled to [0, 255] and cast to 'int32'.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} tensor - The tensor to display (expected to be [0, 255] int32).
 * @param {HTMLElement} parentElement - The DOM element to append the image to.
 * @param {string} title - A title for the image.
 */
export async function displayTensorAsImage(tf, tensor, parentElement, title) {
    const displayTensor = tf.tidy(() => tensor.squeeze()); // Squeeze only, no mul(255).cast('int32')

    const canvas = document.createElement('canvas');
    canvas.width = displayTensor.shape[1];
    canvas.height = displayTensor.shape[0];

    await tf.browser.toPixels(displayTensor, canvas);

    const container = document.createElement('div');
    container.style.display = 'inline-block';
    container.style.margin = '10px';
    const h4 = document.createElement('h4');
    h4.textContent = title;
    container.appendChild(h4);
    container.appendChild(canvas);
    parentElement.appendChild(container);

    displayTensor.dispose();
}

/**
 * Converts a feature map tensor into a displayable grayscale image tensor, scaled to [0, 255] int32.
 * This takes the mean across the last (channel) dimension to reduce it to a single channel,
 * then normalizes the values for display.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} featureTensor - The feature map tensor (e.g., [1, H, W, C_features]).
 * @returns {tf.Tensor} A grayscale image tensor [1, H, W, 1] scaled to [0, 255] int32.
 */
export function visualizeFeatureMap(tf, featureTensor) {
    return tf.tidy(() => {
        const meanFeatures = featureTensor.squeeze(0).mean(2);

        const minVal = meanFeatures.min();
        const maxVal = meanFeatures.max();
        const normalizedFeatures = (maxVal.sub(minVal).greater(tf.scalar(0.0001)) ?
                                   meanFeatures.sub(minVal).div(maxVal.sub(minVal)) :
                                   tf.zerosLike(meanFeatures));

        // Scale to 0-255 and cast to int32 here
        return normalizedFeatures.mul(255).cast('int32').expandDims(0).expandDims(3);
    });
}

/**
 * Creates a heatmap visualization for the difference between two feature tensors, scaled to [0, 255] int32.
 * Red indicates larger differences, black indicates smaller differences.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} diffTensor - The difference tensor (e.g., [1, H, W, C_features]).
 * @returns {tf.Tensor} An RGB image tensor [1, H, W, 3] representing the heatmap, scaled to [0, 255] int32.
 */
export function visualizeDifferenceHeatmap(tf, diffTensor) {
    return tf.tidy(() => {
        const meanDiff = diffTensor.squeeze(0).mean(2).abs();

        const maxVal = meanDiff.max();
        const minVal = meanDiff.min();
        const normalizedDiff = (maxVal.sub(minVal).greater(tf.scalar(0.0001)) ?
                                meanDiff.sub(minVal).div(maxVal.sub(minVal)) :
                                tf.zerosLike(meanDiff));

        const intensity = normalizedDiff;

        // Scale to 0-255 and cast to int32 here, as displayTensorAsImage now expects this.
        const rChannel = intensity.mul(255).cast('int32');
        const gChannel = tf.zerosLike(intensity).cast('int32');
        const bChannel = tf.zerosLike(intensity).cast('int32');

        const rgbImage = tf.stack([rChannel, gChannel, bChannel], -1);

        return rgbImage.expandDims(0);
    });
}

/**
 * Saves VGG feature visualizations for original and generated (lightweight) images,
 * including a heatmap of their difference.
 * @param {object} tf - The TensorFlow.js library object.
 * @param {tf.Tensor} inputImageTensor - The original input image tensor ([1, H, W, 3]).
 * @param {tf.Tensor} lightweightModelOutputTensor - The output of the lightweight model ([1, H_features, W_features, C_features]).
 * @param {string} titlePrefix - A title prefix for the display.
 * @param {HTMLElement} sampleContainerElement - The DOM element to append the visualizations to.
 * @param {tf.LayersModel} fullVggFeatureExtractorModel - The full VGG model (teacher).
 * @param {function} getTimestampFn - Function to get a timestamp for logging.
 */
export async function saveSideBySideImage(
    tf,
    inputImageTensor,
    lightweightModelOutputTensor,
    titlePrefix,
    sampleContainerElement,
    fullVggFeatureExtractorModel,
    getTimestampFn
) {
    if (!fullVggFeatureExtractorModel) {
        console.warn(`[${getTimestampFn()}] Full VGG Feature Extractor not available. Cannot display true VGG features or difference heatmap.`);
        // Ensure sampleContainerElement is cleared even if VGG is not available
        if (sampleContainerElement) sampleContainerElement.innerHTML = '<h3>Sample VGG Feature Visualizations:</h3><p>Full VGG Feature Extractor not available.</p>';
        return;
    }

    tf.engine().startScope();
    try {
        const originalVggFeatures = fullVggFeatureExtractorModel.predict(inputImageTensor);
        
        // Ensure lightweightModelOutputTensor is still within scope for later disposal if passed by reference
        const lightweightModelOutput = lightweightModelOutputTensor;

        const lightweightModelViz = visualizeFeatureMap(tf, lightweightModelOutput);
        const originalVggViz = visualizeFeatureMap(tf, originalVggFeatures);

        const featureDifference = originalVggFeatures.sub(lightweightModelOutput).abs();
        const diffHeatmapViz = visualizeDifferenceHeatmap(tf, featureDifference);


        // Clearing previous samples is now handled by the caller (main.js)
        // if (sampleContainerElement) sampleContainerElement.innerHTML = '';

        await displayTensorAsImage(tf, originalVggViz, sampleContainerElement, `${titlePrefix} Original VGG Features`);
        await displayTensorAsImage(tf, lightweightModelViz, sampleContainerElement, `${titlePrefix} Lightweight Model Features`);
        await displayTensorAsImage(tf, diffHeatmapViz, sampleContainerElement, `${titlePrefix} Difference Heatmap`);

        originalVggFeatures.dispose();
        originalVggViz.dispose();
        lightweightModelViz.dispose();
        featureDifference.dispose();
        diffHeatmapViz.dispose();

    } catch (error) {
        console.error(`[${getTimestampFn()}] Error during sample visualization:`, error);
        if (sampleContainerElement) sampleContainerElement.innerHTML = '<h3>Sample VGG Feature Visualizations:</h3><p>Error displaying features.</p>';
    } finally {
        tf.engine().endScope();
    }
}
