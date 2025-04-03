// model.js
// TensorFlow.js implementation of the Fast Perceptual Loss model

import { logQueue } from './globals.js';

// Custom layers implementations for TensorFlow.js
class WeightedAddLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.initialValue = config.initialValue || 0.1;
  }
  
  build(inputShape) {
    this.gamma = this.addWeight(
      'gamma',
      [],
      'float32',
      tf.initializers.constant({value: this.initialValue}),
      true
    );
    super.build(inputShape);
  }
  
  call(inputs) {
    return tf.tidy(() => {
      return tf.add(inputs[0], tf.mul(inputs[1], this.gamma.read()));
    });
  }
  
  computeOutputShape(inputShape) {
    return inputShape[0];
  }
  
  static get className() {
    return 'WeightedAddLayer';
  }
  
  getConfig() {
    const config = super.getConfig();
    config.initialValue = this.initialValue;
    return config;
  }
}

// Register the custom layer
tf.serialization.registerClass(WeightedAddLayer);

// Squeeze-and-Excitation block
function squeezeExcitationBlock(inputs, reductionRatio = 16, namePrefix = '') {
  const channels = inputs.shape[inputs.shape.length - 1];
  
  // Global average pooling
  const squeeze = tf.layers.globalAveragePooling2d({
    name: `${namePrefix}_se_gap`
  }).apply(inputs);
  
  const reshape = tf.layers.reshape({
    targetShape: [1, 1, channels],
    name: `${namePrefix}_se_reshape`
  }).apply(squeeze);
  
  // Bottleneck FC layers
  const reducedChannels = Math.max(1, Math.floor(channels / reductionRatio));
  
  const excitation = tf.layers.conv2d({
    filters: reducedChannels,
    kernelSize: 1,
    useBias: true,
    kernelInitializer: 'heNormal',
    name: `${namePrefix}_se_reduce`
  }).apply(reshape);
  
  const relu = tf.layers.activation({
    activation: 'relu',
    name: `${namePrefix}_se_relu`
  }).apply(excitation);
  
  const expand = tf.layers.conv2d({
    filters: channels,
    kernelSize: 1,
    useBias: true,
    kernelInitializer: 'heNormal',
    name: `${namePrefix}_se_expand`
  }).apply(relu);
  
  const sigmoid = tf.layers.activation({
    activation: 'sigmoid',
    name: `${namePrefix}_se_sigmoid`
  }).apply(expand);
  
  // Apply attention weights
  return tf.layers.multiply({
    name: `${namePrefix}_se_multiply`
  }).apply([inputs, sigmoid]);
}

// MBConv block (Mobile Inverted Bottleneck Conv)
function mbconvBlock(inputs, outputChannels, expansionFactor = 6, stride = 1,
                    kernelSize = 3, seRatio = 0.25, dropPathRate = 0.0, namePrefix = '') {
  const inputChannels = inputs.shape[inputs.shape.length - 1];
  const expandedChannels = Math.max(1, Math.floor(inputChannels * expansionFactor));
  
  // Shortcut connection (if dimensions change, use 1x1 conv)
  let shortcut;
  if (stride === 1 && inputChannels === outputChannels) {
    shortcut = inputs;
  } else {
    shortcut = tf.layers.conv2d({
      filters: outputChannels,
      kernelSize: 1,
      strides: stride,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal',
      name: `${namePrefix}_shortcut_conv`
    }).apply(inputs);
    
    shortcut = tf.layers.batchNormalization({
      momentum: 0.9,
      name: `${namePrefix}_shortcut_bn`
    }).apply(shortcut);
  }
  
  // Expansion phase
  let x;
  if (expansionFactor !== 1) {
    x = tf.layers.conv2d({
      filters: expandedChannels,
      kernelSize: 1,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal',
      name: `${namePrefix}_expand_conv`
    }).apply(inputs);
    
    x = tf.layers.batchNormalization({
      momentum: 0.9,
      name: `${namePrefix}_expand_bn`
    }).apply(x);
    
    x = tf.layers.activation({
      activation: 'relu',
      name: `${namePrefix}_expand_relu`
    }).apply(x);
  } else {
    x = inputs;
  }
  
  // Depthwise convolution
  x = tf.layers.depthwiseConv2d({
    kernelSize: kernelSize,
    strides: stride,
    padding: 'same',
    useBias: false,
    depthwiseInitializer: 'heNormal',
    name: `${namePrefix}_dw_conv`
  }).apply(x);
  
  x = tf.layers.batchNormalization({
    momentum: 0.9,
    name: `${namePrefix}_dw_bn`
  }).apply(x);
  
  x = tf.layers.activation({
    activation: 'relu',
    name: `${namePrefix}_dw_relu`
  }).apply(x);
  
  // Squeeze-and-Excitation
  if (seRatio > 0 && seRatio <= 1) {
    x = squeezeExcitationBlock(
      x,
      Math.max(1, Math.floor(expandedChannels * seRatio)),
      `${namePrefix}_se`
    );
  }
  
  // Output projection
  x = tf.layers.conv2d({
    filters: outputChannels,
    kernelSize: 1,
    padding: 'same',
    useBias: false,
    kernelInitializer: 'heNormal',
    name: `${namePrefix}_project_conv`
  }).apply(x);
  
  x = tf.layers.batchNormalization({
    momentum: 0.9,
    name: `${namePrefix}_project_bn`
  }).apply(x);
  
  // Skip connection
  if (stride === 1 && inputChannels === outputChannels) {
    if (dropPathRate > 0) {
      // Implement a simplified version of stochastic depth
      // For training mode, we'd randomly drop the path
      // For inference, we scale by survival probability
      const survival = 1.0 - dropPathRate;
      x = tf.layers.multiply({
        name: `${namePrefix}_drop_path_scale`
      }).apply([x, tf.scalar(survival)]);
    }
    
    // Add skip connection
    x = tf.layers.add({
      name: `${namePrefix}_residual_add`
    }).apply([shortcut, x]);
  }
  
  return x;
}

// Simplified Spatial Attention Module
function spatialAttention(inputs, kernelSize = 7, namePrefix = '') {
  // Instead of using tf.mean or tf.max directly, we'll use convolutional layers
  // to achieve similar effects with standard TensorFlow.js operations
  
  // Channel reduction to get spatial features
  const channelReduce = tf.layers.conv2d({
    filters: 1,
    kernelSize: 1,
    padding: 'same',
    kernelInitializer: 'heNormal',
    useBias: true,
    name: `${namePrefix}_channel_reduce`
  }).apply(inputs);
  
  // Apply attention map
  const attentionMap = tf.layers.conv2d({
    filters: 1,
    kernelSize: kernelSize,
    padding: 'same',
    kernelInitializer: 'heNormal',
    useBias: true,
    name: `${namePrefix}_attention_map`
  }).apply(channelReduce);
  
  // Apply sigmoid activation
  const attention = tf.layers.activation({
    activation: 'sigmoid',
    name: `${namePrefix}_spatial_sigmoid`
  }).apply(attentionMap);
  
  // Apply spatial attention
  return tf.layers.multiply({
    name: `${namePrefix}_spatial_multiply`
  }).apply([inputs, attention]);
}

// Create the FastPerceptualLoss model
function createFastPerceptualModel(inputShape = [null, null, 3]) {
  try {
    // Input layer
    const inputs = tf.input({
      shape: inputShape,
      name: "input_image"
    });
    
    // First block - Initial conv (wider)
    let x = tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal',
      name: "conv1"
    }).apply(inputs);
    
    x = tf.layers.batchNormalization({
      momentum: 0.9,
      name: "bn1"
    }).apply(x);
    
    x = tf.layers.activation({
      activation: 'relu',
      name: "relu1"
    }).apply(x);
    
    // Additional initial conv block for more capacity
    x = tf.layers.conv2d({
      filters: 128,  // Increased from 96 to accommodate VGG19 features better
      kernelSize: 3,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal',
      name: "conv1b"
    }).apply(x);
    
    x = tf.layers.batchNormalization({
      momentum: 0.9,
      name: "bn1b"
    }).apply(x);
    
    x = tf.layers.activation({
      activation: 'relu',
      name: "relu1b"
    }).apply(x);
    
    // Skip connection for residual learning
    const skip1 = x;
    
    // FIRST MAX POOLING - reduces dimensions by 2x
    x = tf.layers.maxPooling2d({
      poolSize: [2, 2],
      name: "pool1"
    }).apply(x);
    
    // MBConv blocks with increasing complexity
    x = mbconvBlock(x, 96, 4,
                  1, 3, 0.25, 0.05,
                  'mbconv1');
    
    // Additional block
    x = mbconvBlock(x, 128, 4,
                  1, 3, 0.25, 0.05,
                  'mbconv1b');
    
    x = mbconvBlock(x, 160, 4,
                  1, 3, 0.25, 0.05,
                  'mbconv2');
    
    // Skip connection
    const residual2 = x;
    
    // Add spatial attention with a simplified implementation
    x = spatialAttention(x, 3, 'sa1');
    
    // Residual connection
    x = tf.layers.add({
      name: "add1"
    }).apply([x, residual2]);
    
    // SECOND MAX POOLING - total reduction now 4x
    x = tf.layers.maxPooling2d({
      poolSize: [2, 2],
      name: "pool2"
    }).apply(x);
    
    // Additional blocks for more capacity
    x = mbconvBlock(x, 192, 6,
                  1, 3, 0.25, 0.1,
                  'mbconv3');
    
    // New additional blocks
    x = mbconvBlock(x, 256, 6,
                  1, 3, 0.25, 0.1,
                  'mbconv4');
    
    x = mbconvBlock(x, 320, 6,
                  1, 3, 0.25, 0.1,
                  'mbconv5');
    
    // Multi-level feature fusion with intermediate connections
    // Take first skip connection, downsample with fixed size
    const skip1Down = tf.layers.maxPooling2d({
      poolSize: [4, 4],
      name: "skip1_down"
    }).apply(skip1);
    
    const skip1Proj = tf.layers.conv2d({
      filters: 320,  // Increased from 192 to match the updated filter count
      kernelSize: 1,
      padding: 'same',
      kernelInitializer: 'heNormal',
      name: "skip1_proj"
    }).apply(skip1Down);
    
    // Combine with main path
    x = tf.layers.add({
      name: "multi_level_fusion"
    }).apply([x, skip1Proj]);
    
    // Final squeeze-excitation block
    x = squeezeExcitationBlock(x, 8, 'final_se');
    
    // Final 1x1 projection to match VGG19 feature dimensions (512)
    // Changed from 256 to 512 to match VGG19 block4_conv2 output
    x = tf.layers.conv2d({
      filters: 512,  // Changed from 256 to 512 for VGG19
      kernelSize: 1,
      padding: 'same',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({l2: 1e-5}),
      name: "final_proj"
    }).apply(x);
    
    x = tf.layers.batchNormalization({
      momentum: 0.9,
      name: "final_bn"
    }).apply(x);
    
    x = tf.layers.activation({
      activation: 'relu',
      name: "final_relu"
    }).apply(x);
    
    // Create and return the model
    const model = tf.model({
      inputs: inputs,
      outputs: x,
      name: "EnhancedFastPerceptualLoss"
    });
    
    // Log model information
    logQueue.put(`Input shape: ${JSON.stringify(model.inputs[0].shape)}, Output shape: ${JSON.stringify(model.outputs[0].shape)}`);
    logQueue.put(`Model created successfully`);
    
    return model;
  } catch (error) {
    logQueue.put(`Error creating model: ${error.message}`);
    throw error;
  }
}

// Export model creation function and custom layers
export {
  createFastPerceptualModel,
  WeightedAddLayer,
  squeezeExcitationBlock,
  mbconvBlock,
  spatialAttention
};