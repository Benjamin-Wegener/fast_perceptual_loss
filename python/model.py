import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Input, MaxPooling2D, BatchNormalization, 
    Add, DepthwiseConv2D, SeparableConv2D, 
    GlobalAveragePooling2D, Concatenate, Multiply,
    Reshape, Layer, Lambda, Activation, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2

# Custom layers for serialization
class WeightedAddLayer(tf.keras.layers.Layer):
    """Custom layer for weighted addition with a learnable parameter"""
    def __init__(self, initial_value=0.1, name=None, **kwargs):
        super(WeightedAddLayer, self).__init__(name=name, **kwargs)
        self.initial_value = initial_value
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        super(WeightedAddLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs should be a list of [x, residual]
        return inputs[0] + inputs[1] * self.gamma
        
    def get_config(self):
        config = super(WeightedAddLayer, self).get_config()
        config.update({'initial_value': self.initial_value})
        return config

class MeanReduceLayer(tf.keras.layers.Layer):
    """Custom layer for reducing mean along specified axes"""
    def __init__(self, axis, keepdims=True, name=None, **kwargs):
        super(MeanReduceLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)
        
    def get_config(self):
        config = super(MeanReduceLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config

class MaxReduceLayer(tf.keras.layers.Layer):
    """Custom layer for reducing max along specified axes"""
    def __init__(self, axis, keepdims=True, name=None, **kwargs):
        super(MaxReduceLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
        
    def get_config(self):
        config = super(MaxReduceLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config

class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth layer for improved training.
    
    During training, randomly drops the entire residual branch with probability
    equal to drop_rate. During inference, scales the residual branch by
    the survival probability (1 - drop_rate).
    
    Args:
        drop_rate: Float between 0 and 1. Probability of dropping the residual.
    """
    def __init__(self, drop_rate, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        
    def call(self, inputs, training=None):
        # inputs should be [shortcut, residual]
        shortcut, residual = inputs
        
        if training:
            keep_prob = 1.0 - self.drop_rate
            random_tensor = keep_prob + tf.random.uniform([], 0, 1)
            binary_tensor = tf.floor(random_tensor)
            output = shortcut + binary_tensor * residual
        else:
            output = shortcut + (1.0 - self.drop_rate) * residual
        
        return output
    
    def get_config(self):
        config = super(StochasticDepth, self).get_config()
        config.update({'drop_rate': self.drop_rate})
        return config

# SiLU/Swish activation function (more efficient than Mish)
def swish_activation(x):
    return x * tf.nn.sigmoid(x)

# Standard Squeeze-and-Excitation block
def squeeze_excitation_block(inputs, reduction_ratio=16, name_prefix=''):
    """
    Efficient Squeeze-and-Excitation channel attention
    """
    channels = inputs.shape[-1]
    
    # Global average pooling
    squeeze = GlobalAveragePooling2D(name=f'{name_prefix}_se_gap')(inputs)
    squeeze = Reshape((1, 1, channels), name=f'{name_prefix}_se_reshape')(squeeze)
    
    # Bottleneck FC layers
    reduced_channels = max(1, channels // reduction_ratio)
    excitation = Conv2D(reduced_channels, kernel_size=1, use_bias=True, 
                      kernel_initializer=HeNormal(seed=42),
                      name=f'{name_prefix}_se_reduce')(squeeze)
    excitation = Activation('relu', name=f'{name_prefix}_se_relu')(excitation)
    excitation = Conv2D(channels, kernel_size=1, use_bias=True,
                      kernel_initializer=HeNormal(seed=42),
                      name=f'{name_prefix}_se_expand')(excitation)
    excitation = Activation('sigmoid', name=f'{name_prefix}_se_sigmoid')(excitation)
    
    # Apply attention weights
    return Multiply(name=f'{name_prefix}_se_multiply')([inputs, excitation])

# MBConv block (Mobile Inverted Bottleneck Conv) from EfficientNet
def mbconv_block(inputs, output_channels, expansion_factor=6, stride=1, 
                 kernel_size=3, se_ratio=0.25, drop_path_rate=0.0, name_prefix=''):
    """
    MBConv block: More efficient than Ghost Module with better performance.
    
    Args:
        inputs: Input tensor
        output_channels: Number of output channels
        expansion_factor: Channel expansion factor
        stride: Stride for depthwise conv
        kernel_size: Kernel size for depthwise conv
        se_ratio: Squeeze-and-Excitation ratio
        drop_path_rate: Drop path rate for stochastic depth
        name_prefix: Prefix for layer names
    """
    input_channels = inputs.shape[-1]
    expanded_channels = max(1, int(input_channels * expansion_factor))
    
    # Shortcut connection (if dimensions change, use 1x1 conv)
    if stride == 1 and input_channels == output_channels:
        shortcut = inputs
    else:
        shortcut = Conv2D(output_channels, kernel_size=1, strides=stride,
                         padding='same', use_bias=False,
                         kernel_initializer=HeNormal(seed=42),
                         name=f'{name_prefix}_shortcut_conv')(inputs)
        shortcut = BatchNormalization(momentum=0.9, name=f'{name_prefix}_shortcut_bn')(shortcut)
    
    # Expansion phase
    if expansion_factor != 1:
        x = Conv2D(expanded_channels, kernel_size=1, padding='same',
                  use_bias=False, kernel_initializer=HeNormal(seed=42),
                  name=f'{name_prefix}_expand_conv')(inputs)
        x = BatchNormalization(momentum=0.9, name=f'{name_prefix}_expand_bn')(x)
        x = Lambda(swish_activation, name=f'{name_prefix}_expand_swish')(x)
    else:
        x = inputs
    
    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same',
                      use_bias=False, depthwise_initializer=HeNormal(seed=42),
                      name=f'{name_prefix}_dw_conv')(x)
    x = BatchNormalization(momentum=0.9, name=f'{name_prefix}_dw_bn')(x)
    x = Lambda(swish_activation, name=f'{name_prefix}_dw_swish')(x)
    
    # Squeeze-and-Excitation
    if 0 < se_ratio <= 1:
        x = squeeze_excitation_block(x, 
                                    reduction_ratio=max(1, int(expanded_channels * se_ratio)),
                                    name_prefix=f'{name_prefix}_se')
    
    # Output projection
    x = Conv2D(output_channels, kernel_size=1, padding='same',
              use_bias=False, kernel_initializer=HeNormal(seed=42),
              name=f'{name_prefix}_project_conv')(x)
    x = BatchNormalization(momentum=0.9, name=f'{name_prefix}_project_bn')(x)
    
    # Skip connection with stochastic depth
    if stride == 1 and input_channels == output_channels:
        if drop_path_rate > 0:
            x = StochasticDepth(drop_path_rate, name=f'{name_prefix}_stochastic_depth')([shortcut, x])
        else:
            x = Add(name=f'{name_prefix}_residual_add')([shortcut, x])
    
    return x

# Spatial Attention Module
def spatial_attention(inputs, kernel_size=7, name_prefix=''):
    """
    Efficient spatial attention module to highlight important regions
    """
    # Average pooling along channel dimension
    avg_pool = MeanReduceLayer(
        axis=-1, 
        keepdims=True,
        name=f'{name_prefix}_spatial_avg'
    )(inputs)
    
    # Max pooling along channel dimension
    max_pool = MaxReduceLayer(
        axis=-1, 
        keepdims=True,
        name=f'{name_prefix}_spatial_max'
    )(inputs)
    
    # Concatenate pooled features
    concat = Concatenate(name=f'{name_prefix}_spatial_concat')([avg_pool, max_pool])
    
    # Convolutional layer to generate spatial attention map
    spatial_map = Conv2D(1, kernel_size, padding='same', 
                       kernel_initializer=HeNormal(seed=42),
                       use_bias=False,
                       name=f'{name_prefix}_spatial_conv')(concat)
    
    # Apply sigmoid activation
    spatial_map = Activation('sigmoid', name=f'{name_prefix}_spatial_sigmoid')(spatial_map)
    
    # Apply spatial attention
    return Multiply(name=f'{name_prefix}_spatial_multiply')([inputs, spatial_map])

# Create the enhanced FastPerceptualLoss model - REPLACED WITH LARGER VERSION
def create_fast_perceptual_model(input_shape=(None, None, 3)):
    """
    Enhanced model with ~3x parameters:
    - Wider layers with more filters
    - Additional blocks for more capacity
    - Maintains the same architectural patterns but scaled up
    - Output dimensions matching VGG19 block3_conv3 (1/4 of input size)
    """
    # Advanced initialization strategy
    kernel_init = HeNormal(seed=42)
    kernel_reg = l2(1e-5)
    
    # Input layer
    inputs = Input(shape=input_shape, name="input_image")
    
    # First block - Initial conv (wider)
    x = SeparableConv2D(64, 3, padding='same',  # Increased from 32 to 64
              kernel_initializer=kernel_init,
              use_bias=False,
              name="conv1")(inputs)
    x = BatchNormalization(momentum=0.9, name="bn1")(x)
    x = Lambda(swish_activation, name="swish1")(x)
    
    # Additional initial conv block for more capacity
    x = SeparableConv2D(96, 3, padding='same',  # New layer
              kernel_initializer=kernel_init,
              use_bias=False,
              name="conv1b")(x)
    x = BatchNormalization(momentum=0.9, name="bn1b")(x)
    x = Lambda(swish_activation, name="swish1b")(x)
    
    # Skip connection for residual learning
    skip1 = x
    
    # FIRST MAX POOLING - reduces dimensions by 2x
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # MBConv blocks with increasing complexity
    x = mbconv_block(x, 64, expansion_factor=4,  # Increased from 32 to 64
                    se_ratio=0.25, drop_path_rate=0.05,
                    name_prefix='mbconv1')
    
    # Additional block
    x = mbconv_block(x, 80, expansion_factor=4,  # New block
                    se_ratio=0.25, drop_path_rate=0.05,
                    name_prefix='mbconv1b')
    
    x = mbconv_block(x, 96, expansion_factor=4,  # Increased from 48 to 96
                    se_ratio=0.25, drop_path_rate=0.05,
                    name_prefix='mbconv2')
    
    # Skip connection
    residual2 = x
    
    # Add spatial attention
    x = spatial_attention(x, kernel_size=5, name_prefix='sa1')
    
    # Residual connection
    x = Add(name="add1")([x, residual2])
    
    # SECOND MAX POOLING - total reduction now 4x (matches VGG19)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Additional blocks for more capacity
    x = mbconv_block(x, 128, expansion_factor=6,  # Increased from 64 to 128
                    se_ratio=0.25, drop_path_rate=0.1,
                    name_prefix='mbconv3')
    
    # New additional blocks
    x = mbconv_block(x, 160, expansion_factor=6,  # New block
                    se_ratio=0.25, drop_path_rate=0.1,
                    name_prefix='mbconv4')
    
    x = mbconv_block(x, 192, expansion_factor=6,  # New block
                    se_ratio=0.25, drop_path_rate=0.1,
                    name_prefix='mbconv5')
    
    # Multi-level feature fusion with intermediate connections
    # Take first skip connection, downsample with fixed size
    skip1_down = MaxPooling2D(pool_size=(4, 4), name="skip1_down")(skip1)
    skip1_proj = Conv2D(192, 1, padding='same',  # Increased from 64 to 128
                      kernel_initializer=kernel_init,
                      name="skip1_proj")(skip1_down)
    
    # Combine with main path
    x = Add(name="multi_level_fusion")([x, skip1_proj])
    
    # Final squeeze-excitation block
    x = squeeze_excitation_block(x, reduction_ratio=8, name_prefix='final_se')
    
    # Final 1x1 projection to match VGG19 features dimension (256)
    x = Conv2D(256, 1, padding='same', 
             kernel_initializer=kernel_init,
             kernel_regularizer=kernel_reg,
             name="final_proj")(x)
    x = BatchNormalization(momentum=0.9, name="final_bn")(x)
    x = Lambda(swish_activation, name="final_swish")(x)
    
    # The final output matches VGG19 block3_conv3 output (256 filters, 1/4 spatial dim)
    model = Model(inputs, x, name="EnhancedFastPerceptualLoss")
    
    print(f"Input shape: {model.input.shape}, Output shape: {model.output.shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    return model