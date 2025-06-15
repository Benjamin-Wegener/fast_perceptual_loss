# 🧠 Lightweight Perceptual Feature Extractor

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![WebGPU](https://img.shields.io/badge/WebGPU-005CFF?style=flat&logo=webgl&logoColor=white)](https://webgpu.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Train a compact neural network in your browser to mimic VGG19's feature extraction capabilities. Perfect for style transfer, super-resolution, and image generation tasks without the computational overhead.

## ✨ Features

- 🌐 **Browser-Based Training** - No setup required, runs entirely in your browser
- ⚡ **WebGPU/WebGL Acceleration** - Leverages hardware acceleration for fast training
- 📊 **Real-Time Visualization** - Live loss curves and feature map comparisons
- 💾 **Smart Model Management** - Auto-save to IndexedDB with download/upload support
- 🎯 **Perceptual Loss Training** - Learn from VGG19's block2_pool layer (256×256 → 64×64)

## 🚀 Quick Start

### Prerequisites
- Modern browser with WebGPU/WebGL support
- Local web server (Python's `http.server` recommended)

### Setup
```bash
# Clone the repository
git clone --recurse-submodules https://github.com/Benjamin-Wegener/fast_perceptual_loss
cd fast_perceptual_loss

# Start a local server (need python3 for that)
start_webserver.bat (windows)
./start_webserver.sh (linux)

# Open in browser (FF nightly or Chrome Canary /w webgpu)
open http://localhost:8000
```

### Project Structure
```
├── index.html              # Main application
├── main.js                 # Core logic
├── utils.js                # TensorFlow.js utilities
├── dataset/                # Training images
│   ├── baboon.jpeg
│   └── lenna.jpeg
└── tfjs_vgg19_imagenet-master/
    └── model/
        ├── model.json      # Pre-trained VGG19
        └── weights.bin
```

## 🎮 Usage

| Button | Function |
|--------|----------|
| **Start Training** | Begin training process |
| **Stop Training** | Halt training at current epoch |
| **Save Model** | Download trained model |
| **Load Model** | Upload previously saved model |
| **Delete Model** | Clear cached model from browser |

### Real-Time Monitoring
- 📈 **Loss Curve** - Track training progress
- 🖼️ **Feature Visualizations** - Compare VGG19 vs lightweight model outputs
- 🔥 **Difference Heatmap** - Identify areas needing improvement

## 🔧 Integration Example

```javascript
import * as tf from '@tensorflow/tfjs';

// Load your trained model
const model = await tf.loadLayersModel('indexeddb://lightweight_feature_extractor_model');

// Extract features from an image
async function getFeatures(imageTensor) {
    const features = model.predict(imageTensor.expandDims(0));
    return features; // Shape: [1, 64, 64, 128]
}

// Perceptual loss function
function perceptualLoss(targetFeatures, generatedFeatures) {
    return tf.losses.meanSquaredError(targetFeatures, generatedFeatures);
}
```

## 🧪 Technical Details

- **Input Size**: 256×256×3 images
- **Output Size**: 64×64×128 feature maps
- **Architecture**: Lightweight CNN trained to mimic VGG19's block2_pool
- **Loss Function**: MSE between teacher (VGG19) and student (lightweight) features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VGG19 model adapted from [paulsp94/tfjs_vgg19_imagenet](https://github.com/paulsp94/tfjs_vgg19_imagenet)
- Built with [TensorFlow.js](https://www.tensorflow.org/js)

---

**Author**: Benjamin Wegener

⭐ Star this repo if you find it useful!
