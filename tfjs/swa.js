// swa.js
// Stochastic Weight Averaging implementation for TensorFlow.js

import { logQueue } from './globals.js';

/**
 * Stochastic Weight Averaging (SWA) class
 * 
 * Implements the Stochastic Weight Averaging technique for improving
 * generalization in deep learning models by averaging weights from multiple
 * points in the training trajectory.
 */
class SWA {
  /**
   * Constructor
   * @param {Object} config Configuration object
   * @param {number} config.startEpoch The epoch to start averaging from (default: 10)
   * @param {number} config.swaFreq Frequency of weight averaging in epochs (default: 5)
   * @param {number} config.verbose Verbosity level (default: 1)
   */
  constructor({
    startEpoch = 10,
    swaFreq = 5,
    verbose = 1
  }) {
    this.startEpoch = startEpoch;
    this.swaFreq = swaFreq;
    this.verbose = verbose;
    this.swaWeights = null;
    this.swaCount = 0;
  }
  
  /**
   * Extract model weights as a flat array of tensors
   * @param {Object} model TensorFlow.js model
   * @return {Array} Array of weight tensors
   */
  _getModelWeights(model) {
    return model.weights.map(w => w.val.clone());
  }
  
  /**
   * Set model weights from a flat array of tensors
   * @param {Object} model TensorFlow.js model
   * @param {Array} weights Array of weight tensors
   */
  _setModelWeights(model, weights) {
    // Verify weights array length matches model weights
    if (weights.length !== model.weights.length) {
      throw new Error(`Weight count mismatch: got ${weights.length}, expected ${model.weights.length}`);
    }
    
    // Assign each weight tensor to the corresponding model weight
    model.weights.forEach((w, i) => {
      w.val.assign(weights[i]);
    });
  }
  
  /**
   * Called at the end of each epoch
   * @param {number} epoch Current epoch (0-indexed)
   * @param {Object} model TensorFlow.js model
   */
  onEpochEnd(epoch, model) {
    // Check if we should update SWA
    if (epoch + 1 >= this.startEpoch && (epoch + 1 - this.startEpoch) % this.swaFreq === 0) {
      // Get current model weights
      const currentWeights = this._getModelWeights(model);
      
      // Update SWA weights
      if (this.swaWeights === null) {
        // First time - just copy the weights
        this.swaWeights = currentWeights;
      } else {
        // Update running average
        this.swaWeights = this.swaWeights.map((w, i) => {
          // Calculate running average: (swa_weights * count + new_weights) / (count + 1)
          const updatedWeight = tf.tidy(() => {
            return tf.add(
              tf.mul(w, this.swaCount),
              currentWeights[i]
            ).div(this.swaCount + 1);
          });
          
          // Dispose the old weight tensor
          w.dispose();
          
          return updatedWeight;
        });
      }
      
      // Update count
      this.swaCount++;
      
      if (this.verbose > 0) {
        logQueue.put(`SWA: Updated weights average at epoch ${epoch+1} (total models: ${this.swaCount})`);
      }
    }
  }
  
  /**
   * Called at the end of training
   * @param {Object} model TensorFlow.js model
   */
  onTrainEnd(model) {
    // Apply SWA weights when training ends
    if (this.swaWeights !== null) {
      if (this.verbose > 0) {
        logQueue.put(`SWA: Applying averaged weights from ${this.swaCount} models`);
      }
      
      // Store original weights to allow recovery
      this.originalWeights = this._getModelWeights(model);
      
      // Set SWA weights to model
      this._setModelWeights(model, this.swaWeights);
      
      if (this.verbose > 0) {
        logQueue.put("SWA: Weights successfully applied");
      }
    } else {
      if (this.verbose > 0) {
        logQueue.put("SWA: No weights were averaged, model remains unchanged");
      }
    }
  }
  
  /**
   * Return the original (non-SWA) weights if available
   * @return {Array|null} Original model weights or null
   */
  getOriginalWeights() {
    if (this.originalWeights) {
      return this.originalWeights;
    }
    return null;
  }
  
  /**
   * Reset model to the original weights before SWA
   * @param {Object} model TensorFlow.js model
   * @return {boolean} True if reset was successful, false otherwise
   */
  resetOriginalWeights(model) {
    if (this.originalWeights) {
      this._setModelWeights(model, this.originalWeights);
      return true;
    }
    return false;
  }
  
  /**
   * Dispose of all tensors held by this object
   */
  dispose() {
    // Clean up tensors to prevent memory leaks
    if (this.swaWeights) {
      this.swaWeights.forEach(w => w.dispose());
      this.swaWeights = null;
    }
    
    if (this.originalWeights) {
      this.originalWeights.forEach(w => w.dispose());
      this.originalWeights = null;
    }
  }
}

export { SWA };