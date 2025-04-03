// onecycle_lr.js
// Implementation of the One Cycle Learning Rate policy for TensorFlow.js

import { logQueue } from './globals.js';

// OneCycleLR class for learning rate scheduling
class OneCycleLR {
  /**
   * Enhanced Implementation of the One Cycle Learning Rate Policy.
   * 
   * @param {Object} config Configuration object
   * @param {number} config.maxLr Maximum learning rate
   * @param {number} config.stepsPerEpoch Number of steps per epoch
   * @param {number} config.epochs Total number of epochs
   * @param {number} config.minLr Minimum learning rate (starting). Default is maxLr/10
   * @param {number} config.finalDivFactor Final LR will be maxLr/finalDivFactor. Default is 100
   * @param {number} config.warmupPct Percentage of total iterations for warmup phase. Default is 0.3
   * @param {Function} config.onLrUpdate Function to call when LR updates (for UI updates)
   * @param {Array} config.momentumRange Tuple of (max_momentum, min_momentum) to vary momentum inversely to lr
   * @param {number} config.weightDecay Weight decay factor. Set to null to disable
   * @param {number} config.verbose Whether to print verbose output
   */
  constructor({
    maxLr,
    stepsPerEpoch,
    epochs,
    minLr = null,
    finalDivFactor = 100,
    warmupPct = 0.3,
    onLrUpdate = null,
    momentumRange = [0.95, 0.85],
    weightDecay = null,
    verbose = 1
  }) {
    this.maxLr = maxLr;
    this.minLr = minLr !== null ? minLr : maxLr / 10;
    this.finalLr = maxLr / finalDivFactor;
    this.stepsPerEpoch = stepsPerEpoch;
    this.totalSteps = stepsPerEpoch * epochs;
    this.warmupSteps = Math.floor(this.totalSteps * warmupPct);
    this.annealingSteps = this.totalSteps - this.warmupSteps;
    this.currentStep = 0;
    this.onLrUpdate = onLrUpdate;
    this.history = {
      lr: [],
      momentum: [],
      weightDecay: []
    };
    this.initialized = false;
    this.verbose = verbose;
    
    // Momentum scheduling (inverse to learning rate)
    [this.maxMomentum, this.minMomentum] = momentumRange;
    
    // Weight decay scheduling
    this.weightDecay = weightDecay;
    this.weightDecaySchedule = [];
    
    if (weightDecay !== null) {
      // Create weight decay schedule - follow LR pattern but scaled
      this.initialWeightDecay = weightDecay;
      this.maxWeightDecay = weightDecay * 10;  // Increase during warmup
      this.finalWeightDecay = weightDecay / 10;  // Lower at the end
    }
  }
  
  /**
   * Calculate learning rate for current step according to schedule
   * @param {number} step Current step
   * @return {number} Learning rate for this step
   */
  _calculateLr(step) {
    if (step < this.warmupSteps) {
      // Linear warmup phase
      const progress = step / this.warmupSteps;
      return this.minLr + (this.maxLr - this.minLr) * progress;
    } else {
      // Cosine annealing phase
      const progress = (step - this.warmupSteps) / this.annealingSteps;
      // Smoother cosine annealing with a slight offset to avoid sudden changes
      const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * progress));
      return this.finalLr + (this.maxLr - this.finalLr) * cosineDecay;
    }
  }
  
  /**
   * Calculate momentum for current step - inverse to learning rate
   * @param {number} step Current step
   * @return {number} Momentum for this step
   */
  _calculateMomentum(step) {
    if (step < this.warmupSteps) {
      // Linear decrease during warmup
      const progress = step / this.warmupSteps;
      return this.maxMomentum - (this.maxMomentum - this.minMomentum) * progress;
    } else {
      // Linear increase during annealing
      const progress = (step - this.warmupSteps) / this.annealingSteps;
      const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * progress));
      return this.minMomentum + (this.maxMomentum - this.minMomentum) * cosineDecay;
    }
  }
  
  /**
   * Calculate weight decay for current step if enabled
   * @param {number} step Current step
   * @return {number|null} Weight decay for this step or null if disabled
   */
  _calculateWeightDecay(step) {
    if (this.weightDecay === null) {
      return null;
    }
    
    if (step < this.warmupSteps) {
      // Linear increase during warmup
      const progress = step / this.warmupSteps;
      return this.initialWeightDecay + (this.maxWeightDecay - this.initialWeightDecay) * progress;
    } else {
      // Cosine annealing during cooldown
      const progress = (step - this.warmupSteps) / this.annealingSteps;
      const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * progress));
      return this.finalWeightDecay + (this.maxWeightDecay - this.finalWeightDecay) * cosineDecay;
    }
  }
  
  /**
   * Called at the start of training
   * @param {Object} model TensorFlow.js model
   */
  onTrainBegin(model) {
    // Set initial learning rate
    if (model && model.optimizer) {
      model.optimizer.learningRate = this.minLr;
      
      // Initialize momentum if supported
      if (model.optimizer.momentum !== undefined) {
        model.optimizer.momentum = this.maxMomentum;
      }
      
      // Setup weight decay for Adam (not directly supported in tfjs)
      // Will be handled in training loop
    }
    
    // Update UI
    if (this.onLrUpdate) {
      try {
        this.onLrUpdate(this.minLr);
      } catch (e) {
        console.error("Error updating LR display:", e);
      }
    }
    
    // Log schedule information
    if (!this.initialized && this.verbose > 0) {
      logQueue.put(`OneCycleLR: Warmup: ${this.warmupSteps} steps, Annealing: ${this.annealingSteps} steps`);
      logQueue.put(`OneCycleLR: LR range: [${this.minLr.toFixed(6)}, ${this.maxLr.toFixed(6)}, ${this.finalLr.toFixed(6)}]`);
      logQueue.put(`OneCycleLR: Momentum range: [${this.minMomentum.toFixed(4)}, ${this.maxMomentum.toFixed(4)}]`);
      if (this.weightDecay !== null) {
        logQueue.put(`OneCycleLR: Weight decay active: initial=${this.initialWeightDecay}`);
      }
      this.initialized = true;
    }
  }
  
  /**
   * Called at the beginning of each epoch
   * @param {number} epoch Current epoch (0-indexed)
   * @param {Object} model TensorFlow.js model
   */
  onEpochBegin(epoch, model) {
    // Log current settings at the beginning of each epoch
    if (this.verbose > 0 && model && model.optimizer) {
      const currentLr = model.optimizer.learningRate;
      
      let momentumStr = "";
      if (model.optimizer.momentum !== undefined) {
        const currentMomentum = model.optimizer.momentum;
        momentumStr = `, momentum=${currentMomentum.toFixed(4)}`;
      }
      
      let weightDecayStr = "";
      if (this.weightDecay !== null) {
        const currentWd = this._calculateWeightDecay(this.currentStep);
        weightDecayStr = `, weight_decay=${currentWd.toFixed(6)}`;
      }
      
      logQueue.put(`Epoch ${epoch+1} starting with lr=${currentLr.toFixed(6)}${momentumStr}${weightDecayStr}`);
    }
  }
  
  /**
   * Called at the beginning of each batch
   * @param {number} batch Current batch within the epoch
   * @param {Object} model TensorFlow.js model
   */
  onBatchBegin(batch, model) {
    // Skip if we've gone beyond total steps
    if (this.currentStep >= this.totalSteps) {
      return;
    }
    
    // Calculate the current learning rate
    const lr = this._calculateLr(this.currentStep);
    
    // Set the learning rate
    if (model && model.optimizer) {
      model.optimizer.learningRate = lr;
      
      // Set momentum if available
      if (model.optimizer.momentum !== undefined) {
        const momentum = this._calculateMomentum(this.currentStep);
        model.optimizer.momentum = momentum;
        this.history.momentum.push(momentum);
      }
      
      // Weight decay is handled separately in the optimizer step
    }
    
    // Update UI variable if provided (every 5 batches to reduce overhead)
    if (this.onLrUpdate && (batch % 5 === 0 || batch === 0)) {
      try {
        this.onLrUpdate(lr);
      } catch (e) {
        console.error("Error updating LR display:", e);
      }
    }
    
    // Store history for plotting
    this.history.lr.push(lr);
    
    // Increment step counter
    this.currentStep++;
    
    // Return the current lr and weight decay for use in the optimizer
    return {
      lr,
      weightDecay: this.weightDecay !== null ? this._calculateWeightDecay(this.currentStep - 1) : null
    };
  }
  
  /**
   * Called at the end of each epoch
   * @param {number} epoch Current epoch (0-indexed)
   * @param {Object} model TensorFlow.js model
   */
  onEpochEnd(epoch, model) {
    if (this.verbose > 0 && model && model.optimizer) {
      // Log current learning rate at the end of each epoch
      const currentLr = model.optimizer.learningRate;
      
      // Create momentum message if applicable
      let momentumMsg = "";
      if (model.optimizer.momentum !== undefined) {
        const currentMomentum = model.optimizer.momentum;
        momentumMsg = ` with momentum: ${currentMomentum.toFixed(4)}`;
      }
      
      // Create weight decay message if applicable
      let weightDecayMsg = "";
      if (this.weightDecay !== null) {
        const currentWd = this._calculateWeightDecay(this.currentStep - 1);
        weightDecayMsg = `, weight_decay: ${currentWd.toFixed(6)}`;
      }
      
      logQueue.put(`Epoch ${epoch+1} completed with lr: ${currentLr.toFixed(6)}${momentumMsg}${weightDecayMsg}`);
    }
    
    // Update UI
    if (this.onLrUpdate && model && model.optimizer) {
      try {
        this.onLrUpdate(model.optimizer.learningRate);
      } catch (e) {
        console.error("Error updating LR display:", e);
      }
    }
  }
  
  /**
   * Called at the end of training
   */
  onTrainEnd() {
    if (this.verbose > 0) {
      logQueue.put("OneCycleLR schedule completed");
    }
  }
  
  /**
   * Returns the full learning rate schedule as an array for plotting
   * @return {Array} Array of learning rates for each step
   */
  getLrSchedule() {
    return Array.from({length: this.totalSteps}, (_, i) => this._calculateLr(i));
  }
  
  /**
   * Returns the full momentum schedule as an array for plotting
   * @return {Array} Array of momentum values for each step
   */
  getMomentumSchedule() {
    return Array.from({length: this.totalSteps}, (_, i) => this._calculateMomentum(i));
  }
  
  /**
   * Returns the full weight decay schedule as an array for plotting
   * @return {Array|null} Array of weight decay values for each step, or null if weight decay is disabled
   */
  getWeightDecaySchedule() {
    if (this.weightDecay === null) {
      return null;
    }
    return Array.from({length: this.totalSteps}, (_, i) => this._calculateWeightDecay(i));
  }
}

// Export OneCycleLR class
export { OneCycleLR };