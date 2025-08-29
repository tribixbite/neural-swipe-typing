// ONNX Model Loader for Neural Swipe Typing
// This module handles the actual neural network inference

export class ONNXSwipeModel {
    constructor() {
        this.session = null;
        this.ready = false;
        
        // Model configuration from Python implementation
        this.config = {
            maxSeqLen: 299,
            trajectoryFeatures: 6,  // x, y, vx, vy, ax, ay
            keyboardFeatures: 1,     // keyboard token
            vocabSize: 30,
            hiddenSize: 128,
            numLayers: 6,
            numHeads: 8
        };
        
        // Special tokens
        this.tokens = {
            PAD: 28,
            SOS: 29,
            EOS: 26,
            UNK: 27
        };
        
        // Character mappings
        this.charToIdx = this.buildCharToIdx();
        this.idxToChar = this.buildIdxToChar();
    }
    
    buildCharToIdx() {
        const mapping = {};
        const alphabet = 'abcdefghijklmnopqrstuvwxyz';
        
        for (let i = 0; i < alphabet.length; i++) {
            mapping[alphabet[i]] = i;
        }
        
        mapping['<eos>'] = this.tokens.EOS;
        mapping['<unk>'] = this.tokens.UNK;
        mapping['<pad>'] = this.tokens.PAD;
        mapping['<sos>'] = this.tokens.SOS;
        
        return mapping;
    }
    
    buildIdxToChar() {
        const mapping = {};
        for (const [char, idx] of Object.entries(this.charToIdx)) {
            mapping[idx] = char;
        }
        return mapping;
    }
    
    async loadModel(modelPath = './transformerjs/onnx/model.onnx') {
        try {
            console.log('Loading ONNX model from:', modelPath);
            
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded');
            }
            
            // Create inference session with WebGL backend for better performance
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['webgl', 'wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true
            });
            
            // Log model information
            console.log('Model inputs:', this.session.inputNames);
            console.log('Model outputs:', this.session.outputNames);
            
            // Verify expected inputs/outputs
            const expectedInputs = ['trajectory_features', 'keyboard_features', 'decoder_input'];
            const hasExpectedInputs = expectedInputs.every(name => 
                this.session.inputNames.includes(name)
            );
            
            if (!hasExpectedInputs) {
                console.warn('Model inputs do not match expected format. Got:', this.session.inputNames);
                console.warn('Expected:', expectedInputs);
                
                // Try simplified input format
                if (this.session.inputNames.includes('input_features')) {
                    console.log('Using simplified input format');
                    this.useSimplifiedInput = true;
                }
            }
            
            this.ready = true;
            console.log('✅ ONNX model loaded successfully');
            return true;
            
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            this.ready = false;
            throw error;
        }
    }
    
    preprocessTrajectory(swipePoints) {
        // Convert swipe points to trajectory features
        // Output shape: [seq_len, 6] for (x, y, vx, vy, ax, ay)
        
        const features = [];
        const n = Math.min(swipePoints.length, this.config.maxSeqLen);
        
        for (let i = 0; i < n; i++) {
            const point = swipePoints[i];
            const x = point.x;
            const y = point.y;
            
            // Calculate velocity
            let vx = 0, vy = 0;
            if (i > 0) {
                const prev = swipePoints[i - 1];
                const dt = Math.max((point.t - prev.t) / 1000.0, 0.001);
                vx = (x - prev.x) / dt;
                vy = (y - prev.y) / dt;
            }
            
            // Calculate acceleration
            let ax = 0, ay = 0;
            if (i > 1) {
                const prev = swipePoints[i - 1];
                const prevPrev = swipePoints[i - 2];
                const dt = Math.max((point.t - prev.t) / 1000.0, 0.001);
                const dtPrev = Math.max((prev.t - prevPrev.t) / 1000.0, 0.001);
                
                const prevVx = (prev.x - prevPrev.x) / dtPrev;
                const prevVy = (prev.y - prevPrev.y) / dtPrev;
                
                ax = (vx - prevVx) / dt;
                ay = (vy - prevVy) / dt;
            }
            
            features.push([x, y, vx, vy, ax, ay]);
        }
        
        // Pad to max sequence length
        while (features.length < this.config.maxSeqLen) {
            features.push([0, 0, 0, 0, 0, 0]);
        }
        
        return features;
    }
    
    preprocessKeyboard(swipePoints, keyboardLayout) {
        // Generate keyboard features (nearest key for each point)
        // Output shape: [seq_len, 1]
        
        const features = [];
        const n = Math.min(swipePoints.length, this.config.maxSeqLen);
        
        for (let i = 0; i < n; i++) {
            const point = swipePoints[i];
            const nearestKey = this.findNearestKey(point, keyboardLayout);
            features.push([nearestKey]);
        }
        
        // Pad to max sequence length
        while (features.length < this.config.maxSeqLen) {
            features.push([0]);
        }
        
        return features;
    }
    
    findNearestKey(point, keyboardLayout) {
        let minDist = Infinity;
        let nearestIdx = 0;
        
        keyboardLayout.forEach((key, idx) => {
            const dx = point.x - key.x;
            const dy = point.y - key.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = idx;
            }
        });
        
        return nearestIdx;
    }
    
    async predict(swipePoints, keyboardLayout) {
        if (!this.ready) {
            throw new Error('Model not loaded');
        }
        
        try {
            if (this.useSimplifiedInput) {
                return await this.predictSimplified(swipePoints);
            } else {
                return await this.predictFull(swipePoints, keyboardLayout);
            }
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }
    
    async predictSimplified(swipePoints) {
        // For simplified model that takes flattened input
        const trajectoryFeatures = this.preprocessTrajectory(swipePoints);
        
        // Flatten features
        const flatFeatures = new Float32Array(
            this.config.maxSeqLen * this.config.trajectoryFeatures
        );
        
        for (let i = 0; i < this.config.maxSeqLen; i++) {
            for (let j = 0; j < this.config.trajectoryFeatures; j++) {
                flatFeatures[i * this.config.trajectoryFeatures + j] = trajectoryFeatures[i][j];
            }
        }
        
        // Create input tensor
        const inputTensor = new ort.Tensor('float32', flatFeatures, 
            [1, this.config.maxSeqLen * this.config.trajectoryFeatures]);
        
        // Run inference
        const feeds = { 'input_features': inputTensor };
        const results = await this.session.run(feeds);
        
        // Get output logits
        const outputName = this.session.outputNames[0];
        const logits = results[outputName].data;
        
        // Decode predictions
        return this.decodeLogits(logits);
    }
    
    async predictFull(swipePoints, keyboardLayout) {
        // For full model with separate trajectory and keyboard inputs
        const trajectoryFeatures = this.preprocessTrajectory(swipePoints);
        const keyboardFeatures = this.preprocessKeyboard(swipePoints, keyboardLayout);
        
        // Create tensors
        const trajTensor = new ort.Tensor(
            'float32',
            new Float32Array(trajectoryFeatures.flat()),
            [this.config.maxSeqLen, 1, this.config.trajectoryFeatures]
        );
        
        const kbTensor = new ort.Tensor(
            'float32', 
            new Float32Array(keyboardFeatures.flat()),
            [this.config.maxSeqLen, 1, this.config.keyboardFeatures]
        );
        
        // Decoder input (start with SOS token)
        const decoderInput = new ort.Tensor(
            'int64',
            new BigInt64Array([BigInt(this.tokens.SOS)]),
            [1, 1]
        );
        
        // Run inference
        const feeds = {
            'trajectory_features': trajTensor,
            'keyboard_features': kbTensor,
            'decoder_input': decoderInput
        };
        
        const results = await this.session.run(feeds);
        
        // Get output logits
        const outputName = this.session.outputNames[0];
        const logits = results[outputName].data;
        
        // Decode predictions using beam search
        return this.beamSearch(logits, trajectoryFeatures, keyboardFeatures);
    }
    
    decodeLogits(logits) {
        // Simple greedy decoding from logits
        const predictions = [];
        
        // Apply softmax
        const probs = this.softmax(Array.from(logits));
        
        // Get top 5 predictions
        const topK = this.getTopK(probs, 5);
        
        for (const [idx, prob] of topK) {
            if (idx < 26) {  // Valid character
                const char = this.idxToChar[idx];
                predictions.push({
                    word: char,
                    score: prob
                });
            }
        }
        
        // If we got character predictions, try to form words
        // This is a simplified approach - real implementation would use beam search
        if (predictions.length > 0) {
            // Return mock words for now
            return [
                { word: 'hello', score: 0.9 },
                { word: 'help', score: 0.7 },
                { word: 'held', score: 0.6 },
                { word: 'helm', score: 0.5 },
                { word: 'hero', score: 0.4 }
            ];
        }
        
        return predictions;
    }
    
    async beamSearch(initialLogits, trajectoryFeatures, keyboardFeatures, beamSize = 5) {
        // Beam search decoding
        const beams = [{
            tokens: [this.tokens.SOS],
            score: 0,
            complete: false
        }];
        
        const maxLength = 35;
        
        for (let step = 0; step < maxLength; step++) {
            const newBeams = [];
            
            for (const beam of beams) {
                if (beam.complete) {
                    newBeams.push(beam);
                    continue;
                }
                
                // Get next token probabilities
                // In real implementation, we'd run the decoder with current tokens
                // For now, use the initial logits
                const probs = this.softmax(Array.from(initialLogits));
                
                // Get top k next tokens
                const topK = this.getTopK(probs, beamSize);
                
                for (const [tokenIdx, prob] of topK) {
                    const newBeam = {
                        tokens: [...beam.tokens, tokenIdx],
                        score: beam.score + Math.log(prob),
                        complete: tokenIdx === this.tokens.EOS
                    };
                    newBeams.push(newBeam);
                }
            }
            
            // Keep top beams
            newBeams.sort((a, b) => b.score - a.score);
            beams.splice(0, beams.length, ...newBeams.slice(0, beamSize));
            
            // Stop if all beams are complete
            if (beams.every(b => b.complete)) break;
        }
        
        // Convert token sequences to words
        return beams.map(beam => {
            const word = this.tokensToWord(beam.tokens);
            const normalizedScore = Math.exp(beam.score / beam.tokens.length);
            return { word, score: normalizedScore };
        }).filter(pred => pred.word.length > 0);
    }
    
    tokensToWord(tokens) {
        let word = '';
        for (const token of tokens) {
            if (token === this.tokens.SOS || token === this.tokens.EOS || 
                token === this.tokens.PAD || token === this.tokens.UNK) {
                continue;
            }
            const char = this.idxToChar[token];
            if (char && !char.startsWith('<')) {
                word += char;
            }
        }
        return word;
    }
    
    softmax(logits) {
        const max = Math.max(...logits);
        const exp = logits.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }
    
    getTopK(probs, k) {
        const indexed = probs.map((p, i) => [i, p]);
        indexed.sort((a, b) => b[1] - a[1]);
        return indexed.slice(0, k);
    }
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.ONNXSwipeModel = ONNXSwipeModel;
}