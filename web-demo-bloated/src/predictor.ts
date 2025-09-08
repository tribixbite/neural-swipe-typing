// Use global ort
declare const ort: any;

import { SwipePoint } from './swipe-tracker';

interface Beam {
    tokens: number[];
    score: number;
    finished: boolean;
}

interface Tokenizer {
    charToIdx: Record<string, number>;
    idxToChar: Record<number, string>;
}

export class SwipePredictor {
    private encoderSession?: ort.InferenceSession;
    private decoderSession?: ort.InferenceSession;
    private tokenizer?: Tokenizer;
    private keyboardLayout?: Record<string, {x: number, y: number}>;
    
    // Special tokens - Fixed to match deployment_package/tokenizer_config.json
    private readonly PAD_IDX = 0;
    private readonly UNK_IDX = 1;
    private readonly SOS_IDX = 2;
    private readonly EOS_IDX = 3;
    
    constructor() {}
    
    async loadEncoder(url: string) {
        console.log('Loading encoder model...');
        const options: ort.InferenceSession.SessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        
        this.encoderSession = await ort.InferenceSession.create(url, options);
        console.log('Encoder loaded');
    }
    
    async loadDecoder(url: string) {
        console.log('Loading decoder model...');
        const options: ort.InferenceSession.SessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        
        this.decoderSession = await ort.InferenceSession.create(url, options);
        console.log('Decoder loaded');
    }
    
    // Load a single combined model (for mobile version)
    async loadSingleModel(url: string) {
        console.log('Loading combined model...');
        const options: ort.InferenceSession.SessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        
        this.encoderSession = await ort.InferenceSession.create(url, options);
        // For single model, encoder does both encoding and decoding
        this.decoderSession = this.encoderSession;
        console.log('Combined model loaded');
    }
    
    async loadTokenizer(url: string) {
        console.log('Loading tokenizer...');
        const response = await fetch(url);
        const config = await response.json();
        
        this.tokenizer = {
            charToIdx: config.char_to_idx,
            idxToChar: config.idx_to_char
        };
        
        // Load keyboard layout
        this.keyboardLayout = config.keyboard_layout || this.getDefaultKeyboardLayout();
        
        console.log('Tokenizer loaded');
    }
    
    private getDefaultKeyboardLayout() {
        // Must match the keyboard.ts layout - no number row, fills 360x215 canvas
        return {
            // First row (qwerty) - top third of canvas
            'q': {x: 18, y: 53}, 'w': {x: 54, y: 53}, 'e': {x: 90, y: 53},
            'r': {x: 126, y: 53}, 't': {x: 162, y: 53}, 'y': {x: 198, y: 53},
            'u': {x: 234, y: 53}, 'i': {x: 270, y: 53}, 'o': {x: 306, y: 53},
            'p': {x: 342, y: 53},
            
            // Second row (asdf) - middle third of canvas
            'a': {x: 36, y: 107}, 's': {x: 72, y: 107}, 'd': {x: 108, y: 107},
            'f': {x: 144, y: 107}, 'g': {x: 180, y: 107}, 'h': {x: 216, y: 107},
            'j': {x: 252, y: 107}, 'k': {x: 288, y: 107}, 'l': {x: 324, y: 107},
            
            // Third row (zxcv) - bottom third of canvas
            'z': {x: 72, y: 161}, 'x': {x: 108, y: 161}, 'c': {x: 144, y: 161},
            'v': {x: 180, y: 161}, 'b': {x: 216, y: 161}, 'n': {x: 252, y: 161},
            'm': {x: 288, y: 161}
        };
    }
    
    private padOrTruncatePoints(points: SwipePoint[], targetLength: number): SwipePoint[] {
        if (points.length >= targetLength) {
            // Truncate if too long
            return points.slice(0, targetLength);
        }
        
        // Pad with last point if too short
        const padded = [...points];
        const lastPoint = points[points.length - 1] || {x: 0, y: 0, t: 0};
        
        while (padded.length < targetLength) {
            padded.push({...lastPoint});
        }
        
        return padded;
    }
    
    async predict(swipePoints: SwipePoint[], topK: number = 5): Promise<Array<{word: string, score: number}>> {
        if (!this.encoderSession || !this.decoderSession || !this.tokenizer) {
            throw new Error('Models not loaded');
        }
        
        console.log('Starting prediction with', swipePoints.length, 'points');
        
        // ONNX models now support the full 150 sequence length from training
        const FIXED_SEQ_LENGTH = 150;
        const paddedPoints = this.padOrTruncatePoints(swipePoints, FIXED_SEQ_LENGTH);
        
        // Prepare input features
        const features = this.extractFeatures(paddedPoints);
        const nearestKeys = this.findNearestKeys(paddedPoints);
        // Boolean masks must be Uint8Array for ONNX Runtime
        // Mask should be 1 for padding positions, 0 for real data
        const srcMask = new Uint8Array(FIXED_SEQ_LENGTH);
        for (let i = swipePoints.length; i < FIXED_SEQ_LENGTH; i++) {
            srcMask[i] = 1;  // Mark padded positions
        }
        
        console.log('Features shape:', [1, FIXED_SEQ_LENGTH, 6]);
        console.log('Nearest keys shape:', [1, FIXED_SEQ_LENGTH]);
        console.log('Src mask shape:', [1, FIXED_SEQ_LENGTH]);
        
        // Run encoder
        const encoderInputs = {
            trajectory_features: new ort.Tensor('float32', features, [1, FIXED_SEQ_LENGTH, 6]),
            nearest_keys: new ort.Tensor('int64', nearestKeys, [1, FIXED_SEQ_LENGTH]),
            src_mask: new ort.Tensor('bool', srcMask, [1, FIXED_SEQ_LENGTH])
        };
        
        console.log('Running encoder...');
        try {
            const encoderOutputs = await this.encoderSession.run(encoderInputs);
            console.log('Encoder outputs:', Object.keys(encoderOutputs));
            const memory = encoderOutputs.encoder_output;
            console.log('Memory tensor shape:', memory.dims);
            
            // Run beam search
            console.log('Starting beam search decode...');
            const predictions = await this.beamSearchDecode(memory, topK);
            console.log('Predictions:', predictions);
            
            return predictions;
        } catch (error: any) {
            console.error('Encoder/Decoder error:', error);
            console.error('Error message:', error?.message);
            console.error('Error stack:', error?.stack);
            throw error;
        }
    }
    
    private extractFeatures(points: SwipePoint[]): Float32Array {
        const features: number[] = [];
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            
            // Normalize coordinates
            const x = p.x / 360;
            const y = p.y / 215;
            
            // Calculate velocities
            let vx = 0, vy = 0;
            if (i > 0) {
                const prev = points[i - 1];
                const dt = Math.max(p.t - prev.t, 1);
                vx = (p.x - prev.x) / dt;
                vy = (p.y - prev.y) / dt;
            }
            
            // Calculate accelerations
            let ax = 0, ay = 0;
            if (i > 1) {
                const prev = points[i - 1];
                const prev2 = points[i - 2];
                const dt1 = Math.max(p.t - prev.t, 1);
                const dt2 = Math.max(prev.t - prev2.t, 1);
                
                const vxPrev = (prev.x - prev2.x) / dt2;
                const vyPrev = (prev.y - prev2.y) / dt2;
                
                ax = (vx - vxPrev) / dt1;
                ay = (vy - vyPrev) / dt1;
            }
            
            // Normalize and clip
            vx = Math.max(-1, Math.min(1, vx / 1000));
            vy = Math.max(-1, Math.min(1, vy / 1000));
            ax = Math.max(-1, Math.min(1, ax / 500));
            ay = Math.max(-1, Math.min(1, ay / 500));
            
            features.push(x, y, vx, vy, ax, ay);
        }
        
        return new Float32Array(features);
    }
    
    private findNearestKeys(points: SwipePoint[]): BigInt64Array {
        const nearestKeys: bigint[] = [];
        
        for (const point of points) {
            // Find nearest key
            let nearestKey = this.UNK_IDX;
            let minDist = Infinity;
            
            for (const [key, pos] of Object.entries(this.keyboardLayout!)) {
                const dist = Math.sqrt(
                    Math.pow(point.x - pos.x, 2) + 
                    Math.pow(point.y - pos.y, 2)
                );
                
                if (dist < minDist) {
                    minDist = dist;
                    nearestKey = this.tokenizer!.charToIdx[key] || this.UNK_IDX;
                }
            }
            
            nearestKeys.push(BigInt(nearestKey));
        }
        
        return new BigInt64Array(nearestKeys);
    }
    
    private async beamSearchDecode(
        memory: ort.Tensor,
        beamSize: number
    ): Promise<Array<{word: string, score: number}>> {
        const DECODER_SEQ_LENGTH = 20;  // Fixed decoder sequence length
        const maxGeneratedTokens = 15;  // Max tokens to actually generate
        
        let beams: Beam[] = [{
            tokens: [this.SOS_IDX],
            score: 0,
            finished: false
        }];
        
        // console.log('Beam search - memory shape:', memory.dims);
        
        for (let step = 0; step < maxGeneratedTokens; step++) {
            // console.log(`Beam search step ${step}`);
            const allCandidates: Beam[] = [];
            
            for (const beam of beams) {
                if (beam.finished) {
                    allCandidates.push(beam);
                    continue;
                }
                
                // Prepare decoder inputs - pad to fixed length
                const paddedTokens = new BigInt64Array(DECODER_SEQ_LENGTH);
                for (let i = 0; i < beam.tokens.length && i < DECODER_SEQ_LENGTH; i++) {
                    paddedTokens[i] = BigInt(beam.tokens[i]);
                }
                // Pad rest with PAD_IDX
                for (let i = beam.tokens.length; i < DECODER_SEQ_LENGTH; i++) {
                    paddedTokens[i] = BigInt(this.PAD_IDX);
                }
                
                // Boolean masks must be Uint8Array
                // Mask should be 1 for padded positions, 0 for real tokens
                const tgtMask = new Uint8Array(DECODER_SEQ_LENGTH);
                for (let i = beam.tokens.length; i < DECODER_SEQ_LENGTH; i++) {
                    tgtMask[i] = 1;  // Mark padded positions
                }
                const srcMask = new Uint8Array(memory.dims[1] as number).fill(0);
                
                // console.log('Decoder input shapes:');
                // console.log('  target_tokens:', [1, DECODER_SEQ_LENGTH]);
                // console.log('  target_mask:', [1, DECODER_SEQ_LENGTH]);
                // console.log('  src_mask:', [1, memory.dims[1]]);
                // console.log('  Current token count:', beam.tokens.length);
                
                const decoderInputs = {
                    memory: memory,
                    target_tokens: new ort.Tensor('int64', paddedTokens, [1, DECODER_SEQ_LENGTH]),
                    target_mask: new ort.Tensor('bool', tgtMask, [1, DECODER_SEQ_LENGTH]),
                    src_mask: new ort.Tensor('bool', srcMask, [1, memory.dims[1] as number])
                };
                
                // Run decoder
                let decoderOutputs: any;
                try {
                    // console.log('Running decoder...');
                    decoderOutputs = await this.decoderSession!.run(decoderInputs);
                    // console.log('Decoder outputs:', Object.keys(decoderOutputs));
                    // const logits = decoderOutputs.logits;
                    // console.log('Logits shape:', logits.dims);
                } catch (decodeError: any) {
                    console.error('Decoder run failed:', decodeError);
                    console.error('Error details:', decodeError?.message);
                    throw decodeError;
                }
                const logits = decoderOutputs.logits;
                
                // Get predictions for the position after the last real token
                // logits shape is [1, DECODER_SEQ_LENGTH, 30]
                const logitsData = logits.data as Float32Array;
                const vocabSize = 30;
                // We want the logits at position beam.tokens.length - 1 (0-indexed)
                const tokenPosition = Math.min(beam.tokens.length - 1, DECODER_SEQ_LENGTH - 1);
                const startIdx = tokenPosition * vocabSize;
                const endIdx = startIdx + vocabSize;
                const relevantLogits = logitsData.slice(startIdx, endIdx);
                // console.log(`Getting logits for position ${tokenPosition}, indices ${startIdx}-${endIdx}`);
                
                // Apply softmax and get top k
                const probs = this.softmax(relevantLogits);
                const topK = this.getTopK(probs, Math.min(beamSize, vocabSize));
                
                // Create new beams
                for (const {idx, prob} of topK) {
                    const newTokens = [...beam.tokens, idx];
                    const finished = idx === this.EOS_IDX;
                    
                    // Debug: log what character we're adding
                    if (step < 3) {  // Only log first few steps
                        const char = this.tokenizer?.idxToChar[idx] || `<${idx}>`;
                        console.log(`Step ${step}, beam: adding token ${idx} = "${char}", prob=${prob.toFixed(3)}`);
                    }
                    
                    allCandidates.push({
                        tokens: newTokens,
                        score: beam.score + Math.log(prob),
                        finished: finished
                    });
                }
            }
            
            // Keep top beams
            beams = allCandidates
                .sort((a, b) => b.score - a.score)
                .slice(0, beamSize);
            
            // Check if all finished
            if (beams.every(b => b.finished)) break;
        }
        
        // Decode and return predictions
        const predictions = beams.map(beam => {
            const word = this.decodeTokens(beam.tokens);
            console.log(`Beam tokens: [${beam.tokens.join(',')}] -> "${word}"`);
            return {
                word: word,
                score: Math.exp(beam.score / beam.tokens.length) // Normalize by length
            };
        });
        
        // Filter out empty predictions
        return predictions.filter(p => p.word.length > 0);
    }
    
    private softmax(logits: Float32Array): Float32Array {
        const maxLogit = Math.max(...logits);
        const expScores = Array.from(logits).map(l => Math.exp(l - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return new Float32Array(expScores.map(e => e / sumExp));
    }
    
    private getTopK(
        probs: Float32Array,
        k: number
    ): Array<{idx: number, prob: number}> {
        const indexed = Array.from(probs).map((prob, idx) => ({idx, prob}));
        indexed.sort((a, b) => b.prob - a.prob);
        return indexed.slice(0, k);
    }
    
    private decodeTokens(tokens: number[]): string {
        let word = '';
        
        for (const token of tokens) {
            if (token === this.EOS_IDX) break;
            if (token === this.SOS_IDX || token === this.PAD_IDX) continue;
            
            const char = this.tokenizer!.idxToChar[token];
            if (char && !char.startsWith('<')) {
                word += char;
            }
        }
        
        return word;
    }
}