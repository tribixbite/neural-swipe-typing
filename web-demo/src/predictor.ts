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
    
    // Special tokens
    private readonly PAD_IDX = 0;
    private readonly EOS_IDX = 1;
    private readonly UNK_IDX = 2;
    private readonly SOS_IDX = 3;
    
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
        return {
            'q': {x: 18, y: 111}, 'w': {x: 54, y: 111}, 'e': {x: 90, y: 111},
            'r': {x: 126, y: 111}, 't': {x: 162, y: 111}, 'y': {x: 198, y: 111},
            'u': {x: 234, y: 111}, 'i': {x: 270, y: 111}, 'o': {x: 306, y: 111},
            'p': {x: 342, y: 111},
            'a': {x: 36, y: 155}, 's': {x: 72, y: 155}, 'd': {x: 108, y: 155},
            'f': {x: 144, y: 155}, 'g': {x: 180, y: 155}, 'h': {x: 216, y: 155},
            'j': {x: 252, y: 155}, 'k': {x: 288, y: 155}, 'l': {x: 324, y: 155},
            'z': {x: 72, y: 199}, 'x': {x: 108, y: 199}, 'c': {x: 144, y: 199},
            'v': {x: 180, y: 199}, 'b': {x: 216, y: 199}, 'n': {x: 252, y: 199},
            'm': {x: 288, y: 199}
        };
    }
    
    async predict(swipePoints: SwipePoint[], topK: number = 5): Promise<Array<{word: string, score: number}>> {
        if (!this.encoderSession || !this.decoderSession || !this.tokenizer) {
            throw new Error('Models not loaded');
        }
        
        // Prepare input features
        const features = this.extractFeatures(swipePoints);
        const nearestKeys = this.findNearestKeys(swipePoints);
        // Boolean masks must be Uint8Array for ONNX Runtime
        const srcMask = new Uint8Array(swipePoints.length).fill(0);
        
        // Run encoder
        const encoderInputs = {
            trajectory_features: new ort.Tensor('float32', features, [1, swipePoints.length, 6]),
            nearest_keys: new ort.Tensor('int64', nearestKeys, [1, swipePoints.length]),
            src_mask: new ort.Tensor('bool', srcMask, [1, swipePoints.length])
        };
        
        const encoderOutputs = await this.encoderSession.run(encoderInputs);
        const memory = encoderOutputs.encoder_output;
        
        // Run beam search
        const predictions = await this.beamSearchDecode(memory, topK);
        
        return predictions;
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
        const maxLength = 20;
        let beams: Beam[] = [{
            tokens: [this.SOS_IDX],
            score: 0,
            finished: false
        }];
        
        for (let step = 0; step < maxLength; step++) {
            const allCandidates: Beam[] = [];
            
            for (const beam of beams) {
                if (beam.finished) {
                    allCandidates.push(beam);
                    continue;
                }
                
                // Prepare decoder inputs
                const tgtTokens = new BigInt64Array(beam.tokens.map(t => BigInt(t)));
                // Boolean masks must be Uint8Array
                const tgtMask = new Uint8Array(beam.tokens.length).fill(0);
                const srcMask = new Uint8Array(memory.dims[1] as number).fill(0);
                
                const decoderInputs = {
                    memory: memory,
                    target_tokens: new ort.Tensor('int64', tgtTokens, [1, beam.tokens.length]),
                    target_mask: new ort.Tensor('bool', tgtMask, [1, beam.tokens.length]),
                    src_mask: new ort.Tensor('bool', srcMask, [1, memory.dims[1] as number])
                };
                
                // Run decoder
                const decoderOutputs = await this.decoderSession!.run(decoderInputs);
                const logits = decoderOutputs.logits;
                
                // Get last token predictions
                const logitsData = logits.data as Float32Array;
                const vocabSize = 30;
                const lastLogits = logitsData.slice(-vocabSize);
                
                // Apply softmax and get top k
                const probs = this.softmax(lastLogits);
                const topK = this.getTopK(probs, Math.min(beamSize, vocabSize));
                
                // Create new beams
                for (const {idx, prob} of topK) {
                    allCandidates.push({
                        tokens: [...beam.tokens, idx],
                        score: beam.score + Math.log(prob),
                        finished: idx === this.EOS_IDX
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
        const predictions = beams.map(beam => ({
            word: this.decodeTokens(beam.tokens),
            score: Math.exp(beam.score / beam.tokens.length) // Normalize by length
        }));
        
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