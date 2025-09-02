// Complete Web Integration Example for Character-Level Swipe Model
// File: swipe-predictor.ts

import * as ort from 'onnxruntime-web';

interface SwipePoint {
    x: number;
    y: number;
    t: number;
}

interface Beam {
    tokens: number[];
    score: number;
    finished: boolean;
}

export class CharacterSwipePredictor {
    private encoderSession?: ort.InferenceSession;
    private decoderSession?: ort.InferenceSession;
    private tokenizer: any;
    private keyboardLayout: any;
    
    // Special tokens
    private readonly PAD_IDX = 0;
    private readonly EOS_IDX = 1;
    private readonly UNK_IDX = 2;
    private readonly SOS_IDX = 3;
    
    constructor() {
        this.loadTokenizer();
        this.loadKeyboardLayout();
    }
    
    private loadTokenizer() {
        // Load from tokenizer_config.json
        this.tokenizer = {
            charToIdx: {
                '<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3,
                'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9,
                'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15,
                'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20, 'r': 21,
                's': 22, 't': 23, 'u': 24, 'v': 25, 'w': 26, 'x': 27,
                'y': 28, 'z': 29
            },
            idxToChar: {} as any
        };
        
        // Create reverse mapping
        for (const [char, idx] of Object.entries(this.tokenizer.charToIdx)) {
            this.tokenizer.idxToChar[idx] = char;
        }
    }
    
    private loadKeyboardLayout() {
        // QWERTY keyboard layout with positions
        this.keyboardLayout = {
            'q': {x: 18, y: 111}, 'w': {x: 54, y: 111}, 'e': {x: 90, y: 111},
            'r': {x: 126, y: 111}, 't': {x: 162, y: 111}, 'y': {x: 198, y: 111},
            'u': {x: 234, y: 111}, 'i': {x: 270, y: 111}, 'o': {x: 306, y: 111},
            'p': {x: 342, y: 111},
            'a': {x: 36, y: 167}, 's': {x: 72, y: 167}, 'd': {x: 108, y: 167},
            'f': {x: 144, y: 167}, 'g': {x: 180, y: 167}, 'h': {x: 216, y: 167},
            'j': {x: 252, y: 167}, 'k': {x: 288, y: 167}, 'l': {x: 324, y: 167},
            'z': {x: 72, y: 223}, 'x': {x: 108, y: 223}, 'c': {x: 144, y: 223},
            'v': {x: 180, y: 223}, 'b': {x: 216, y: 223}, 'n': {x: 252, y: 223},
            'm': {x: 288, y: 223}
        };
    }
    
    async loadModels(encoderUrl: string, decoderUrl: string) {
        console.log('Loading ONNX models...');
        
        // Configure session options
        const options: ort.InferenceSession.SessionOptions = {
            executionProviders: ['wasm'],  // Use 'webgl' for GPU
            graphOptimizationLevel: 'all'
        };
        
        this.encoderSession = await ort.InferenceSession.create(encoderUrl, options);
        this.decoderSession = await ort.InferenceSession.create(decoderUrl, options);
        
        console.log('Models loaded successfully');
    }
    
    async predictWord(swipePoints: SwipePoint[], beamSize: number = 5): Promise<string> {
        if (!this.encoderSession || !this.decoderSession) {
            throw new Error('Models not loaded');
        }
        
        // 1. Extract features
        const features = this.extractFeatures(swipePoints);
        const nearestKeys = this.findNearestKeys(swipePoints);
        const srcMask = new Float32Array(swipePoints.length).fill(0);
        
        // 2. Prepare encoder inputs
        const encoderInputs = {
            trajectory_features: new ort.Tensor(
                'float32',
                features,
                [1, swipePoints.length, 6]
            ),
            nearest_keys: new ort.Tensor(
                'int64',
                nearestKeys,
                [1, swipePoints.length, 3]
            ),
            src_mask: new ort.Tensor(
                'bool',
                srcMask,
                [1, swipePoints.length]
            )
        };
        
        // 3. Run encoder
        const encoderOutputs = await this.encoderSession.run(encoderInputs);
        const memory = encoderOutputs.encoder_output;
        
        // 4. Run beam search with decoder
        const result = await this.beamSearchDecode(memory, beamSize);
        
        return result;
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
            // Find 3 nearest keys for each point
            const distances: Array<{key: string, dist: number}> = [];
            
            for (const [key, pos] of Object.entries(this.keyboardLayout)) {
                const dist = Math.sqrt(
                    Math.pow(point.x - pos.x, 2) + 
                    Math.pow(point.y - pos.y, 2)
                );
                distances.push({key, dist});
            }
            
            // Sort by distance and take top 3
            distances.sort((a, b) => a.dist - b.dist);
            const top3 = distances.slice(0, 3);
            
            // Convert to indices
            for (const {key} of top3) {
                const idx = this.tokenizer.charToIdx[key] || this.UNK_IDX;
                nearestKeys.push(BigInt(idx));
            }
        }
        
        return new BigInt64Array(nearestKeys);
    }
    
    private async beamSearchDecode(
        memory: ort.Tensor,
        beamSize: number
    ): Promise<string> {
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
                const tgtTokens = new BigInt64Array(
                    beam.tokens.map(t => BigInt(t))
                );
                const tgtMask = new Float32Array(beam.tokens.length).fill(0);
                
                const decoderInputs = {
                    memory: memory,
                    target_tokens: new ort.Tensor(
                        'int64',
                        tgtTokens,
                        [1, beam.tokens.length]
                    ),
                    target_mask: new ort.Tensor(
                        'bool',
                        tgtMask,
                        [1, beam.tokens.length]
                    )
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
                const topK = this.getTopK(probs, beamSize);
                
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
        
        // Decode best beam
        return this.decodeTokens(beams[0].tokens);
    }
    
    private softmax(logits: Float32Array): Float32Array {
        const maxLogit = Math.max(...logits);
        const expScores = logits.map(l => Math.exp(l - maxLogit));
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
            
            const char = this.tokenizer.idxToChar[token];
            if (char && !char.startsWith('<')) {
                word += char;
            }
        }
        
        return word;
    }
}

// Usage example
async function demo() {
    const predictor = new CharacterSwipePredictor();
    
    // Load models
    await predictor.loadModels(
        '/models/swipe_model_character.onnx',
        '/models/swipe_decoder_character.onnx'
    );
    
    // Example swipe for "hello"
    const swipePoints: SwipePoint[] = [
        {x: 216, y: 167, t: 0},   // h
        {x: 90, y: 111, t: 100},  // e
        {x: 324, y: 167, t: 200}, // l
        {x: 324, y: 167, t: 300}, // l
        {x: 306, y: 111, t: 400}  // o
    ];
    
    const prediction = await predictor.predictWord(swipePoints);
    console.log('Predicted word:', prediction); // Should output: "hello"
}

export default CharacterSwipePredictor;
