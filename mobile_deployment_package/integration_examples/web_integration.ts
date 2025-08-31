// Web TypeScript integration example
import * as ort from 'onnxruntime-web';

class MobileSwipePredictor {
    private session: ort.InferenceSession | null = null;
    private charToIdx = {
        '<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3,
        'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10,
        'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17,
        'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24,
        'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29
    };
    private idxToChar = Object.fromEntries(
        Object.entries(this.charToIdx).map(([k, v]) => [v, k])
    );
    
    async loadModel(modelUrl: string) {
        this.session = await ort.InferenceSession.create(modelUrl);
        console.log('Mobile swipe model loaded');
    }
    
    async predictWord(swipePoints: {x: number, y: number, t: number}[]): Promise<string> {
        if (!this.session) throw new Error('Model not loaded');
        
        // Extract 6D features: x, y, vx, vy, ax, ay
        const features = this.extractFeatures(swipePoints);
        const inputTensor = new ort.Tensor('float32', features, [1, features.length / 6, 6]);
        
        // Run inference
        const feeds = { trajectory_input: inputTensor };
        const results = await this.session.run(feeds);
        
        // Decode character probabilities to word
        return this.decodeOutput(results.character_output);
    }
    
    private extractFeatures(points: {x: number, y: number, t: number}[]): Float32Array {
        if (points.length < 2) return new Float32Array(0);
        
        const features = [];
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            
            // Normalize coordinates to [0, 1]
            const x = p.x / 360;
            const y = p.y / 215;
            
            // Calculate velocities
            let vx = 0, vy = 0;
            if (i > 0) {
                const prev = points[i - 1];
                const dt = Math.max(p.t - prev.t, 1); // Avoid division by zero
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
                const vx_prev = (prev.x - prev2.x) / dt2;
                const vy_prev = (prev.y - prev2.y) / dt2;
                ax = (vx - vx_prev) / dt1;
                ay = (vy - vy_prev) / dt1;
            }
            
            // Clip velocities and accelerations
            vx = Math.max(-1000, Math.min(1000, vx));
            vy = Math.max(-1000, Math.min(1000, vy));
            ax = Math.max(-500, Math.min(500, ax));
            ay = Math.max(-500, Math.min(500, ay));
            
            features.push(x, y, vx / 1000, vy / 1000, ax / 500, ay / 500);
        }
        
        return new Float32Array(features);
    }
    
    private decodeOutput(tensor: ort.Tensor): string {
        const data = tensor.data as Float32Array;
        const [batchSize, seqLen, vocabSize] = tensor.dims;
        
        let word = '';
        for (let i = 0; i < seqLen; i++) {
            let maxIdx = 0;
            let maxVal = -Infinity;
            
            for (let j = 0; j < vocabSize; j++) {
                const val = data[i * vocabSize + j];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            
            const char = this.idxToChar[maxIdx];
            if (char === '<eos>') break;
            if (char && char !== '<pad>' && char !== '<sos>') {
                word += char;
            }
        }
        
        return word;
    }
}

// Usage example
async function main() {
    const predictor = new MobileSwipePredictor();
    await predictor.loadModel('swipe_model_web.onnx');
    
    // Example swipe points for "hello"
    const swipePoints = [
        {x: 96, y: 167, t: 0},    // h
        {x: 124, y: 167, t: 50},  // e
        {x: 152, y: 167, t: 100}, // l
        {x: 152, y: 167, t: 150}, // l
        {x: 208, y: 167, t: 200}  // o
    ];
    
    const prediction = await predictor.predictWord(swipePoints);
    console.log('Predicted word:', prediction);
}
