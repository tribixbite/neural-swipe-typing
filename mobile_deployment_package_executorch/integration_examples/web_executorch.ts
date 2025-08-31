// Web TypeScript integration (ONNX.js)
// Note: Same as previous ONNX integration, but model outputs fixed 20 characters
import * as ort from 'onnxruntime-web';

class MobileSwipePredictorExecutorchWeb {
    private session: ort.InferenceSession | null = null;
    
    async loadModel(modelUrl: string) {
        this.session = await ort.InferenceSession.create(modelUrl);
        console.log('ExecuTorch-compatible ONNX model loaded');
    }
    
    async predictWord(swipePoints: {x: number, y: number, t: number}[]): Promise<string> {
        if (!this.session) throw new Error('Model not loaded');
        
        const features = this.extractFeatures(swipePoints);
        const inputTensor = new ort.Tensor('float32', features, [1, features.length / 6, 6]);
        
        const feeds = { trajectory_input: inputTensor };
        const results = await this.session.run(feeds);
        
        // Decode fixed-length output (20 characters)
        return this.decodeFixedLengthOutput(results.character_output);
    }
    
    private decodeFixedLengthOutput(tensor: ort.Tensor): string {
        const data = tensor.data as Float32Array;
        const [batchSize, seqLen, vocabSize] = tensor.dims;  // [1, 20, 30]
        
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
            
            // Convert index to character
            if (maxIdx === 0) continue;      // <pad>
            if (maxIdx === 1) break;         // <eos> 
            if (maxIdx === 2) continue;      // <unk>
            if (maxIdx === 3) continue;      // <sos>
            if (maxIdx >= 4 && maxIdx <= 29) {
                word += String.fromCharCode('a'.charCodeAt(0) + (maxIdx - 4));
            }
        }
        
        return word;
    }
    
    // Same feature extraction as before...
    private extractFeatures(points: {x: number, y: number, t: number}[]): Float32Array {
        // Implementation same as previous version
        return new Float32Array([]);  // Placeholder
    }
}
