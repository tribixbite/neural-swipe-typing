import * as ort from 'onnxruntime-node';

async function testDecoder() {
    try {
        console.log('Loading decoder model...');
        const session = await ort.InferenceSession.create('../deployment_package/swipe_decoder_character.onnx');
        
        console.log('Model loaded!');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
        
        // Create test inputs based on encoder output
        const batchSize = 1;
        const encSeqLength = 50;  // From encoder
        const decSeqLength = 20;  // Decoder expects fixed length of 20
        
        // memory: [batch, enc_sequence, 256]
        const memory = new Float32Array(batchSize * encSeqLength * 256);
        for (let i = 0; i < memory.length; i++) {
            memory[i] = Math.random();
        }
        
        // target_tokens: [batch, dec_sequence]
        const targetTokens = new BigInt64Array(batchSize * decSeqLength);
        targetTokens[0] = BigInt(3); // SOS token
        for (let i = 1; i < targetTokens.length; i++) {
            targetTokens[i] = BigInt(Math.floor(Math.random() * 26) + 4);
        }
        
        // src_mask: [batch, enc_sequence] - BOOL
        const srcMask = new Uint8Array(batchSize * encSeqLength);
        
        // target_mask: [batch, dec_sequence] - BOOL  
        const targetMask = new Uint8Array(batchSize * decSeqLength);
        
        const inputs = {
            memory: new ort.Tensor('float32', memory, [batchSize, encSeqLength, 256]),
            target_tokens: new ort.Tensor('int64', targetTokens, [batchSize, decSeqLength]),
            src_mask: new ort.Tensor('bool', srcMask, [batchSize, encSeqLength]),
            target_mask: new ort.Tensor('bool', targetMask, [batchSize, decSeqLength])
        };
        
        console.log('\nRunning inference...');
        const outputs = await session.run(inputs);
        
        console.log('SUCCESS!');
        console.log('Output shape:', outputs.logits.dims);
        
    } catch (error) {
        console.error('ERROR:', error);
        console.error('Message:', error.message);
        console.error('Stack:', error.stack);
    }
}

testDecoder();