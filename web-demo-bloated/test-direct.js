import * as ort from 'onnxruntime-node';

async function testModel() {
    try {
        console.log('Loading encoder model...');
        const session = await ort.InferenceSession.create('../deployment_package/swipe_model_character.onnx');
        
        console.log('Model loaded!');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
        
        // Create test inputs
        const batchSize = 1;
        const sequenceLength = 50;  // Model expects fixed length of 50
        
        // trajectory_features: [batch, sequence, 6]
        const features = new Float32Array(batchSize * sequenceLength * 6);
        for (let i = 0; i < features.length; i++) {
            features[i] = Math.random();
        }
        
        // nearest_keys: [batch, sequence]
        const nearestKeys = new BigInt64Array(batchSize * sequenceLength);
        for (let i = 0; i < nearestKeys.length; i++) {
            nearestKeys[i] = BigInt(Math.floor(Math.random() * 26) + 4);
        }
        
        // src_mask: [batch, sequence] - as Uint8Array for bool
        const srcMask = new Uint8Array(batchSize * sequenceLength);
        
        const inputs = {
            trajectory_features: new ort.Tensor('float32', features, [batchSize, sequenceLength, 6]),
            nearest_keys: new ort.Tensor('int64', nearestKeys, [batchSize, sequenceLength]),
            src_mask: new ort.Tensor('bool', srcMask, [batchSize, sequenceLength])
        };
        
        console.log('\nRunning inference...');
        const outputs = await session.run(inputs);
        
        console.log('SUCCESS!');
        console.log('Output shape:', outputs.encoder_output.dims);
        
    } catch (error) {
        console.error('ERROR:', error);
        console.error('Message:', error.message);
        console.error('Stack:', error.stack);
    }
}

testModel();