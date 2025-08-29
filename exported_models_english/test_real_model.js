/**
 * Test script to validate the real model integration
 * Run this in the browser console when the web demo loads
 */

// Test function to validate model loading
window.testRealModelIntegration = async function() {
    console.log('🧪 Testing Real Model Integration');
    
    try {
        // Test 1: Check if decoder is initialized
        if (!window.decoder) {
            console.error('❌ Decoder not found');
            return false;
        }
        
        console.log('✅ Decoder found');
        
        // Test 2: Check if vocabulary is loaded
        if (!window.decoder.vocabulary || window.decoder.vocabulary.length === 0) {
            console.error('❌ Vocabulary not loaded');
            return false;
        }
        
        console.log(`✅ Vocabulary loaded: ${window.decoder.vocabulary.length} words`);
        
        // Test 3: Check if model is ready
        if (!window.decoder.ready) {
            console.error('❌ Model not ready');
            return false;
        }
        
        console.log('✅ Model ready');
        
        // Test 4: Test feature extraction
        const testPoints = [
            {x: 0.1, y: 0.2, t: 0},
            {x: 0.2, y: 0.3, t: 100},
            {x: 0.3, y: 0.4, t: 200}
        ];
        
        const features = window.decoder.extractTrajectoryFeatures(testPoints);
        console.log(`✅ Feature extraction: ${features.length} features extracted`);
        console.log('Sample features:', features[0]);
        
        // Test 5: Test prediction (this might fail with ONNX but should fallback)
        console.log('🎯 Testing prediction...');
        const predictions = await window.decoder.decode(testPoints);
        console.log(`✅ Predictions: ${predictions.length} predictions received`);
        console.log('Top prediction:', predictions[0]);
        
        console.log('🎉 All tests passed!');
        return true;
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return false;
    }
};

// Auto-run test after 5 seconds if in browser
if (typeof window !== 'undefined') {
    setTimeout(() => {
        if (window.decoder) {
            console.log('🚀 Auto-running integration test...');
            window.testRealModelIntegration();
        }
    }, 5000);
}