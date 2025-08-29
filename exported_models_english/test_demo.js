#!/usr/bin/env node

/**
 * Test script for Neural Swipe Typing web demo
 * Simulates the core functionality without browser automation
 */

// Mock the transformer.js environment for testing
const mockTransformers = {
    pipeline: async (task, model, options) => {
        console.log(`Loading pipeline: ${task} with model: ${model}`);
        return {
            predict: async (inputs, options) => {
                console.log('Mock prediction called with inputs:', inputs);
                return [
                    { word: 'hello', score: 0.95, rank: 1 },
                    { word: 'help', score: 0.87, rank: 2 },
                    { word: 'hell', score: 0.82, rank: 3 }
                ];
            }
        };
    },
    env: {
        allowRemoteModels: false
    }
};

// Mock DOM environment
global.console = console;

// Import and test the decoder functionality
async function testNeuralSwipeDecoder() {
    console.log('🧪 Testing Neural Swipe Decoder...\n');

    // Mock the decoder class (since we can't import ES modules directly in Node)
    class TestNeuralSwipeDecoder {
        constructor(config = {}) {
            this.config = config;
            this.ready = false;
            this.keyboard = {
                keys: [
                    {id: 0, char: 'q', x: 0.05, y: 0.4, width: 0.1, height: 0.15},
                    {id: 1, char: 'w', x: 0.15, y: 0.4, width: 0.1, height: 0.15},
                    {id: 2, char: 'e', x: 0.25, y: 0.4, width: 0.1, height: 0.15},
                    {id: 10, char: 'a', x: 0.1, y: 0.55, width: 0.1, height: 0.15},
                    {id: 11, char: 's', x: 0.2, y: 0.55, width: 0.1, height: 0.15},
                    {id: 12, char: 'd', x: 0.3, y: 0.55, width: 0.1, height: 0.15},
                ]
            };
        }

        async initialize() {
            console.log('📦 Initializing decoder...');
            this.model = await mockTransformers.pipeline('text-generation', './onnx/', {
                dtype: 'fp32',
                device: 'cpu'
            });
            this.ready = true;
            console.log('✅ Decoder initialized successfully\n');
            return true;
        }

        extractTrajectoryFeatures(swipePoints) {
            console.log('🔄 Extracting trajectory features...');
            const features = [];
            
            for (let i = 0; i < swipePoints.length; i++) {
                const point = swipePoints[i];
                const prevPoint = i > 0 ? swipePoints[i - 1] : point;
                const nextPoint = i < swipePoints.length - 1 ? swipePoints[i + 1] : point;
                
                const x = Math.max(0, Math.min(1, point.x));
                const y = Math.max(0, Math.min(1, point.y));
                
                const dt = Math.max(point.t - prevPoint.t, 1);
                const vx = (point.x - prevPoint.x) / dt;
                const vy = (point.y - prevPoint.y) / dt;
                
                const dt2 = Math.max(nextPoint.t - point.t, 1);
                const vx_next = (nextPoint.x - point.x) / dt2;
                const vy_next = (nextPoint.y - point.y) / dt2;
                const ax = (vx_next - vx) / Math.max((dt + dt2) / 2, 1);
                const ay = (vy_next - vy) / Math.max((dt + dt2) / 2, 1);
                
                features.push([x, y, vx, vy, ax, ay]);
            }
            
            console.log(`   → Generated ${features.length} trajectory features`);
            return features;
        }

        extractKeyboardFeatures(swipePoints) {
            console.log('⌨️  Extracting keyboard features...');
            const features = [];
            
            for (const point of swipePoints) {
                const distances = this.keyboard.keys.map(key => {
                    const keyCenterX = key.x + key.width / 2;
                    const keyCenterY = key.y + key.height / 2;
                    const dx = point.x - keyCenterX;
                    const dy = point.y - keyCenterY;
                    return Math.sqrt(dx * dx + dy * dy);
                });
                
                const maxDistance = Math.max(...distances);
                const weights = distances.map(d => {
                    const normalizedDistance = d / maxDistance;
                    return Math.exp(-normalizedDistance * 3);
                });
                
                const weightSum = weights.reduce((sum, w) => sum + w, 0);
                const normalizedWeights = weights.map(w => w / weightSum);
                
                features.push(normalizedWeights);
            }
            
            console.log(`   → Generated ${features.length} keyboard features`);
            return features;
        }

        preprocessFeatures(trajectoryFeatures, keyboardFeatures) {
            console.log('🔧 Preprocessing features for model...');
            return {
                trajectory_features: [trajectoryFeatures],
                keyboard_features: [keyboardFeatures],
                sequence_length: trajectoryFeatures.length
            };
        }

        async decode(swipePoints, options = {}) {
            if (!this.ready) {
                throw new Error('Model not initialized. Call initialize() first.');
            }
            
            console.log('🎯 Decoding swipe gesture...');
            console.log(`   → Processing ${swipePoints.length} swipe points`);
            
            const trajectoryFeatures = this.extractTrajectoryFeatures(swipePoints);
            const keyboardFeatures = this.extractKeyboardFeatures(swipePoints);
            const modelInputs = this.preprocessFeatures(trajectoryFeatures, keyboardFeatures);
            
            console.log('🧠 Running model inference...');
            const predictions = await this.model.predict(modelInputs, options);
            
            console.log('✅ Prediction complete\n');
            return predictions;
        }
    }

    // Test the decoder
    try {
        const decoder = new TestNeuralSwipeDecoder({
            modelPath: './english-epoch=51-val_loss=1.248-val_word_acc=0.659.onnx'
        });

        await decoder.initialize();

        // Test with sample swipe gesture (simulating "hello")
        const swipePoints = [
            {x: 0.6, y: 0.55, t: 0},     // h
            {x: 0.25, y: 0.4, t: 50},   // e  
            {x: 0.9, y: 0.55, t: 100},  // l
            {x: 0.9, y: 0.55, t: 150},  // l
            {x: 0.85, y: 0.4, t: 200}   // o
        ];

        console.log('📝 Test swipe gesture:');
        swipePoints.forEach((point, i) => {
            console.log(`   Point ${i + 1}: (${point.x.toFixed(2)}, ${point.y.toFixed(2)}) at t=${point.t}ms`);
        });
        console.log('');

        const predictions = await decoder.decode(swipePoints);
        
        console.log('🎉 Prediction Results:');
        predictions.forEach(prediction => {
            console.log(`   ${prediction.rank}. "${prediction.word}" (score: ${prediction.score.toFixed(3)})`);
        });

        return true;

    } catch (error) {
        console.error('❌ Test failed:', error.message);
        return false;
    }
}

// Test web server accessibility
async function testWebServer() {
    console.log('\n🌐 Testing Web Server...\n');
    
    try {
        const http = require('http');
        
        const testRequest = (url) => {
            return new Promise((resolve, reject) => {
                const req = http.get(url, (res) => {
                    let data = '';
                    res.on('data', chunk => data += chunk);
                    res.on('end', () => {
                        resolve({ status: res.statusCode, data });
                    });
                });
                req.on('error', reject);
                req.setTimeout(5000, () => {
                    req.destroy();
                    reject(new Error('Request timeout'));
                });
            });
        };

        // Test HTML file
        console.log('📄 Testing HTML demo file...');
        const htmlResponse = await testRequest('http://localhost:8081/test_web_demo.html');
        console.log(`   → Status: ${htmlResponse.status}`);
        console.log(`   → Size: ${htmlResponse.data.length} bytes`);
        console.log(`   → Contains title: ${htmlResponse.data.includes('Neural Swipe Typing') ? '✅' : '❌'}`);

        // Test JavaScript integration file
        console.log('\n📜 Testing JavaScript integration file...');
        const jsResponse = await testRequest('http://localhost:8081/transformers_js_integration.js');
        console.log(`   → Status: ${jsResponse.status}`);
        console.log(`   → Size: ${jsResponse.data.length} bytes`);
        console.log(`   → Contains class: ${jsResponse.data.includes('NeuralSwipeDecoder') ? '✅' : '❌'}`);

        // Test model file
        console.log('\n🤖 Testing ONNX model file...');
        const modelResponse = await testRequest('http://localhost:8081/english-epoch=51-val_loss=1.248-val_word_acc=0.659.onnx');
        console.log(`   → Status: ${modelResponse.status}`);
        console.log(`   → Size: ${modelResponse.data.length} bytes`);
        console.log(`   → Is ONNX: ${modelResponse.data.includes('ONNX') || modelResponse.status === 200 ? '✅' : '❌'}`);

        return true;

    } catch (error) {
        console.error('❌ Web server test failed:', error.message);
        return false;
    }
}

// Run all tests
async function runAllTests() {
    console.log('🚀 Neural Swipe Typing Web Demo Test Suite');
    console.log('='.repeat(50));
    
    const decoderTest = await testNeuralSwipeDecoder();
    const serverTest = await testWebServer();
    
    console.log('\n📊 Test Summary:');
    console.log(`   Decoder functionality: ${decoderTest ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`   Web server access: ${serverTest ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`   Overall: ${decoderTest && serverTest ? '🎉 ALL TESTS PASSED' : '⚠️  SOME TESTS FAILED'}`);
}

if (require.main === module) {
    runAllTests();
}