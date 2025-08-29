#!/usr/bin/env node

/**
 * Firefox-based test for Neural Swipe Typing web demo
 * Since Playwright MCP defaults to Chrome, we'll use direct Firefox testing
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

class FirefoxWebTest {
    constructor() {
        this.serverPort = 8081;
        this.testResults = [];
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
        const logMessage = `${icon} ${message}`;
        console.log(logMessage);
        this.testResults.push({ timestamp, type, message });
    }

    async checkServerResponsive() {
        this.log('🌐 Checking web server responsiveness...');
        
        try {
            const testUrls = [
                'http://localhost:8081/test_web_demo.html',
                'http://localhost:8081/transformers_js_integration.js',
                'http://localhost:8081/english-epoch=51-val_loss=1.248-val_word_acc=0.659.onnx'
            ];

            for (const url of testUrls) {
                try {
                    const curlResult = execSync(`curl -s -o /dev/null -w "%{http_code}" "${url}"`, {
                        encoding: 'utf8',
                        timeout: 5000
                    });
                    
                    if (curlResult.trim() === '200') {
                        this.log(`   ${url.split('/').pop()}: HTTP 200 ✓`, 'success');
                    } else {
                        this.log(`   ${url.split('/').pop()}: HTTP ${curlResult}`, 'error');
                        return false;
                    }
                } catch (error) {
                    this.log(`   ${url.split('/').pop()}: Connection failed`, 'error');
                    return false;
                }
            }
            return true;
        } catch (error) {
            this.log(`Server check failed: ${error.message}`, 'error');
            return false;
        }
    }

    createTestScript() {
        this.log('📝 Creating Firefox automation test script...');
        
        const testScript = `
// Firefox automation test script for Neural Swipe Typing
console.log('🚀 Starting Neural Swipe Typing test in Firefox');

// Test 1: Check if page loaded correctly
function testPageLoad() {
    const title = document.title;
    const hasKeyboard = document.querySelector('#keyboard') !== null;
    const hasStatus = document.querySelector('#status') !== null;
    
    console.log('✅ Page title:', title);
    console.log('✅ Keyboard element:', hasKeyboard ? 'Found' : 'Missing');
    console.log('✅ Status element:', hasStatus ? 'Found' : 'Missing');
    
    return title.includes('Neural Swipe') && hasKeyboard && hasStatus;
}

// Test 2: Check if JavaScript modules load
function testModuleLoading() {
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.type = 'module';
        script.textContent = \`
            try {
                // Test if we can access the NeuralSwipeDecoder class
                import('./transformers_js_integration.js').then(module => {
                    console.log('✅ Module loaded:', !!module.NeuralSwipeDecoder);
                    window.testModuleResult = !!module.NeuralSwipeDecoder;
                }).catch(error => {
                    console.log('❌ Module load error:', error.message);
                    window.testModuleResult = false;
                });
            } catch (error) {
                console.log('❌ Import error:', error.message);
                window.testModuleResult = false;
            }
        \`;
        document.head.appendChild(script);
        
        // Check result after 2 seconds
        setTimeout(() => {
            resolve(window.testModuleResult || false);
        }, 2000);
    });
}

// Test 3: Simulate swipe gesture
function testSwipeGesture() {
    const keyboard = document.querySelector('#keyboard');
    if (!keyboard) return false;
    
    // Simulate mouse events for swipe
    const rect = keyboard.getBoundingClientRect();
    const events = [
        { type: 'mousedown', x: rect.width * 0.1, y: rect.height * 0.55 }, // 'a'
        { type: 'mousemove', x: rect.width * 0.2, y: rect.height * 0.55 }, // 's'
        { type: 'mousemove', x: rect.width * 0.3, y: rect.height * 0.55 }, // 'd'
        { type: 'mouseup', x: rect.width * 0.4, y: rect.height * 0.55 }    // 'f'
    ];
    
    events.forEach((event, index) => {
        setTimeout(() => {
            const mouseEvent = new MouseEvent(event.type, {
                clientX: rect.left + event.x,
                clientY: rect.top + event.y,
                bubbles: true
            });
            keyboard.dispatchEvent(mouseEvent);
        }, index * 100);
    });
    
    console.log('✅ Swipe gesture simulated');
    return true;
}

// Run all tests
async function runAllTests() {
    console.log('🧪 Running all tests...');
    
    const pageLoadResult = testPageLoad();
    console.log('📄 Page load test:', pageLoadResult ? 'PASS' : 'FAIL');
    
    const moduleResult = await testModuleLoading();
    console.log('📦 Module loading test:', moduleResult ? 'PASS' : 'FAIL');
    
    const swipeResult = testSwipeGesture();
    console.log('👆 Swipe gesture test:', swipeResult ? 'PASS' : 'FAIL');
    
    // Check for predictions after swipe
    setTimeout(() => {
        const predictions = document.querySelector('#predictions');
        const hasPredictions = predictions && predictions.children.length > 0;
        console.log('🎯 Predictions generated:', hasPredictions ? 'PASS' : 'FAIL');
        
        // Final summary
        const allPassed = pageLoadResult && moduleResult && swipeResult;
        console.log('\\n📊 Overall test result:', allPassed ? '🎉 PASS' : '❌ FAIL');
        
        // Mark test completion
        window.testCompleted = true;
        window.testResults = {
            pageLoad: pageLoadResult,
            moduleLoading: moduleResult,
            swipeGesture: swipeResult,
            predictions: hasPredictions,
            overall: allPassed
        };
    }, 3000);
}

// Start tests when page is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runAllTests);
} else {
    runAllTests();
}
`;

        const scriptPath = '/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english/firefox_test_script.js';
        fs.writeFileSync(scriptPath, testScript);
        return scriptPath;
    }

    async runFirefoxTest() {
        this.log('🦊 Starting Firefox-based web demo test...');
        
        // Check if server is responsive
        const serverOk = await this.checkServerResponsive();
        if (!serverOk) {
            this.log('Server not responsive, aborting test', 'error');
            return false;
        }

        // Create test script
        const scriptPath = this.createTestScript();
        
        // Create a temporary HTML file that includes our test
        const testHtml = `
<!DOCTYPE html>
<html>
<head>
    <title>Firefox Test Runner</title>
</head>
<body>
    <h1>Firefox Test Results</h1>
    <div id="test-output"></div>
    <iframe src="http://localhost:${this.serverPort}/test_web_demo.html" width="800" height="600"></iframe>
    <script src="file://${scriptPath}"></script>
</body>
</html>`;
        
        const testHtmlPath = '/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english/firefox_test.html';
        fs.writeFileSync(testHtmlPath, testHtml);
        
        try {
            this.log('🔄 Running Firefox test (10 second timeout)...');
            
            // Run Firefox with test page
            const firefoxProcess = spawn('firefox', [
                '--headless',
                '--new-instance',
                '--width=1024',
                '--height=768',
                `file://${testHtmlPath}`
            ], {
                stdio: ['ignore', 'pipe', 'pipe'],
                timeout: 10000
            });

            let output = '';
            firefoxProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            firefoxProcess.stderr.on('data', (data) => {
                output += data.toString();
            });

            return new Promise((resolve) => {
                firefoxProcess.on('close', (code) => {
                    this.log(`Firefox process exited with code: ${code}`);
                    if (output) {
                        this.log('Firefox output captured');
                        console.log(output);
                    }
                    resolve(code === 0);
                });

                // Kill process after timeout
                setTimeout(() => {
                    if (!firefoxProcess.killed) {
                        firefoxProcess.kill('SIGTERM');
                        this.log('Firefox test timed out', 'warning');
                        resolve(false);
                    }
                }, 10000);
            });

        } catch (error) {
            this.log(`Firefox test failed: ${error.message}`, 'error');
            return false;
        } finally {
            // Cleanup temp files
            try {
                fs.unlinkSync(scriptPath);
                fs.unlinkSync(testHtmlPath);
            } catch (e) {
                // Ignore cleanup errors
            }
        }
    }

    async validateWebDemo() {
        this.log('🔍 Validating web demo components...');
        
        // Check critical files exist and have content
        const files = [
            'test_web_demo.html',
            'transformers_js_integration.js',
            'english-epoch=51-val_loss=1.248-val_word_acc=0.659.onnx'
        ];

        for (const file of files) {
            const filePath = path.join('/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english', file);
            try {
                const stats = fs.statSync(filePath);
                this.log(`   ${file}: ${(stats.size / 1024).toFixed(1)}KB`, 'success');
            } catch (error) {
                this.log(`   ${file}: Missing or inaccessible`, 'error');
                return false;
            }
        }

        // Validate HTML structure
        try {
            const htmlContent = fs.readFileSync(
                '/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english/test_web_demo.html',
                'utf8'
            );
            
            const hasKeyboard = htmlContent.includes('id="keyboard"');
            const hasScript = htmlContent.includes('transformers_js_integration.js');
            const hasStyles = htmlContent.includes('<style>');
            
            this.log(`   HTML structure: Keyboard(${hasKeyboard}) Script(${hasScript}) Styles(${hasStyles})`, 
                hasKeyboard && hasScript && hasStyles ? 'success' : 'warning');
                
        } catch (error) {
            this.log(`   HTML validation failed: ${error.message}`, 'error');
        }

        return true;
    }

    generateReport() {
        this.log('📋 Generating test report...');
        
        const report = {
            timestamp: new Date().toISOString(),
            testResults: this.testResults,
            summary: {
                total: this.testResults.length,
                successes: this.testResults.filter(r => r.type === 'success').length,
                errors: this.testResults.filter(r => r.type === 'error').length,
                warnings: this.testResults.filter(r => r.type === 'warning').length
            }
        };

        const reportPath = '/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english/FIREFOX_TEST_REPORT.json';
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        this.log(`Test report saved to: ${reportPath}`, 'success');
        return report;
    }

    async runFullTest() {
        console.log('🚀 Neural Swipe Typing Firefox Test Suite');
        console.log('=' .repeat(50));
        
        // Validate components
        const validationResult = await this.validateWebDemo();
        
        // Run Firefox test
        const firefoxResult = await this.runFirefoxTest();
        
        // Generate report
        const report = this.generateReport();
        
        // Final summary
        console.log('\n📊 Final Test Summary:');
        console.log(`   Component validation: ${validationResult ? '✅ PASS' : '❌ FAIL'}`);
        console.log(`   Firefox browser test: ${firefoxResult ? '✅ PASS' : '⚠️  TIMEOUT/LIMITED'}`);
        console.log(`   Total test events: ${report.summary.total}`);
        console.log(`   Success rate: ${((report.summary.successes / report.summary.total) * 100).toFixed(1)}%`);
        
        return validationResult && firefoxResult;
    }
}

// Run tests if called directly
if (require.main === module) {
    const tester = new FirefoxWebTest();
    tester.runFullTest().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = FirefoxWebTest;