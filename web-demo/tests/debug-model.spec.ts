import { test, expect } from '@playwright/test';

test.describe('Debug ONNX Model Loading', () => {
  test('capture console logs and errors during model loading', async ({ page }) => {
    const consoleLogs: string[] = [];
    const consoleErrors: string[] = [];
    const networkErrors: string[] = [];
    
    // Capture all console messages
    page.on('console', msg => {
      const text = msg.text();
      if (msg.type() === 'error') {
        consoleErrors.push(text);
        console.error('Browser Error:', text);
      } else {
        consoleLogs.push(text);
        console.log('Browser Log:', text);
      }
    });
    
    // Capture page errors
    page.on('pageerror', error => {
      consoleErrors.push(error.message);
      console.error('Page Error:', error.message);
    });
    
    // Monitor network failures
    page.on('requestfailed', request => {
      const failure = `${request.method()} ${request.url()}: ${request.failure()?.errorText}`;
      networkErrors.push(failure);
      console.error('Network Error:', failure);
    });
    
    // Monitor successful network requests for ONNX files
    page.on('response', response => {
      const url = response.url();
      if (url.includes('.onnx') || url.includes('tokenizer') || url.includes('/models/')) {
        console.log(`Network Request: ${url} - Status: ${response.status()}`);
      }
    });
    
    console.log('Navigating to page...');
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Give it some time to attempt loading
    console.log('Waiting for model loading attempts...');
    await page.waitForTimeout(10000);
    
    // Check if loading overlay is still visible
    const loadingVisible = await page.locator('#loading').isVisible();
    const loadingText = await page.locator('#loading-progress').textContent();
    
    console.log('Loading overlay visible:', loadingVisible);
    console.log('Loading text:', loadingText);
    
    // Try to get more information from the page
    const pageInfo = await page.evaluate(() => {
      const loading = document.getElementById('loading');
      const progress = document.getElementById('loading-progress');
      
      return {
        loadingDisplay: loading?.style.display,
        loadingClass: loading?.className,
        progressText: progress?.textContent,
        hasOnnxRuntime: typeof (window as any).ort !== 'undefined',
        documentReady: document.readyState
      };
    });
    
    console.log('Page Info:', pageInfo);
    
    // Report findings
    if (consoleErrors.length > 0) {
      console.error('\n=== Console Errors Found ===');
      consoleErrors.forEach(err => console.error(err));
    }
    
    if (networkErrors.length > 0) {
      console.error('\n=== Network Errors Found ===');
      networkErrors.forEach(err => console.error(err));
    }
    
    // Fail the test with diagnostic information
    if (loadingVisible) {
      throw new Error(`
        Model loading failed!
        Loading text: ${loadingText}
        Console errors: ${consoleErrors.length}
        Network errors: ${networkErrors.length}
        First error: ${consoleErrors[0] || networkErrors[0] || 'No errors captured'}
      `);
    }
  });
  
  test('check if ONNX runtime loads', async ({ page }) => {
    const logs: string[] = [];
    
    page.on('console', msg => {
      logs.push(`${msg.type()}: ${msg.text()}`);
    });
    
    await page.goto('/');
    
    // Check if ONNX runtime is available
    const hasOnnxRuntime = await page.evaluate(() => {
      return typeof (window as any).ort !== 'undefined';
    });
    
    console.log('ONNX Runtime available:', hasOnnxRuntime);
    console.log('Console logs:', logs);
    
    expect(hasOnnxRuntime).toBeTruthy();
  });
  
  test('manually test model loading', async ({ page }) => {
    await page.goto('/');
    
    // Try to manually load models via console
    const result = await page.evaluate(async () => {
      try {
        // Check if ort is available
        const ort = (window as any).ort;
        if (!ort) {
          return { error: 'ONNX Runtime not found on window' };
        }
        
        // Try to create a simple session
        const testSession = await ort.InferenceSession.create('/models/swipe_model_character.onnx');
        
        return { 
          success: true, 
          inputNames: testSession.inputNames,
          outputNames: testSession.outputNames 
        };
      } catch (error: any) {
        return { 
          error: error.message,
          stack: error.stack
        };
      }
    });
    
    console.log('Manual model load result:', result);
    
    if ((result as any).error) {
      throw new Error(`Failed to load model: ${(result as any).error}`);
    }
    
    expect((result as any).success).toBeTruthy();
  });
});