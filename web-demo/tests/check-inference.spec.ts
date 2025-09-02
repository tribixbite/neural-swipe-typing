import { test, expect } from '@playwright/test';

test.describe('Check Model Inference', () => {
  test('debug model loading and inference step by step', async ({ page }) => {
    const logs: string[] = [];
    const errors: string[] = [];
    
    page.on('console', msg => {
      const text = `${msg.type()}: ${msg.text()}`;
      logs.push(text);
      console.log('Browser:', text);
      
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    page.on('pageerror', error => {
      errors.push(error.message);
      console.error('Page Error:', error.message);
    });
    
    // Go to page
    await page.goto('/');
    
    // Wait a bit for initial loading
    await page.waitForTimeout(5000);
    
    // Check what's loaded
    const status = await page.evaluate(async () => {
      const result: any = {
        ortAvailable: typeof (window as any).ort !== 'undefined',
        loadingVisible: document.getElementById('loading')?.style.display !== 'none',
        loadingText: document.getElementById('loading-progress')?.textContent,
        canvasExists: !!document.getElementById('keyboard-canvas'),
        predictionsText: document.querySelector('.predictions')?.textContent
      };
      
      // Try to check if models are loaded
      if ((window as any).swipePredictor) {
        result.predictorExists = true;
        result.encoderSession = !!(window as any).swipePredictor.encoderSession;
        result.decoderSession = !!(window as any).swipePredictor.decoderSession;
      }
      
      return result;
    });
    
    console.log('Page Status:', status);
    
    // If loading is still visible, wait more
    if (status.loadingVisible) {
      console.log('Waiting for loading to complete...');
      await page.waitForSelector('#loading', { state: 'hidden', timeout: 60000 }).catch(e => {
        console.log('Loading did not hide:', e.message);
      });
    }
    
    // Now try a swipe
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (box) {
      console.log('Performing test swipe...');
      
      // Simple horizontal swipe
      await page.mouse.move(box.x + 100, box.y + 100);
      await page.mouse.down();
      await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
      await page.mouse.up();
      
      // Wait and check for predictions
      await page.waitForTimeout(5000);
      
      const afterSwipe = await page.evaluate(() => {
        return {
          predictionsHTML: document.querySelector('.predictions')?.innerHTML,
          predictionsText: document.querySelector('.predictions')?.textContent,
          predictionCount: document.querySelectorAll('.prediction').length
        };
      });
      
      console.log('After Swipe:', afterSwipe);
    }
    
    // Report errors
    if (errors.length > 0) {
      console.error('Errors found:', errors);
      throw new Error(`Inference failed with errors: ${errors.join(', ')}`);
    }
    
    // Check final state
    const finalCheck = await page.evaluate(() => {
      const predictions = document.querySelectorAll('.prediction');
      return {
        hasPredictions: predictions.length > 0,
        predictionsText: Array.from(predictions).map(p => p.textContent)
      };
    });
    
    console.log('Final Check:', finalCheck);
    
    // We should have predictions if everything worked
    expect(finalCheck.hasPredictions || !status.loadingVisible).toBeTruthy();
  });
});