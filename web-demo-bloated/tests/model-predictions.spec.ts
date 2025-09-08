import { test, expect } from '@playwright/test';

test.describe('ONNX Model Predictions', () => {
  test.setTimeout(60000); // Give model time to load
  
  test('should load ONNX models and make predictions', async ({ page }) => {
    // Navigate to the demo
    await page.goto('/');
    
    // Wait for models to actually load - check console for success message
    await page.waitForFunction(() => {
      const loading = document.getElementById('loading');
      return loading && loading.style.display === 'none';
    }, { timeout: 45000 });
    
    // Verify canvas is ready
    const canvas = page.locator('#keyboard-canvas');
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Calculate scale for keyboard coordinates (360x215 -> actual canvas size)
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Swipe pattern for the word "hello"
    // Based on QWERTY layout from keyboard.ts:
    // h: (216, 155), e: (90, 111), l: (324, 155), l: (324, 155), o: (306, 111)
    const swipePattern = [
      { x: 216, y: 155 }, // h
      { x: 90, y: 111 },  // e  
      { x: 324, y: 155 }, // l
      { x: 324, y: 155 }, // l (stay on same key)
      { x: 306, y: 111 }  // o
    ];
    
    // Perform the swipe gesture
    console.log('Performing swipe for "hello"...');
    await page.mouse.move(
      box.x + swipePattern[0].x * scaleX,
      box.y + swipePattern[0].y * scaleY
    );
    await page.mouse.down();
    
    for (let i = 1; i < swipePattern.length; i++) {
      await page.mouse.move(
        box.x + swipePattern[i].x * scaleX,
        box.y + swipePattern[i].y * scaleY,
        { steps: 10 }
      );
      await page.waitForTimeout(100); // Small delay between points
    }
    
    await page.mouse.up();
    
    // Wait for predictions to appear
    await page.waitForSelector('.prediction', { 
      timeout: 10000,
      state: 'visible' 
    });
    
    // Verify predictions were generated
    const predictions = await page.locator('.prediction').all();
    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions.length).toBeLessThanOrEqual(5);
    
    // Get all prediction texts
    const predictionTexts = await Promise.all(
      predictions.map(p => p.textContent())
    );
    
    console.log('Predictions received:', predictionTexts);
    
    // At least one prediction should be a valid word
    const hasValidWord = predictionTexts.some(text => 
      text && text.length > 0 && /^[a-z]+$/i.test(text)
    );
    expect(hasValidWord).toBeTruthy();
    
    // Check that predictions container doesn't show error or loading
    const predictionsContainer = page.locator('.predictions');
    await expect(predictionsContainer).not.toContainText('Processing...');
    await expect(predictionsContainer).not.toContainText('Error');
    await expect(predictionsContainer).not.toContainText('Swipe on the keyboard');
  });

  test('should make predictions for simple swipe', async ({ page }) => {
    await page.goto('/');
    
    // Wait for models to load
    await page.waitForFunction(() => {
      const loading = document.getElementById('loading');
      return loading && loading.style.display === 'none';
    }, { timeout: 45000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Simple horizontal swipe (might predict "we", "er", "re", etc.)
    await page.mouse.move(box.x + 50, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 200, box.y + 100, { steps: 20 });
    await page.mouse.up();
    
    // Wait for predictions
    await page.waitForSelector('.prediction', { timeout: 10000 });
    
    // Should have predictions
    const predictionCount = await page.locator('.prediction').count();
    expect(predictionCount).toBeGreaterThan(0);
  });

  test('should handle multiple swipes in succession', async ({ page }) => {
    await page.goto('/');
    
    // Wait for models
    await page.waitForFunction(() => {
      const loading = document.getElementById('loading');
      return loading && loading.style.display === 'none';
    }, { timeout: 45000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // First swipe
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 250, box.y + 100, { steps: 10 });
    await page.mouse.up();
    
    // Wait for first predictions
    await page.waitForSelector('.prediction', { timeout: 10000 });
    const firstPredictions = await page.locator('.prediction').allTextContents();
    expect(firstPredictions.length).toBeGreaterThan(0);
    
    // Clear and do second swipe
    await page.click('#clear-btn');
    await page.waitForTimeout(500);
    
    // Second swipe (different pattern)
    await page.mouse.move(box.x + 50, box.y + 50);
    await page.mouse.down();
    await page.mouse.move(box.x + 150, box.y + 150, { steps: 10 });
    await page.mouse.up();
    
    // Wait for second predictions
    await page.waitForSelector('.prediction', { timeout: 10000 });
    const secondPredictions = await page.locator('.prediction').allTextContents();
    expect(secondPredictions.length).toBeGreaterThan(0);
    
    // Predictions should be different
    expect(firstPredictions).not.toEqual(secondPredictions);
  });

  test('should verify ONNX model files are loaded', async ({ page }) => {
    const modelRequests: string[] = [];
    
    // Track network requests for ONNX models
    page.on('response', response => {
      const url = response.url();
      if (url.includes('.onnx') || url.includes('tokenizer_config.json')) {
        modelRequests.push(url);
        console.log(`Model loaded: ${url} - Status: ${response.status()}`);
      }
    });
    
    await page.goto('/');
    
    // Wait for models to load
    await page.waitForFunction(() => {
      const loading = document.getElementById('loading');
      return loading && loading.style.display === 'none';
    }, { timeout: 45000 });
    
    // Verify model files were requested
    expect(modelRequests.some(url => url.includes('swipe_model_character.onnx'))).toBeTruthy();
    expect(modelRequests.some(url => url.includes('swipe_decoder_character.onnx'))).toBeTruthy();
    expect(modelRequests.some(url => url.includes('tokenizer_config.json'))).toBeTruthy();
  });

  test('should show meaningful predictions for common words', async ({ page }) => {
    await page.goto('/');
    
    // Wait for models
    await page.waitForFunction(() => {
      const loading = document.getElementById('loading');
      return loading && loading.style.display === 'none';
    }, { timeout: 45000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Swipe for "the" - a very common word
    // t: (162, 111), h: (216, 155), e: (90, 111)
    const thePattern = [
      { x: 162, y: 111 }, // t
      { x: 216, y: 155 }, // h
      { x: 90, y: 111 }   // e
    ];
    
    // Perform swipe
    await page.mouse.move(
      box.x + thePattern[0].x * scaleX,
      box.y + thePattern[0].y * scaleY
    );
    await page.mouse.down();
    
    for (const point of thePattern.slice(1)) {
      await page.mouse.move(
        box.x + point.x * scaleX,
        box.y + point.y * scaleY,
        { steps: 10 }
      );
      await page.waitForTimeout(50);
    }
    
    await page.mouse.up();
    
    // Wait for predictions
    await page.waitForSelector('.prediction', { timeout: 10000 });
    
    const predictions = await page.locator('.prediction').allTextContents();
    console.log('Predictions for "the" swipe:', predictions);
    
    // Should get reasonable predictions
    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions.length).toBeLessThanOrEqual(5);
    
    // All predictions should be valid words (only letters)
    predictions.forEach(pred => {
      expect(pred).toMatch(/^[a-z]+$/i);
    });
  });
});