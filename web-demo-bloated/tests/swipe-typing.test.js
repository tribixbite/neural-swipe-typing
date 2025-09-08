import { test, expect } from '@playwright/test';

test.describe('Neural Swipe Typing Demo', () => {
  test.beforeEach(async ({ page }) => {
    // Start local server or use GitHub Pages
    const baseURL = process.env.BASE_URL || 'http://localhost:8080';
    await page.goto(baseURL);
  });

  test('should load the application successfully', async ({ page }) => {
    // Check title
    await expect(page).toHaveTitle(/Neural Swipe Typing/);
    
    // Check main elements are present
    await expect(page.locator('#keyboard-canvas')).toBeVisible();
    await expect(page.locator('#predictions')).toBeVisible();
    await expect(page.locator('#clear-btn')).toBeVisible();
    await expect(page.locator('#debug-btn')).toBeVisible();
  });

  test('should initialize ONNX models', async ({ page }) => {
    // Wait for models to load (loading overlay should disappear)
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    // Check status shows ready
    const status = page.locator('#status');
    await expect(status).toContainText('Ready');
  });

  test('should handle swipe gestures on keyboard', async ({ page }) => {
    // Wait for app to be ready
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas bounding box not found');
    }

    // Simulate swipe for "hi" - h(216,107) to i(270,53)
    const points = [
      { x: 0.6, y: 0.5 },  // h position (relative)
      { x: 0.75, y: 0.25 }  // i position (relative)
    ];

    // Start swipe
    await page.mouse.move(
      boundingBox.x + boundingBox.width * points[0].x,
      boundingBox.y + boundingBox.height * points[0].y
    );
    await page.mouse.down();

    // Move through points
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(
        boundingBox.x + boundingBox.width * points[i].x,
        boundingBox.y + boundingBox.height * points[i].y,
        { steps: 5 }
      );
    }

    // End swipe
    await page.mouse.up();

    // Wait for predictions
    await page.waitForTimeout(1000);

    // Check predictions appeared
    const predictions = page.locator('#predictions button');
    await expect(predictions).toHaveCount(5, { timeout: 5000 });
  });

  test('should clear traces and predictions', async ({ page }) => {
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    // Do a swipe first
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (boundingBox) {
      await page.mouse.move(boundingBox.x + 100, boundingBox.y + 50);
      await page.mouse.down();
      await page.mouse.move(boundingBox.x + 200, boundingBox.y + 100, { steps: 5 });
      await page.mouse.up();
      
      // Wait for predictions
      await page.waitForTimeout(1000);
      
      // Click clear button
      await page.click('#clear-btn');
      
      // Check predictions are cleared
      const predictionsText = await page.locator('#predictions').textContent();
      expect(predictionsText).toContain('Swipe on the keyboard');
    }
  });

  test('should toggle debug mode', async ({ page }) => {
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    const debugBtn = page.locator('#debug-btn');
    
    // Initially debug is OFF
    await expect(debugBtn).toContainText('Debug: OFF');
    
    // Click to turn ON
    await debugBtn.click();
    await expect(debugBtn).toContainText('Debug: ON');
    
    // Click to turn OFF again
    await debugBtn.click();
    await expect(debugBtn).toContainText('Debug: OFF');
  });

  test('should handle mobile viewport correctly', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(process.env.BASE_URL || 'http://localhost:8080');
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas not found');
    }

    // Check aspect ratio is approximately 1.4:1 for mobile
    const aspectRatio = boundingBox.width / boundingBox.height;
    expect(aspectRatio).toBeCloseTo(1.4, 1);
  });

  test('should display swipe path characters', async ({ page }) => {
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas not found');
    }

    // Swipe through specific letters
    const swipePath = [
      { x: 0.1, y: 0.5 },   // a
      { x: 0.2, y: 0.5 },   // s  
      { x: 0.3, y: 0.5 }    // d
    ];

    await page.mouse.move(
      boundingBox.x + boundingBox.width * swipePath[0].x,
      boundingBox.y + boundingBox.height * swipePath[0].y
    );
    await page.mouse.down();

    for (let i = 1; i < swipePath.length; i++) {
      await page.mouse.move(
        boundingBox.x + boundingBox.width * swipePath[i].x,
        boundingBox.y + boundingBox.height * swipePath[i].y,
        { steps: 10 }
      );
      await page.waitForTimeout(100);
    }

    // Check swipe chars display updated
    const swipeChars = page.locator('#swipe-chars');
    const text = await swipeChars.textContent();
    expect(text).toMatch(/[A-Z]/); // Should contain capital letters
  });

  test('should handle prediction selection', async ({ page }) => {
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    // Perform a swipe
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas not found');
    }

    await page.mouse.move(boundingBox.x + 100, boundingBox.y + 50);
    await page.mouse.down();
    await page.mouse.move(boundingBox.x + 200, boundingBox.y + 100, { steps: 5 });
    await page.mouse.up();
    
    // Wait for predictions
    await page.waitForTimeout(1500);
    
    // Try to click first prediction
    const firstPrediction = page.locator('#predictions button').first();
    const predictionExists = await firstPrediction.count() > 0;
    
    if (predictionExists) {
      const word = await firstPrediction.textContent();
      await firstPrediction.click();
      
      // In a real app, this would insert the word somewhere
      console.log('Selected word:', word);
    }
  });
});

test.describe('Advanced Neural Swipe Typing', () => {
  test.beforeEach(async ({ page }) => {
    const baseURL = process.env.BASE_URL || 'http://localhost:8080';
    await page.goto(`${baseURL}/advanced.html`);
  });

  test('should load advanced page with enhanced UI', async ({ page }) => {
    await expect(page).toHaveTitle(/Advanced Neural Swipe Typing/);
    
    // Check enhanced elements
    await expect(page.locator('.header h1')).toContainText('Advanced Neural Swipe Typing');
    await expect(page.locator('#demo-btn')).toBeVisible();
    await expect(page.locator('#debug-btn')).toBeVisible();
    await expect(page.locator('.instructions')).toBeVisible();
  });

  test('should run demo animation', async ({ page }) => {
    // Wait for models to load
    await page.waitForSelector('#loading-overlay', { state: 'hidden', timeout: 30000 });
    
    // Wait for demo button to be enabled
    await page.waitForSelector('#demo-btn:not([disabled])', { timeout: 10000 });
    
    // Click demo button
    await page.click('#demo-btn');
    
    // Wait for demo to complete
    await page.waitForTimeout(3000);
    
    // Check predictions appeared
    const predictions = page.locator('.prediction-item');
    await expect(predictions).toHaveCount(5, { timeout: 5000 });
    
    // Check if "hello" is in predictions (should be top prediction)
    const firstPrediction = predictions.first();
    const text = await firstPrediction.textContent();
    expect(text.toLowerCase()).toContain('hello');
  });

  test('should show swipe path visualization', async ({ page }) => {
    await page.waitForSelector('#loading-overlay', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas not found');
    }

    // Perform swipe
    await page.mouse.move(boundingBox.x + 100, boundingBox.y + 100);
    await page.mouse.down();
    await page.mouse.move(boundingBox.x + 200, boundingBox.y + 100, { steps: 10 });
    await page.mouse.move(boundingBox.x + 300, boundingBox.y + 150, { steps: 10 });
    await page.mouse.up();
    
    // Check swipe info is displayed
    const swipeInfo = page.locator('#swipe-info');
    await expect(swipeInfo).toHaveClass(/active/);
    
    // Check swipe path shows letters
    const swipePath = page.locator('#swipe-path');
    const pathText = await swipePath.textContent();
    expect(pathText).toMatch(/[A-Z].*→.*[A-Z]/); // Should show letter arrows
  });

  test('should toggle debug mode and show logs', async ({ page }) => {
    await page.waitForSelector('#loading-overlay', { state: 'hidden', timeout: 30000 });
    
    // Enable debug mode
    await page.click('#debug-btn');
    
    // Check debug info is visible
    const debugInfo = page.locator('#debug-info');
    await expect(debugInfo).toHaveClass(/active/);
    
    // Perform a swipe to generate debug logs
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (boundingBox) {
      await page.mouse.move(boundingBox.x + 100, boundingBox.y + 100);
      await page.mouse.down();
      await page.mouse.move(boundingBox.x + 200, boundingBox.y + 150, { steps: 5 });
      await page.mouse.up();
      
      await page.waitForTimeout(1500);
      
      // Check debug logs contain relevant info
      const debugText = await debugInfo.textContent();
      expect(debugText).toContain('Inference time');
      expect(debugText).toContain('Swipe points');
    }
  });

  test('should show performance metrics', async ({ page }) => {
    await page.waitForSelector('#loading-overlay', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const boundingBox = await canvas.boundingBox();
    
    if (!boundingBox) {
      throw new Error('Canvas not found');
    }

    // Perform swipe
    await page.mouse.move(boundingBox.x + 100, boundingBox.y + 100);
    await page.mouse.down();
    await page.mouse.move(boundingBox.x + 200, boundingBox.y + 150, { steps: 10 });
    await page.mouse.up();
    
    await page.waitForTimeout(1500);
    
    // Check swipe details show metrics
    const swipeDetails = page.locator('#swipe-details');
    const detailsText = await swipeDetails.textContent();
    expect(detailsText).toContain('Points:');
    expect(detailsText).toContain('Inference:');
    expect(detailsText).toContain('ms');
  });
});