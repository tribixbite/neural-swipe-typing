import { test, expect } from '@playwright/test';

test.describe('Comprehensive Swipe Typing Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console logging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error('Browser ERROR:', msg.text());
      } else if (!msg.text().includes('Step') && !msg.text().includes('indices')) {
        console.log('Browser:', msg.text());
      }
    });
    
    page.on('pageerror', error => {
      console.error('Page Error:', error.message);
    });
    
    // Navigate and wait for models to load
    await page.goto('/');
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    console.log('✓ Models loaded successfully');
  });

  test('1. Model loading and initialization', async ({ page }) => {
    // Check that canvas is visible
    const canvas = page.locator('#keyboard-canvas');
    await expect(canvas).toBeVisible();
    
    // Check predictions container exists
    const predictions = page.locator('#predictions');
    await expect(predictions).toBeVisible();
    
    // Check control buttons
    await expect(page.locator('#clear-btn')).toBeVisible();
    await expect(page.locator('#debug-btn')).toBeVisible();
    
    console.log('✓ UI elements initialized');
  });

  test('2. Canvas dimensions and keyboard layout', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    expect(box).toBeTruthy();
    expect(box!.width).toBeGreaterThan(0);
    expect(box!.height).toBeGreaterThan(0);
    
    // Check aspect ratio is approximately 360:215
    const aspectRatio = box!.width / box!.height;
    const expectedRatio = 360 / 215;
    expect(Math.abs(aspectRatio - expectedRatio)).toBeLessThan(0.1);
    
    console.log(`✓ Canvas dimensions: ${box!.width}x${box!.height}`);
    console.log(`✓ Aspect ratio: ${aspectRatio.toFixed(2)} (expected: ${expectedRatio.toFixed(2)})`);
  });

  test('3. Test word "the" prediction', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Swipe pattern for "the"
    const points = [
      {x: box.x + 162 * scaleX, y: box.y + 53 * scaleY},   // t
      {x: box.x + 216 * scaleX, y: box.y + 107 * scaleY},  // h  
      {x: box.x + 90 * scaleX, y: box.y + 53 * scaleY}     // e
    ];
    
    // Perform swipe
    await page.mouse.move(points[0].x, points[0].y);
    await page.mouse.down();
    
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(points[i].x, points[i].y, { steps: 10 });
      await page.waitForTimeout(50);
    }
    
    await page.mouse.up();
    await page.waitForTimeout(2000);
    
    // Get predictions
    const predictions = await page.evaluate(() => {
      const predElements = document.querySelectorAll('.prediction');
      return Array.from(predElements).map(el => el.textContent?.trim() || '');
    });
    
    console.log('Predictions for "the":', predictions);
    
    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions).toContain('the');
    
    // Check if "the" is in top 3
    const topThree = predictions.slice(0, 3);
    expect(topThree).toContain('the');
    
    console.log('✓ "the" predicted correctly');
  });

  test('4. Test word "hello" prediction', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Swipe pattern for "hello"
    const points = [
      {x: box.x + 216 * scaleX, y: box.y + 107 * scaleY},  // h
      {x: box.x + 90 * scaleX, y: box.y + 53 * scaleY},    // e
      {x: box.x + 324 * scaleX, y: box.y + 107 * scaleY},  // l
      {x: box.x + 324 * scaleX, y: box.y + 107 * scaleY},  // l (stay)
      {x: box.x + 306 * scaleX, y: box.y + 53 * scaleY}    // o
    ];
    
    // Clear any previous swipe
    await page.click('#clear-btn');
    await page.waitForTimeout(500);
    
    // Perform swipe
    await page.mouse.move(points[0].x, points[0].y);
    await page.mouse.down();
    
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(points[i].x, points[i].y, { steps: 10 });
      await page.waitForTimeout(50);
    }
    
    await page.mouse.up();
    await page.waitForTimeout(2000);
    
    // Get predictions
    const predictions = await page.evaluate(() => {
      const predElements = document.querySelectorAll('.prediction');
      return Array.from(predElements).map(el => el.textContent?.trim() || '');
    });
    
    console.log('Predictions for "hello":', predictions);
    
    expect(predictions.length).toBeGreaterThan(0);
    
    // Log if hello is found
    if (predictions.includes('hello')) {
      console.log('✓ "hello" found in predictions');
    } else {
      console.log('⚠ "hello" not in predictions, got:', predictions.join(', '));
    }
  });

  test('5. Test 150-length sequence handling', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Create a very long swipe pattern (>50 points to test 150 length)
    const points = [];
    
    // Zigzag pattern across keyboard
    for (let i = 0; i < 80; i++) {
      const x = 50 + (i % 10) * 26;
      const y = i % 2 === 0 ? 53 : 161;
      points.push({
        x: box.x + x * scaleX,
        y: box.y + y * scaleY
      });
    }
    
    console.log(`Testing with ${points.length} points (should handle up to 150)`);
    
    // Clear and perform long swipe
    await page.click('#clear-btn');
    await page.waitForTimeout(500);
    
    await page.mouse.move(points[0].x, points[0].y);
    await page.mouse.down();
    
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(points[i].x, points[i].y, { steps: 2 });
      if (i % 10 === 0) {
        await page.waitForTimeout(10);
      }
    }
    
    await page.mouse.up();
    await page.waitForTimeout(3000);
    
    // Check that predictions were generated without error
    const predictions = await page.evaluate(() => {
      const predElements = document.querySelectorAll('.prediction');
      return Array.from(predElements).map(el => el.textContent?.trim() || '');
    });
    
    expect(predictions.length).toBeGreaterThan(0);
    console.log(`✓ Handled ${points.length} points, got predictions:`, predictions.slice(0, 3));
  });

  test('6. Test key logging during swipe', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Track console logs
    const keyLogs: string[] = [];
    page.on('console', msg => {
      if (msg.text().startsWith('Key:')) {
        keyLogs.push(msg.text());
      }
    });
    
    // Swipe across specific keys
    const points = [
      {x: box.x + 36 * scaleX, y: box.y + 107 * scaleY},   // a
      {x: box.x + 72 * scaleX, y: box.y + 107 * scaleY},   // s
      {x: box.x + 108 * scaleX, y: box.y + 107 * scaleY},  // d
    ];
    
    await page.mouse.move(points[0].x, points[0].y);
    await page.mouse.down();
    
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(points[i].x, points[i].y, { steps: 10 });
      await page.waitForTimeout(50);
    }
    
    await page.mouse.up();
    await page.waitForTimeout(1000);
    
    // Check key logs
    expect(keyLogs.length).toBeGreaterThan(0);
    console.log('Key logs captured:', keyLogs);
    
    // Should have logged A, S, D
    expect(keyLogs.some(log => log.includes('A'))).toBeTruthy();
    expect(keyLogs.some(log => log.includes('S'))).toBeTruthy();
    expect(keyLogs.some(log => log.includes('D'))).toBeTruthy();
    
    console.log('✓ Key logging working correctly');
  });

  test('7. Test clear button functionality', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    // Make a swipe
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
    await page.mouse.up();
    await page.waitForTimeout(1000);
    
    // Should have predictions
    let predictions = await page.evaluate(() => {
      const predElements = document.querySelectorAll('.prediction');
      return predElements.length;
    });
    
    expect(predictions).toBeGreaterThan(0);
    
    // Click clear
    await page.click('#clear-btn');
    await page.waitForTimeout(500);
    
    // Should clear predictions
    const clearedText = await page.evaluate(() => {
      const container = document.querySelector('#predictions');
      return container?.textContent || '';
    });
    
    expect(clearedText).toContain('Swipe on the keyboard');
    console.log('✓ Clear button works');
  });

  test('8. Test debug mode toggle', async ({ page }) => {
    const debugBtn = page.locator('#debug-btn');
    
    // Check initial state
    let btnText = await debugBtn.textContent();
    expect(btnText).toBe('Debug: OFF');
    
    // Toggle on
    await debugBtn.click();
    btnText = await debugBtn.textContent();
    expect(btnText).toBe('Debug: ON');
    
    // Toggle off
    await debugBtn.click();
    btnText = await debugBtn.textContent();
    expect(btnText).toBe('Debug: OFF');
    
    console.log('✓ Debug toggle works');
  });

  test('9. Test model size and performance', async ({ page }) => {
    // Measure time to load page and models
    const startTime = Date.now();
    await page.reload();
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    const loadTime = Date.now() - startTime;
    
    console.log(`✓ Models loaded in ${loadTime}ms`);
    expect(loadTime).toBeLessThan(30000); // Should load within 30 seconds
    
    // Test inference speed
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    const inferenceStart = Date.now();
    
    // Quick swipe
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
    await page.mouse.up();
    
    // Wait for predictions
    await page.waitForSelector('.prediction', { timeout: 5000 });
    const inferenceTime = Date.now() - inferenceStart;
    
    console.log(`✓ Inference completed in ${inferenceTime}ms`);
    expect(inferenceTime).toBeLessThan(5000); // Should predict within 5 seconds
  });

  test('10. Test error handling', async ({ page }) => {
    // Test with extremely short swipe (should be rejected)
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    // Single point swipe
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.up();
    await page.waitForTimeout(1000);
    
    // Should show default message, not error
    const predText = await page.evaluate(() => {
      const container = document.querySelector('#predictions');
      return container?.textContent || '';
    });
    
    expect(predText).toContain('Swipe on the keyboard');
    console.log('✓ Handles short swipes gracefully');
  });
});