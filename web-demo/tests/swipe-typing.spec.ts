import { test, expect, Page } from '@playwright/test';

test.describe('Swipe Typing Demo', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for models to load
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 30000 });
  });

  test('should load the page and display keyboard', async ({ page }) => {
    // Check main elements are visible
    await expect(page.locator('h1')).toHaveText('Neural Swipe Typing');
    await expect(page.locator('.subtitle')).toContainText('70.1% Accuracy');
    await expect(page.locator('#keyboard-canvas')).toBeVisible();
    await expect(page.locator('.predictions')).toBeVisible();
  });

  test('should render QWERTY keyboard layout', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    await expect(canvas).toBeVisible();
    
    // Take screenshot to verify keyboard rendering
    await expect(canvas).toHaveScreenshot('keyboard-layout.png');
  });

  test('should show initial prediction placeholder', async ({ page }) => {
    const predictions = page.locator('.predictions');
    await expect(predictions).toContainText('Swipe on the keyboard to see predictions');
  });

  test('should track mouse swipe gestures', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Simulate swipe for "hello"
    // h: (216, 167), e: (90, 111), l: (324, 167), l: (324, 167), o: (306, 111)
    const points = [
      { x: 216, y: 167 }, // h
      { x: 90, y: 111 },  // e
      { x: 324, y: 167 }, // l
      { x: 324, y: 167 }, // l
      { x: 306, y: 111 }  // o
    ];

    // Scale points to canvas size
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;

    // Start swipe
    await page.mouse.move(
      box.x + points[0].x * scaleX,
      box.y + points[0].y * scaleY
    );
    await page.mouse.down();

    // Move through points
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(
        box.x + points[i].x * scaleX,
        box.y + points[i].y * scaleY,
        { steps: 5 }
      );
      await page.waitForTimeout(50);
    }

    // End swipe
    await page.mouse.up();

    // Wait for predictions to appear
    await page.waitForSelector('.prediction', { timeout: 5000 });
    
    // Check that predictions are shown
    const predictions = await page.locator('.prediction').count();
    expect(predictions).toBeGreaterThan(0);
    expect(predictions).toBeLessThanOrEqual(5);
  });

  test('should handle touch gestures on mobile', async ({ page, isMobile }) => {
    if (!isMobile) {
      test.skip();
      return;
    }

    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Simulate touch swipe
    const scaleX = box.width / 360;
    const scaleY = box.height / 215;
    
    // Swipe for "the"
    await page.touchscreen.tap(
      box.x + 162 * scaleX, // t
      box.y + 111 * scaleY
    );

    await page.waitForTimeout(100);

    // Check canvas was touched
    await expect(page.locator('.predictions')).not.toContainText('Swipe on the keyboard');
  });

  test('should clear trace and predictions', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Make a swipe
    await page.mouse.move(box.x + 50, box.y + 50);
    await page.mouse.down();
    await page.mouse.move(box.x + 150, box.y + 150, { steps: 5 });
    await page.mouse.up();

    // Wait for predictions
    await page.waitForSelector('.prediction', { timeout: 5000 });

    // Click clear button
    await page.click('#clear-btn');

    // Check predictions are cleared
    await expect(page.locator('.predictions')).toContainText('Swipe on the keyboard to see predictions');
  });

  test('should toggle debug mode', async ({ page }) => {
    const debugBtn = page.locator('#debug-btn');
    
    // Initial state
    await expect(debugBtn).toHaveText('Debug Mode');
    
    // Toggle on
    await debugBtn.click();
    await expect(debugBtn).toHaveText('Debug: ON');
    
    // Toggle off
    await debugBtn.click();
    await expect(debugBtn).toHaveText('Debug: OFF');
  });

  test('should be responsive on different screen sizes', async ({ page }) => {
    // Test desktop size
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('.container')).toBeVisible();
    await expect(page.locator('#keyboard-canvas')).toBeVisible();

    // Test tablet size
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('.container')).toBeVisible();
    await expect(page.locator('#keyboard-canvas')).toBeVisible();

    // Test mobile size
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('.container')).toBeVisible();
    await expect(page.locator('#keyboard-canvas')).toBeVisible();
  });

  test('should maintain aspect ratio', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Check aspect ratio is approximately 360:215
    const expectedRatio = 360 / 215;
    const actualRatio = box.width / box.height;
    
    expect(Math.abs(actualRatio - expectedRatio)).toBeLessThan(0.01);
  });

  test('should handle rapid swipes', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Perform multiple rapid swipes
    for (let i = 0; i < 3; i++) {
      await page.mouse.move(box.x + 50, box.y + 50);
      await page.mouse.down();
      await page.mouse.move(box.x + 250, box.y + 150, { steps: 3 });
      await page.mouse.up();
      await page.waitForTimeout(100);
    }

    // Should not crash and should show predictions
    await expect(page.locator('.predictions')).toBeVisible();
  });

  test('should handle edge cases', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Test very short swipe (should be ignored)
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 102, box.y + 102);
    await page.mouse.up();

    // Should still show initial message
    await expect(page.locator('.predictions')).toContainText('Swipe on the keyboard to see predictions');

    // Test swipe outside keyboard area
    await page.mouse.move(box.x - 10, box.y - 10);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width + 10, box.y + box.height + 10, { steps: 5 });
    await page.mouse.up();

    // Should handle gracefully
    await expect(page).not.toHaveTitle(/Error/);
  });

  test('should select prediction on click', async ({ page }) => {
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Make a swipe
    await page.mouse.move(box.x + 50, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 250, box.y + 100, { steps: 5 });
    await page.mouse.up();

    // Wait for predictions
    await page.waitForSelector('.prediction', { timeout: 5000 });

    // Click first prediction
    const firstPrediction = page.locator('.prediction').first();
    const predictionText = await firstPrediction.textContent();
    
    // Listen for console log
    const consolePromise = page.waitForEvent('console', msg => 
      msg.text().includes('Selected word:')
    );
    
    await firstPrediction.click();
    
    const consoleMsg = await consolePromise;
    expect(consoleMsg.text()).toContain(predictionText);
  });
});

test.describe('Performance Tests', () => {
  test('should load page within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 30000 });
    const loadTime = Date.now() - startTime;
    
    // Page should load within 30 seconds (including model loading)
    expect(loadTime).toBeLessThan(30000);
  });

  test('should respond to swipes quickly', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');

    // Measure swipe response time
    const startTime = Date.now();
    
    await page.mouse.move(box.x + 50, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 250, box.y + 100, { steps: 5 });
    await page.mouse.up();
    
    await page.waitForSelector('.prediction', { timeout: 5000 });
    const responseTime = Date.now() - startTime;
    
    // Predictions should appear within 5 seconds
    expect(responseTime).toBeLessThan(5000);
  });
});

test.describe('Accessibility Tests', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/');
    
    // Check for keyboard accessibility
    const canvas = page.locator('#keyboard-canvas');
    // Canvas should have a role or aria-label
    const role = await canvas.getAttribute('role');
    const ariaLabel = await canvas.getAttribute('aria-label');
    expect(role || ariaLabel).toBeTruthy();
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 30000 });
    
    // Tab to clear button
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Clear button should be focused
    const clearBtn = page.locator('#clear-btn');
    await expect(clearBtn).toBeFocused();
    
    // Tab to debug button
    await page.keyboard.press('Tab');
    const debugBtn = page.locator('#debug-btn');
    await expect(debugBtn).toBeFocused();
    
    // Activate with Enter
    await page.keyboard.press('Enter');
    await expect(debugBtn).toHaveText('Debug: ON');
  });
});