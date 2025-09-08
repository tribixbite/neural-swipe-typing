import { test, expect } from '@playwright/test';

test.describe('Swipe ONNX Keyboard', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the swipe-onnx.html page (served from public directory)
    await page.goto('http://localhost:3456/swipe-onnx.html');
    
    // Wait for models to load (loading overlay to disappear)
    await page.waitForSelector('#loadingOverlay', { state: 'hidden', timeout: 30000 });
  });

  test('should load models successfully', async ({ page }) => {
    // Check that status shows ready
    const statusText = await page.locator('#statusText').textContent();
    expect(statusText).toBe('Ready');
    
    // Check model status text
    const modelStatus = await page.locator('#modelStatus').textContent();
    expect(modelStatus).toContain('ONNX Models Ready');
  });

  test('should perform mouse drag swipe', async ({ page }) => {
    // Get canvas element
    const canvas = page.locator('#swipeCanvas');
    await expect(canvas).toBeVisible();
    
    // Get canvas bounding box
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Simulate swipe from H to I (hello pattern)
    const startX = box.x + box.width * 0.5;  // Middle horizontally
    const startY = box.y + box.height * 0.5; // Middle vertically
    
    // Start drag
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    
    // Move across keyboard (simulate swiping "hello")
    await page.mouse.move(startX + 50, startY, { steps: 5 });
    await page.mouse.move(startX + 100, startY - 20, { steps: 5 });
    await page.mouse.move(startX + 150, startY, { steps: 5 });
    await page.mouse.move(startX + 200, startY + 20, { steps: 5 });
    
    // End drag
    await page.mouse.up();
    
    // Wait for predictions to appear
    await page.waitForTimeout(500);
    
    // Check that suggestions div has content
    const suggestions = await page.locator('#suggestions').innerHTML();
    expect(suggestions).not.toContain('Swipe on the keyboard to begin');
    
    // Should have prediction buttons
    const predictionButtons = await page.locator('#suggestions > div').count();
    expect(predictionButtons).toBeGreaterThan(0);
  });

  test('should highlight keys during swipe', async ({ page }) => {
    const canvas = page.locator('#swipeCanvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Find a key element to swipe over
    const keyQ = page.locator('.key[data-key="q"]');
    const keyBox = await keyQ.boundingBox();
    if (!keyBox) throw new Error('Key not found');
    
    // Start swipe on the Q key
    await page.mouse.move(keyBox.x + keyBox.width / 2, keyBox.y + keyBox.height / 2);
    await page.mouse.down();
    
    // Check that key gets active class
    await expect(keyQ).toHaveClass(/key-active/);
    
    // Move to another key
    const keyW = page.locator('.key[data-key="w"]');
    const keyWBox = await keyW.boundingBox();
    if (!keyWBox) throw new Error('Key W not found');
    
    await page.mouse.move(keyWBox.x + keyWBox.width / 2, keyWBox.y + keyWBox.height / 2, { steps: 5 });
    
    // W should be active, Q should not
    await expect(keyW).toHaveClass(/key-active/);
    
    await page.mouse.up();
    
    // After release, no keys should be active
    await page.waitForTimeout(100);
    const activeKeys = await page.locator('.key-active').count();
    expect(activeKeys).toBe(0);
  });

  test('should handle key taps', async ({ page }) => {
    // Click on a letter key
    const keyH = page.locator('.key[data-key="h"]');
    await keyH.click();
    
    // Check that letter appears in input text
    const inputText = await page.locator('#inputText').textContent();
    expect(inputText).toContain('h');
    
    // Click another key
    const keyI = page.locator('.key[data-key="i"]');
    await keyI.click();
    
    const updatedText = await page.locator('#inputText').textContent();
    expect(updatedText).toBe('hi');
  });

  test('should clear input on clear button', async ({ page }) => {
    // Type something first
    await page.locator('.key[data-key="t"]').click();
    await page.locator('.key[data-key="e"]').click();
    await page.locator('.key[data-key="s"]').click();
    await page.locator('.key[data-key="t"]').click();
    
    let inputText = await page.locator('#inputText').textContent();
    expect(inputText).toBe('test');
    
    // Click clear button
    await page.getByRole('button', { name: 'Clear' }).click();
    
    // Canvas should be cleared (no trail visible)
    // Input text should remain (clear only clears the swipe trail)
    inputText = await page.locator('#inputText').textContent();
    expect(inputText).toBe('test'); // Text remains
  });

  test('should toggle debug mode', async ({ page }) => {
    // Check initial state
    const debugStatus = await page.locator('#debugStatus').textContent();
    expect(debugStatus).toBe('OFF');
    
    // Toggle debug
    await page.getByRole('button', { name: /Debug:/ }).click();
    
    const updatedStatus = await page.locator('#debugStatus').textContent();
    expect(updatedStatus).toBe('ON');
    
    // Debug overlay should be visible
    await expect(page.locator('#debugOverlay')).toBeVisible();
    
    // Toggle back off
    await page.getByRole('button', { name: /Debug:/ }).click();
    await expect(page.locator('#debugOverlay')).toBeHidden();
  });

  test('should handle backspace', async ({ page }) => {
    // Type some text
    await page.locator('.key[data-key="h"]').click();
    await page.locator('.key[data-key="e"]').click();
    await page.locator('.key[data-key="l"]').click();
    await page.locator('.key[data-key="l"]').click();
    await page.locator('.key[data-key="o"]').click();
    
    let inputText = await page.locator('#inputText').textContent();
    expect(inputText).toBe('hello');
    
    // Click backspace
    await page.locator('.key[data-key="backspace"]').click();
    
    inputText = await page.locator('#inputText').textContent();
    expect(inputText).toBe('hell');
  });

  test('should handle space', async ({ page }) => {
    // Type a word
    await page.locator('.key[data-key="h"]').click();
    await page.locator('.key[data-key="i"]').click();
    
    // Click space
    await page.getByRole('button', { name: 'space' }).click();
    
    const inputText = await page.locator('#inputText').textContent();
    expect(inputText).toBe('hi ');
  });

  test('should select predicted word', async ({ page }) => {
    // Perform a swipe
    const canvas = page.locator('#swipeCanvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Simple swipe
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + 200, box.y + 100, { steps: 10 });
    await page.mouse.up();
    
    // Wait for predictions
    await page.waitForTimeout(500);
    
    // Click first prediction
    const firstPrediction = page.locator('#suggestions > div').first();
    const predictionText = await firstPrediction.textContent();
    await firstPrediction.click();
    
    // Check that word was added to input
    const inputText = await page.locator('#inputText').textContent();
    expect(inputText).toContain(predictionText);
  });

  test('should display coordinate debug info when enabled', async ({ page }) => {
    // Enable debug mode
    await page.getByRole('button', { name: /Debug:/ }).click();
    
    // Start a swipe
    const canvas = page.locator('#swipeCanvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    await page.mouse.move(box.x + 100, box.y + 100);
    await page.mouse.down();
    
    // Coordinate display should be visible
    await expect(page.locator('#coordinateDisplay')).toBeVisible();
    
    // Check that coordinates are updating
    const xCoord = await page.locator('#xCoord').textContent();
    expect(parseInt(xCoord || '0')).toBeGreaterThanOrEqual(0);
    
    await page.mouse.move(box.x + 150, box.y + 150, { steps: 5 });
    
    const pathLength = await page.locator('#pathLength').textContent();
    expect(parseInt(pathLength || '0')).toBeGreaterThan(1);
    
    await page.mouse.up();
  });

  test('should handle touch events on mobile', async ({ page, browserName }) => {
    // Skip on non-chromium browsers as touch simulation varies
    test.skip(browserName !== 'chromium', 'Touch simulation only reliable in Chromium');
    
    const canvas = page.locator('#swipeCanvas');
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas not found');
    
    // Simulate touch swipe
    await page.touchscreen.tap(box.x + 100, box.y + 100);
    
    // Should work without errors (basic smoke test for touch)
    const statusText = await page.locator('#statusText').textContent();
    expect(statusText).toBe('Ready');
  });
});