import { test, expect } from '@playwright/test';

test('Swipe ONNX Page Loads', async ({ page }) => {
  // Navigate to the page with a longer timeout
  await page.goto('http://localhost:3456/swipe-onnx.html', { 
    waitUntil: 'networkidle',
    timeout: 30000 
  });
  
  // Take a screenshot to see what's happening
  await page.screenshot({ path: 'swipe-onnx-load.png' });
  
  // Check if the page title is correct
  const title = await page.title();
  console.log('Page title:', title);
  expect(title).toBe('Neural Swipe Typing - ONNX Demo');
  
  // Check if the header is visible
  const header = await page.locator('h1').textContent();
  console.log('Header text:', header);
  
  // Check if canvas exists (might not be visible initially due to loading)
  const canvasExists = await page.locator('#swipeCanvas').count();
  console.log('Canvas elements found:', canvasExists);
  
  // Wait for loading to complete (if loading overlay exists)
  const loadingOverlay = page.locator('#loadingOverlay');
  if (await loadingOverlay.count() > 0) {
    console.log('Waiting for loading overlay to hide...');
    await loadingOverlay.waitFor({ state: 'hidden', timeout: 30000 }).catch(() => {
      console.log('Loading overlay did not hide in time');
    });
  }
  
  // Try clicking and dragging on the canvas
  const canvas = page.locator('#swipeCanvas');
  if (await canvas.count() > 0) {
    const box = await canvas.boundingBox();
    if (box) {
      console.log('Canvas dimensions:', box);
      
      // Attempt a simple drag
      await page.mouse.move(box.x + 100, box.y + 100);
      await page.mouse.down();
      await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
      await page.mouse.up();
      
      console.log('Drag performed');
      
      // Take another screenshot after drag
      await page.screenshot({ path: 'swipe-onnx-after-drag.png' });
    }
  }
});