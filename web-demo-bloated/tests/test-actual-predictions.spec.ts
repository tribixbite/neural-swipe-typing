import { test, expect } from '@playwright/test';

test('check actual word predictions', async ({ page }) => {
  page.on('console', msg => {
    console.log('Browser:', msg.text());
  });
  
  page.on('pageerror', error => {
    console.error('Page Error:', error.message);
  });
  
  await page.goto('/');
  
  // Wait for models to load
  await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
  
  // Get canvas
  const canvas = page.locator('#keyboard-canvas');
  const box = await canvas.boundingBox();
  
  if (!box) {
    throw new Error('Canvas not found');
  }
  
  // Create a swipe pattern for "hello"
  // Need to scale keyboard coordinates (360x215) to actual canvas size
  const scaleX = box.width / 360;
  const scaleY = box.height / 215;
  
  // h -> e -> l -> l -> o
  const points = [
    {x: box.x + 216 * scaleX, y: box.y + 107 * scaleY},  // h
    {x: box.x + 90 * scaleX, y: box.y + 53 * scaleY},    // e
    {x: box.x + 324 * scaleX, y: box.y + 107 * scaleY},  // l
    {x: box.x + 324 * scaleX, y: box.y + 107 * scaleY},  // l (stay)
    {x: box.x + 306 * scaleX, y: box.y + 53 * scaleY}    // o
  ];
  
  console.log('Swiping pattern for "hello"...');
  
  // Perform swipe
  await page.mouse.move(points[0].x, points[0].y);
  await page.mouse.down();
  
  for (let i = 1; i < points.length; i++) {
    await page.mouse.move(points[i].x, points[i].y, { steps: 5 });
    await page.waitForTimeout(50);
  }
  
  await page.mouse.up();
  
  // Wait for predictions
  await page.waitForTimeout(2000);
  
  // Get predictions
  const predictions = await page.evaluate(() => {
    const predElements = document.querySelectorAll('.prediction');
    return Array.from(predElements).map(el => ({
      word: el.textContent,
      isPrimary: el.classList.contains('primary')
    }));
  });
  
  console.log('\n=== PREDICTIONS ===');
  predictions.forEach((pred, i) => {
    console.log(`${i+1}. "${pred.word}"${pred.isPrimary ? ' (PRIMARY)' : ''}`);
  });
  
  // Check if we have any predictions
  expect(predictions.length).toBeGreaterThan(0);
  
  // Log what we expected vs what we got
  console.log('\nExpected: "hello" or similar');
  console.log('Got:', predictions.map(p => p.word).join(', '));
});