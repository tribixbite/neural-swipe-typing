import { test, expect } from '@playwright/test';

test('test simple word "the"', async ({ page }) => {
  page.on('console', msg => {
    if (!msg.text().includes('Step') && !msg.text().includes('indices')) {
      console.log('Browser:', msg.text());
    }
  });
  
  await page.goto('/');
  await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
  
  const canvas = page.locator('#keyboard-canvas');
  const box = await canvas.boundingBox();
  
  if (!box) {
    throw new Error('Canvas not found');
  }
  
  // Create a swipe pattern for "the"
  // t -> h -> e
  const points = [
    {x: box.x + 162, y: box.y + 53},   // t
    {x: box.x + 216, y: box.y + 107},  // h  
    {x: box.x + 90, y: box.y + 53}     // e
  ];
  
  console.log('Swiping pattern for "the"...');
  
  // Perform swipe
  await page.mouse.move(points[0].x, points[0].y);
  await page.mouse.down();
  
  for (let i = 1; i < points.length; i++) {
    await page.mouse.move(points[i].x, points[i].y, { steps: 10 });
    await page.waitForTimeout(100);
  }
  
  await page.mouse.up();
  await page.waitForTimeout(2000);
  
  // Get predictions
  const predictions = await page.evaluate(() => {
    const predElements = document.querySelectorAll('.prediction');
    return Array.from(predElements).map(el => el.textContent?.trim() || '');
  });
  
  console.log('\n=== PREDICTIONS FOR "the" ===');
  predictions.forEach((pred, i) => {
    console.log(`${i+1}. "${pred}"`);
  });
  
  // Check if "the" is in predictions
  const hasThe = predictions.some(p => p.toLowerCase() === 'the');
  console.log(`\nContains "the": ${hasThe}`);
});