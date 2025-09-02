import { test, expect } from '@playwright/test';

test('test encoder model only', async ({ page }) => {
  const logs: string[] = [];
  
  page.on('console', msg => {
    logs.push(msg.text());
    console.log('Browser:', msg.text());
  });
  
  page.on('pageerror', error => {
    console.error('Page Error:', error.message);
  });
  
  await page.goto('/test-encoder.html');
  
  // Wait for result
  await page.waitForTimeout(10000);
  
  // Check status
  const status = await page.locator('#status').textContent();
  const output = await page.locator('#output').textContent();
  
  console.log('Status:', status);
  console.log('Output:', output);
  
  // Check if successful
  expect(status).toContain('Success');
});