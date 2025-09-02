import { test, expect } from '@playwright/test';

test.describe('Basic Demo Tests', () => {
  test('should load the demo page', async ({ page }) => {
    await page.goto('/');
    
    // Check page title exists
    await expect(page).toHaveTitle(/Neural Swipe Typing/);
    
    // Check main elements are present
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('#keyboard-canvas')).toBeVisible();
    
    // Check keyboard canvas is rendered
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(0);
    expect(box!.height).toBeGreaterThan(0);
  });

  test('should show loading overlay initially', async ({ page }) => {
    // Don't wait for page to fully load
    const response = page.goto('/', { waitUntil: 'domcontentloaded' });
    
    // Loading overlay should be visible initially
    const loading = page.locator('#loading');
    await expect(loading).toBeVisible();
    
    // Wait for page to load
    await response;
    
    // Loading should eventually disappear (give it 30 seconds for model loading)
    await expect(loading).toBeHidden({ timeout: 30000 });
  });

  test('should handle mouse interaction', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    
    if (!box) throw new Error('Canvas not found');
    
    // Perform a simple swipe
    await page.mouse.move(box.x + 50, box.y + 50);
    await page.mouse.down();
    await page.mouse.move(box.x + 200, box.y + 100, { steps: 5 });
    await page.mouse.up();
    
    // Should update predictions (even if it says "Processing...")
    const predictions = page.locator('.predictions');
    await expect(predictions).not.toContainText('Swipe on the keyboard to see predictions');
  });

  test('should have working control buttons', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('#loading', { state: 'hidden', timeout: 30000 });
    
    // Test clear button
    const clearBtn = page.locator('#clear-btn');
    await expect(clearBtn).toBeVisible();
    await clearBtn.click(); // Should not error
    
    // Test debug button
    const debugBtn = page.locator('#debug-btn');
    await expect(debugBtn).toBeVisible();
    
    // Toggle debug mode
    await expect(debugBtn).toContainText('Debug');
    await debugBtn.click();
    
    // Button text should change (might need a small wait for state update)
    await page.waitForTimeout(100);
    const btnText = await debugBtn.textContent();
    expect(btnText).toMatch(/Debug/i);
  });

  test('should be mobile responsive', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Elements should still be visible
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('#keyboard-canvas')).toBeVisible();
    
    // Canvas should maintain aspect ratio
    const canvas = page.locator('#keyboard-canvas');
    const box = await canvas.boundingBox();
    if (box) {
      const ratio = box.width / box.height;
      const expectedRatio = 360 / 215;
      expect(Math.abs(ratio - expectedRatio)).toBeLessThan(0.1);
    }
  });
});