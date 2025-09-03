import { chromium } from '@playwright/test';

async function checkLightMode() {
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    
    await page.goto('http://localhost:8080');
    await page.waitForTimeout(3000); // Wait for models
    
    // Toggle to light mode
    await page.click('#theme-toggle');
    await page.waitForTimeout(500);
    
    await page.screenshot({ path: 'lightmode-viewport.png' });
    
    console.log('Light mode screenshot saved');
    await browser.close();
}

checkLightMode().catch(console.error);