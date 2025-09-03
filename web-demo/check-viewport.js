import { chromium } from '@playwright/test';

async function checkViewport() {
    const browser = await chromium.launch();
    
    // Test desktop viewport
    const desktopPage = await browser.newPage({
        viewport: { width: 1280, height: 720 }
    });
    
    await desktopPage.goto('http://localhost:8080');
    await desktopPage.waitForTimeout(3000); // Wait for models to load
    
    // Check if content overflows
    const overflowInfo = await desktopPage.evaluate(() => {
        const body = document.body;
        const html = document.documentElement;
        return {
            bodyHeight: body.scrollHeight,
            bodyWidth: body.scrollWidth,
            viewportHeight: window.innerHeight,
            viewportWidth: window.innerWidth,
            overflowY: body.scrollHeight > window.innerHeight,
            overflowX: body.scrollWidth > window.innerWidth
        };
    });
    
    console.log('Desktop viewport (1280x720):');
    console.log(overflowInfo);
    
    if (overflowInfo.overflowY || overflowInfo.overflowX) {
        console.log('❌ OVERFLOW DETECTED!');
    } else {
        console.log('✅ No overflow');
    }
    
    await desktopPage.screenshot({ path: 'desktop-viewport.png', fullPage: false });
    
    // Test mobile viewport
    const mobilePage = await browser.newPage({
        viewport: { width: 375, height: 667 }
    });
    
    await mobilePage.goto('http://localhost:8080');
    await mobilePage.waitForTimeout(3000);
    
    const mobileOverflow = await mobilePage.evaluate(() => {
        const body = document.body;
        return {
            bodyHeight: body.scrollHeight,
            bodyWidth: body.scrollWidth,
            viewportHeight: window.innerHeight,
            viewportWidth: window.innerWidth,
            overflowY: body.scrollHeight > window.innerHeight,
            overflowX: body.scrollWidth > window.innerWidth
        };
    });
    
    console.log('\nMobile viewport (375x667):');
    console.log(mobileOverflow);
    
    if (mobileOverflow.overflowY || mobileOverflow.overflowX) {
        console.log('❌ OVERFLOW DETECTED!');
    } else {
        console.log('✅ No overflow');
    }
    
    await mobilePage.screenshot({ path: 'mobile-viewport.png', fullPage: false });
    
    await browser.close();
}

checkViewport().catch(console.error);