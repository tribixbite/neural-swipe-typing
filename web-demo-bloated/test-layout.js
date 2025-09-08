import { chromium } from '@playwright/test';

async function testLayout() {
    const browser = await chromium.launch();
    
    // Test mobile viewport first
    console.log('Testing mobile viewport (375x667):');
    let page = await browser.newPage({ viewport: { width: 375, height: 667 } });
    await testViewport(page);
    
    // Test desktop viewport
    console.log('\nTesting desktop viewport (1920x1080):');
    page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    await testViewport(page);
    
    await browser.close();
}

async function testViewport(page) {
    await page.goto('http://localhost:8080');
    await page.waitForTimeout(3000);
    
    // Simulate predictions by injecting HTML
    await page.evaluate(() => {
        const predictions = document.getElementById('predictions');
        if (predictions) {
            // Add many predictions to test overflow handling
            predictions.innerHTML = `
                <button class="px-2 py-1">hello</button>
                <button class="px-2 py-1">help</button>
                <button class="px-2 py-1">held</button>
                <button class="px-2 py-1">hell</button>
                <button class="px-2 py-1">helm</button>
                <button class="px-2 py-1">helicopter</button>
                <button class="px-2 py-1">helping</button>
                <button class="px-2 py-1">helped</button>
                <button class="px-2 py-1">helpful</button>
                <button class="px-2 py-1">helpless</button>
            `;
        }
    });
    
    // Check for overflow
    const overflow = await page.evaluate(() => {
        const body = document.body;
        const html = document.documentElement;
        
        // Check if any element is outside viewport
        const elements = document.querySelectorAll('*');
        let outsideViewport = [];
        
        elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            // Skip elements that are meant to scroll (have overflow-y: auto)
            const computed = window.getComputedStyle(el);
            const isScrollable = computed.overflowY === 'auto' || computed.overflowY === 'scroll';
            
            // Only report elements that are visible and not in scrollable containers
            if (!isScrollable && rect.height > 0 && rect.width > 0) {
                if (rect.bottom > window.innerHeight || rect.right > window.innerWidth) {
                    outsideViewport.push({
                        element: el.tagName + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                        bottom: rect.bottom,
                        viewportHeight: window.innerHeight,
                        overflow: rect.bottom - window.innerHeight
                    });
                }
            }
        });
        
        // Check predictions specifically
        const predictions = document.getElementById('predictions');
        const predRect = predictions?.getBoundingClientRect();
        
        return {
            bodyHeight: body.scrollHeight,
            viewportHeight: window.innerHeight,
            overflowY: body.scrollHeight > window.innerHeight,
            elementsOutside: outsideViewport,
            predictionsInfo: predRect ? {
                top: predRect.top,
                bottom: predRect.bottom,
                height: predRect.height,
                scrollHeight: predictions.scrollHeight
            } : null
        };
    });
    
    console.log(overflow);
    
    if (overflow.elementsOutside.length > 0) {
        console.log('❌ Elements outside viewport:', overflow.elementsOutside);
    } else {
        console.log('✅ All elements within viewport');
    }
}

testLayout().catch(console.error);