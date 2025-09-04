// Test the swipe functionality via console
// Run this in browser console on http://localhost:3456/swipe-onnx.html

async function testSwipe() {
    console.log('Starting swipe test...');
    
    // Wait for models to load
    if (!window.isModelReady) {
        console.log('Waiting for models to load...');
        await new Promise(resolve => {
            const checkInterval = setInterval(() => {
                if (window.isModelReady) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);
        });
    }
    
    console.log('Models ready, simulating swipe for "hello"');
    
    // Get canvas element
    const canvas = document.getElementById('swipeCanvas');
    if (!canvas) {
        console.error('Canvas not found!');
        return;
    }
    
    // Simulate swipe path for "hello"
    const swipePath = [
        {x: 216, y: 107, key: 'h'}, // h
        {x: 90, y: 53, key: 'e'},    // e
        {x: 324, y: 107, key: 'l'},  // l
        {x: 324, y: 107, key: 'l'},  // l (hold)
        {x: 306, y: 53, key: 'o'}    // o
    ];
    
    // Create mouse events
    const rect = canvas.getBoundingClientRect();
    
    // Start swipe
    const startEvent = new MouseEvent('mousedown', {
        clientX: rect.left + swipePath[0].x,
        clientY: rect.top + swipePath[0].y,
        bubbles: true
    });
    canvas.dispatchEvent(startEvent);
    
    // Move through points
    for (let i = 1; i < swipePath.length; i++) {
        const moveEvent = new MouseEvent('mousemove', {
            clientX: rect.left + swipePath[i].x,
            clientY: rect.top + swipePath[i].y,
            bubbles: true
        });
        canvas.dispatchEvent(moveEvent);
        await new Promise(r => setTimeout(r, 50)); // Small delay between points
    }
    
    // End swipe
    const endEvent = new MouseEvent('mouseup', {
        clientX: rect.left + swipePath[swipePath.length - 1].x,
        clientY: rect.top + swipePath[swipePath.length - 1].y,
        bubbles: true
    });
    canvas.dispatchEvent(endEvent);
    
    console.log('Swipe simulation complete. Check predictions.');
}

// Auto-run test
testSwipe().catch(console.error);