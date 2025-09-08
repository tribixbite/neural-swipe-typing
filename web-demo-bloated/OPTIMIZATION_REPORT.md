# Swipe-ONNX.html Optimization Report

## Executive Summary
This report identifies key performance bottlenecks, redundancies, and optimization opportunities in the swipe-onnx.html file.

## 1. DOM Performance Issues

### Problem: Repeated DOM Queries
The code repeatedly queries for the same DOM elements:
```javascript
// Called multiple times throughout the code
document.getElementById('suggestions')
document.getElementById('inputText')
document.getElementById('debugStatus')
```

### Solution: Cache DOM References
```javascript
// Cache all DOM elements at initialization
const DOM = {
    suggestions: null,
    inputText: null,
    debugStatus: null,
    coordinateDisplay: null,
    // ... other elements
};

function initDOMCache() {
    DOM.suggestions = document.getElementById('suggestions');
    DOM.inputText = document.getElementById('inputText');
    DOM.debugStatus = document.getElementById('debugStatus');
    // ... cache other elements
}
```

## 2. Canvas Rendering Inefficiencies

### Problem: Full Canvas Redraw on Every Move
```javascript
function drawNeonTrail() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clears entire canvas
    // Recreates gradient every time
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    // ...
}
```

### Solution: Incremental Drawing + Gradient Caching
```javascript
let cachedGradient = null;

function initCanvasGradient() {
    cachedGradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    cachedGradient.addColorStop(0, '#00d4ff');
    cachedGradient.addColorStop(0.5, '#b300ff');
    cachedGradient.addColorStop(1, '#ff00d4');
}

function drawNeonTrailIncremental() {
    if (swipePath.length < 2) return;
    
    // Only draw the new segment
    const lastIdx = swipePath.length - 1;
    const prevPoint = swipePath[lastIdx - 1];
    const currPoint = swipePath[lastIdx];
    
    // Draw only the new line segment
    ctx.strokeStyle = cachedGradient;
    ctx.beginPath();
    ctx.moveTo(prevPoint.canvasX, prevPoint.canvasY);
    ctx.lineTo(currPoint.canvasX, currPoint.canvasY);
    ctx.stroke();
}
```

## 3. Memory Leaks and Inefficiencies

### Problem: Unbounded Array Growth
```javascript
swipePath.push({...}); // Can grow infinitely during long swipes
```

### Solution: Implement Path Sampling
```javascript
const MAX_PATH_POINTS = 100;
let pathSampleCounter = 0;
const SAMPLE_RATE = 3; // Keep every 3rd point

function addToSwipePath(point) {
    pathSampleCounter++;
    
    // Always keep first and last points
    if (swipePath.length === 0 || pathSampleCounter % SAMPLE_RATE === 0) {
        swipePath.push(point);
        
        // Limit total points
        if (swipePath.length > MAX_PATH_POINTS) {
            // Downsample: keep every other point
            swipePath = swipePath.filter((_, i) => i % 2 === 0);
        }
    }
}
```

### Problem: Tensor Memory Not Released
```javascript
// Tensors created but never explicitly disposed
const trajectoryTensor = new ort.Tensor(...);
const nearestKeysTensor = new ort.Tensor(...);
```

### Solution: Proper Tensor Disposal
```javascript
async function runInference(features) {
    let trajectoryTensor, nearestKeysTensor, srcMaskTensor;
    
    try {
        trajectoryTensor = new ort.Tensor(...);
        nearestKeysTensor = new ort.Tensor(...);
        srcMaskTensor = new ort.Tensor(...);
        
        const result = await encoderSession.run({...});
        return result;
    } finally {
        // Dispose tensors to free memory
        trajectoryTensor?.dispose?.();
        nearestKeysTensor?.dispose?.();
        srcMaskTensor?.dispose?.();
    }
}
```

## 4. Redundant Calculations

### Problem: Key Bounds Calculated on Every Touch
```javascript
function getKeyAtPosition(clientX, clientY) {
    const keys = document.querySelectorAll('.key[data-key]');
    for (let key of keys) {
        const rect = key.getBoundingClientRect(); // Expensive!
        // ...
    }
}
```

### Solution: Pre-calculate and Cache Key Bounds
```javascript
const keyBoundsCache = new Map();

function cacheKeyBounds() {
    keyBoundsCache.clear();
    document.querySelectorAll('.key[data-key]').forEach(key => {
        const rect = key.getBoundingClientRect();
        keyBoundsCache.set(key.dataset.key, {
            element: key,
            left: rect.left,
            right: rect.right,
            top: rect.top,
            bottom: rect.bottom
        });
    });
}

function getKeyAtPositionCached(clientX, clientY) {
    for (const [key, bounds] of keyBoundsCache) {
        if (clientX >= bounds.left && clientX <= bounds.right &&
            clientY >= bounds.top && clientY <= bounds.bottom) {
            return bounds.element;
        }
    }
    return null;
}
```

## 5. Vocabulary Loading Optimization

### Problem: Loading Entire 1.3MB File into Memory
```javascript
const text = await response.text(); // Loads entire file
const lines = text.split('\n'); // Creates huge array
```

### Solution: Stream Processing
```javascript
async function loadVocabularyStreaming() {
    const response = await fetch('models/english_vocab.txt');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    let buffer = '';
    let lineNumber = 0;
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line
        
        for (const line of lines) {
            const word = line.trim().toLowerCase();
            if (word && word.match(/^[a-z]+$/)) {
                vocabularyDict.set(word, lineNumber++);
            }
        }
    }
}
```

## 6. Event Handler Optimization

### Problem: Individual Event Listeners per Key
```javascript
document.querySelectorAll('.key').forEach(key => {
    key.addEventListener('click', function(e) { ... });
});
```

### Solution: Event Delegation
```javascript
// Single event listener on keyboard container
document.getElementById('keyboard').addEventListener('click', function(e) {
    const key = e.target.closest('.key[data-key]');
    if (key && !isDrawing) {
        handleKeyTap(key);
    }
});
```

## 7. Unused Code Removal

### Remove:
- `commonTwoCharWords` array (populated but barely used)
- `wordsByLength` Map (created but never queried)
- `window.isModelReady` (duplicate of `isModelReady`)
- Extensive console.log statements in production

## 8. Beam Search Optimization

### Problem: Creating Many Intermediate Objects
```javascript
const candidates = [];
for (const beam of beams) {
    // Creates new objects for each candidate
    const newBeam = {
        tokens: [...beam.tokens, BigInt(idx)],
        score: beam.score + Math.log(probs[idx]),
        finished: idx === EOS_IDX
    };
}
```

### Solution: Object Pool Pattern
```javascript
class BeamPool {
    constructor(size) {
        this.pool = Array(size).fill(null).map(() => ({
            tokens: new Array(20),
            tokenLength: 0,
            score: 0,
            finished: false
        }));
        this.index = 0;
    }
    
    get() {
        const beam = this.pool[this.index++ % this.pool.length];
        beam.tokenLength = 0;
        beam.score = 0;
        beam.finished = false;
        return beam;
    }
}
```

## 9. Performance Monitoring

### Add Performance Metrics
```javascript
const Performance = {
    swipeStartTime: 0,
    inferenceStartTime: 0,
    
    startSwipeTimer() {
        this.swipeStartTime = performance.now();
    },
    
    logInferenceTime() {
        const duration = performance.now() - this.inferenceStartTime;
        console.debug(`Inference took ${duration.toFixed(2)}ms`);
    }
};
```

## 10. Recommended Refactoring Structure

```javascript
// Modularize the code
const SwipeKeyboard = {
    // Configuration
    config: {
        NORMALIZED_WIDTH: 360,
        NORMALIZED_HEIGHT: 215,
        MAX_SEQUENCE_LENGTH: 150,
        MAX_PATH_POINTS: 100
    },
    
    // Cached DOM elements
    dom: {},
    
    // Cached data
    cache: {
        keyBounds: new Map(),
        gradient: null
    },
    
    // State
    state: {
        isDrawing: false,
        swipePath: [],
        keySequence: []
    },
    
    // Methods organized by functionality
    init() {},
    rendering: {},
    input: {},
    inference: {},
    ui: {}
};
```

## Performance Impact Summary

| Optimization | Expected Impact | Difficulty |
|--------------|----------------|------------|
| DOM Caching | 10-20% UI responsiveness | Easy |
| Canvas Incremental Drawing | 30-50% rendering performance | Medium |
| Key Bounds Caching | 20-30% touch response | Easy |
| Memory Management | Prevents degradation over time | Medium |
| Event Delegation | 5-10% initialization time | Easy |
| Vocabulary Streaming | 50% memory usage reduction | Hard |
| Tensor Disposal | Prevents memory leaks | Easy |
| Remove Unused Code | 5-10% file size reduction | Easy |

## Implementation Priority

1. **High Priority** (Quick wins):
   - DOM element caching
   - Remove console.logs and unused code
   - Event delegation
   - Tensor disposal

2. **Medium Priority** (Significant impact):
   - Key bounds caching
   - Canvas optimization
   - Path sampling

3. **Low Priority** (Nice to have):
   - Vocabulary streaming
   - Beam search optimization
   - Code modularization
