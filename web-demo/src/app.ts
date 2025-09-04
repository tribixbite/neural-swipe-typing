// Use global ort if available, otherwise try to import
declare const ort: any;

import { KeyboardRenderer } from './keyboard';
import { SwipePredictor } from './predictor';
import { SwipeTracker } from './swipe-tracker';

class SwipeTypingApp {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private keyboard: KeyboardRenderer;
    private predictor: SwipePredictor;
    private swipeTracker: SwipeTracker;
    private predictionsEl: HTMLElement;
    private loadingEl: HTMLElement;
    private loadingProgressEl: HTMLElement;
    private swipeCharsEl: HTMLElement;
    private statusEl: HTMLElement;
    private debugMode: boolean = false;

    constructor() {
        this.canvas = document.getElementById('keyboard-canvas') as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.predictionsEl = document.getElementById('predictions')!;
        this.loadingEl = document.getElementById('loading')!;
        this.loadingProgressEl = document.getElementById('loading-progress')!;
        this.swipeCharsEl = document.getElementById('swipe-chars')!;
        this.statusEl = document.getElementById('status')!;
        
        this.keyboard = new KeyboardRenderer(this.canvas, this.ctx);
        this.predictor = new SwipePredictor();
        this.swipeTracker = new SwipeTracker(this.canvas, this.keyboard);
        
        this.init();
    }

    private async init() {
        try {
            // Setup canvas dimensions
            this.setupCanvas();
            window.addEventListener('resize', () => this.setupCanvas());
            
            // Load models
            await this.loadModels();
            
            // Setup event handlers
            this.setupEventHandlers();
            
            // Initial render
            this.keyboard.render();
            
            // Hide loading overlay
            this.loadingEl.style.display = 'none';
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.loadingProgressEl.textContent = 'Failed to load models. Please refresh.';
            this.loadingProgressEl.style.color = '#ff5252';
        }
    }

    private setupCanvas() {
        // Canvas dimensions are fixed at 360x215 for consistent coordinate mapping
        // CSS handles the display sizing
        this.canvas.width = 360;
        this.canvas.height = 215;
        
        // Update keyboard renderer
        this.keyboard.updateDimensions(360, 215);
        this.keyboard.render();
    }

    private async loadModels() {
        // Use relative paths for GitHub Pages compatibility
        const basePath = window.location.hostname === 'localhost' ? '' : '.';
        
        this.loadingProgressEl.textContent = 'Loading encoder model...';
        await this.predictor.loadEncoder(`${basePath}/models/swipe_model_character_quant.onnx`);
        
        this.loadingProgressEl.textContent = 'Loading decoder model...';
        await this.predictor.loadDecoder(`${basePath}/models/swipe_decoder_character_quant.onnx`);
        
        this.loadingProgressEl.textContent = 'Loading tokenizer...';
        await this.predictor.loadTokenizer(`${basePath}/models/tokenizer_config.json`);
        
        this.loadingProgressEl.textContent = 'Models loaded successfully!';
    }

    private setupEventHandlers() {
        let loggedKeys: Set<string> = new Set();
        let swipedChars: string[] = [];
        
        // Swipe tracking events
        this.swipeTracker.on('swipeStart', () => {
            this.keyboard.clearTrace();
            this.clearPredictions();
            loggedKeys.clear();  // Reset logged keys for new swipe
            swipedChars = [];    // Reset swiped characters
            this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.7); font-size: 16px;">Swiping...</span>';
            this.updateStatus('Swiping');
        });

        this.swipeTracker.on('swipeMove', (points) => {
            this.keyboard.drawTrace(points);
            
            // Log unique keys as they're touched
            if (points.length > 0) {
                const lastPoint = points[points.length - 1];
                // Points are already in keyboard space (360x215)
                // getKeyAt also expects keyboard space coordinates
                const key = this.keyboard.getKeyAt(lastPoint.x, lastPoint.y);
                if (key && !loggedKeys.has(key)) {
                    console.log(`Key: ${key.toUpperCase()}`);
                    loggedKeys.add(key);
                    swipedChars.push(key.toUpperCase());
                    
                    // Update real-time display
                    this.swipeCharsEl.innerHTML = swipedChars.join(' → ');
                }
            }
        });

        this.swipeTracker.on('swipeEnd', async (points) => {
            if (points.length < 3) {
                // Too short to be a meaningful swipe
                this.keyboard.clearTrace();
                this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.7); font-size: 16px;">Too short - try again</span>';
                this.updateStatus('Ready');
                return;
            }
            
            // Show loading state
            this.showLoadingPredictions();
            this.updateStatus('Processing');
            
            try {
                // Get predictions
                const predictions = await this.predictor.predict(points, 5);
                this.showPredictions(predictions);
                this.updateStatus('Ready');
                
                if (this.debugMode) {
                    console.log('Swipe points:', points);
                    console.log('Predictions:', predictions);
                }
            } catch (error: any) {
                console.error('Prediction error:', error);
                console.error('Error stack:', error?.stack);
                console.error('Error message:', error?.message);
                this.showError();
                this.updateStatus('Error');
            }
            
            // Clear trace after a delay
            setTimeout(() => {
                this.keyboard.clearTrace();
            }, 1000);
        });

        // Control buttons
        document.getElementById('clear-btn')?.addEventListener('click', () => {
            this.keyboard.clearTrace();
            this.clearPredictions();
            this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.6); font-size: 14px;">Touch the keyboard to start swiping...</span>';
            this.updateStatus('Ready');
        });

        document.getElementById('debug-btn')?.addEventListener('click', () => {
            this.debugMode = !this.debugMode;
            const btn = document.getElementById('debug-btn') as HTMLButtonElement;
            btn.textContent = this.debugMode ? 'Debug: ON' : 'Debug: OFF';
            // Button style stays the same (inline styles already set in HTML)
        });

        // Prediction click handler
        this.predictionsEl.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            if (target.tagName === 'BUTTON') {
                const word = target.textContent?.trim();
                if (word) {
                    this.selectWord(word);
                }
            }
        });
    }

    private clearPredictions() {
        this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.6); text-align: center; width: 100%; font-size: 14px;">Swipe on the keyboard to see predictions</p>';
    }

    private showLoadingPredictions() {
        this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.8); text-align: center; width: 100%; animation: pulse 2s infinite;">Processing...</p>';
    }

    private showPredictions(predictions: Array<{word: string, score: number}>) {
        if (predictions.length === 0) {
            this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.6); text-align: center; width: 100%;">No predictions found</p>';
            return;
        }

        this.predictionsEl.innerHTML = predictions
            .map((pred, i) => `
                <button style="
                    background: ${i === 0 ? 'rgba(255, 255, 255, 0.25)' : 'rgba(255, 255, 255, 0.15)'};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    font-weight: ${i === 0 ? '600' : '400'};
                    font-family: monospace;"
                    onmouseover="this.style.background='rgba(255,255,255,0.3)'; this.style.transform='translateY(-2px)';"
                    onmouseout="this.style.background='${i === 0 ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.15)'}'; this.style.transform='translateY(0)';"
                    data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </button>
            `)
            .join('');
    }

    private showError() {
        this.predictionsEl.innerHTML = '<p style="color: #ff6b6b; text-align: center; width: 100%;">Error processing swipe</p>';
    }

    private updateStatus(text: string) {
        this.statusEl.textContent = `Status: ${text}`;
    }

    private selectWord(word: string) {
        // In a real app, this would insert the word into a text field
        console.log('Selected word:', word);
        
        // Visual feedback
        const predictions = this.predictionsEl.querySelectorAll('.prediction');
        predictions.forEach(pred => {
            if (pred.textContent === word) {
                pred.classList.add('selected');
                // Flash effect
                setTimeout(() => pred.classList.remove('selected'), 300);
            }
        });
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new SwipeTypingApp());
} else {
    new SwipeTypingApp();
}