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
        const container = this.canvas.parentElement!;
        const rect = container.getBoundingClientRect();
        
        // Set canvas size to match container while maintaining aspect ratio
        this.canvas.width = rect.width;
        this.canvas.height = rect.width * (215 / 360);
        
        // Update keyboard renderer with new dimensions
        this.keyboard.updateDimensions(this.canvas.width, this.canvas.height);
        this.keyboard.render();
    }

    private async loadModels() {
        this.loadingProgressEl.textContent = 'Loading encoder model...';
        await this.predictor.loadEncoder('/models/swipe_model_character.onnx');
        
        this.loadingProgressEl.textContent = 'Loading decoder model...';
        await this.predictor.loadDecoder('/models/swipe_decoder_character.onnx');
        
        this.loadingProgressEl.textContent = 'Loading tokenizer...';
        await this.predictor.loadTokenizer('/models/tokenizer_config.json');
        
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
            this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Swiping...</span>';
            this.statusEl.textContent = 'Swiping';
            this.statusEl.className = 'text-sm font-semibold text-blue-600 dark:text-blue-400';
        });

        this.swipeTracker.on('swipeMove', (points) => {
            this.keyboard.drawTrace(points);
            
            // Log unique keys as they're touched
            if (points.length > 0) {
                const lastPoint = points[points.length - 1];
                const key = this.keyboard.getKeyAt(lastPoint.x * (this.canvas.width / 360), lastPoint.y * (this.canvas.height / 215));
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
                this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Too short - try again</span>';
                this.statusEl.textContent = 'Ready';
                this.statusEl.className = 'text-sm font-semibold text-green-600 dark:text-green-400';
                return;
            }
            
            // Show loading state
            this.showLoadingPredictions();
            this.statusEl.textContent = 'Processing';
            this.statusEl.className = 'text-sm font-semibold text-yellow-600 dark:text-yellow-400';
            
            try {
                // Get predictions
                const predictions = await this.predictor.predict(points, 5);
                this.showPredictions(predictions);
                this.statusEl.textContent = 'Ready';
                this.statusEl.className = 'text-sm font-semibold text-green-600 dark:text-green-400';
                
                if (this.debugMode) {
                    console.log('Swipe points:', points);
                    console.log('Predictions:', predictions);
                }
            } catch (error: any) {
                console.error('Prediction error:', error);
                console.error('Error stack:', error?.stack);
                console.error('Error message:', error?.message);
                this.showError();
                this.statusEl.textContent = 'Error';
                this.statusEl.className = 'text-sm font-semibold text-red-600 dark:text-red-400';
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
            this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Touch the keyboard to start swiping...</span>';
            this.statusEl.textContent = 'Ready';
            this.statusEl.className = 'text-sm font-semibold text-green-600 dark:text-green-400';
        });

        document.getElementById('debug-btn')?.addEventListener('click', () => {
            this.debugMode = !this.debugMode;
            const btn = document.getElementById('debug-btn') as HTMLButtonElement;
            btn.textContent = this.debugMode ? 'Debug: ON' : 'Debug: OFF';
            if (this.debugMode) {
                btn.className = 'px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-all duration-150 active:scale-95';
            } else {
                btn.className = 'px-3 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 text-sm font-medium rounded-lg transition-all duration-150 active:scale-95';
            }
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
        this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full py-8">Swipe on the keyboard to see predictions</p>';
    }

    private showLoadingPredictions() {
        this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full py-8 animate-pulse">Processing...</p>';
    }

    private showPredictions(predictions: Array<{word: string, score: number}>) {
        if (predictions.length === 0) {
            this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full py-8">No predictions found</p>';
            return;
        }

        this.predictionsEl.innerHTML = predictions
            .map((pred, i) => `
                <button class="px-4 py-2 rounded-lg font-mono font-semibold transition-all duration-150
                              ${i === 0 ? 
                                'bg-indigo-600 hover:bg-indigo-700 text-white shadow-md' : 
                                'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 text-gray-900 dark:text-gray-100'}"
                        data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </button>
            `)
            .join('');
    }

    private showError() {
        this.predictionsEl.innerHTML = '<p class="text-red-500 dark:text-red-400 text-center w-full py-8">Error processing swipe</p>';
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