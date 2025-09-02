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
    private debugMode: boolean = false;

    constructor() {
        this.canvas = document.getElementById('keyboard-canvas') as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.predictionsEl = document.getElementById('predictions')!;
        this.loadingEl = document.getElementById('loading')!;
        this.loadingProgressEl = document.getElementById('loading-progress')!;
        
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
        // Swipe tracking events
        this.swipeTracker.on('swipeStart', () => {
            this.keyboard.clearTrace();
            this.clearPredictions();
        });

        this.swipeTracker.on('swipeMove', (points) => {
            this.keyboard.drawTrace(points);
        });

        this.swipeTracker.on('swipeEnd', async (points) => {
            if (points.length < 3) {
                // Too short to be a meaningful swipe
                this.keyboard.clearTrace();
                return;
            }
            
            // Show loading state
            this.showLoadingPredictions();
            
            try {
                // Get predictions
                const predictions = await this.predictor.predict(points, 5);
                this.showPredictions(predictions);
                
                if (this.debugMode) {
                    console.log('Swipe points:', points);
                    console.log('Predictions:', predictions);
                }
            } catch (error: any) {
                console.error('Prediction error:', error);
                console.error('Error stack:', error?.stack);
                console.error('Error message:', error?.message);
                this.showError();
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
        });

        document.getElementById('debug-btn')?.addEventListener('click', () => {
            this.debugMode = !this.debugMode;
            const btn = document.getElementById('debug-btn') as HTMLButtonElement;
            btn.textContent = this.debugMode ? 'Debug: ON' : 'Debug: OFF';
            btn.style.background = this.debugMode ? '#f44336' : '#667eea';
        });

        // Prediction click handler
        this.predictionsEl.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            if (target.classList.contains('prediction')) {
                const word = target.textContent;
                if (word) {
                    this.selectWord(word);
                }
            }
        });
    }

    private clearPredictions() {
        this.predictionsEl.innerHTML = '<div class="no-predictions">Swipe on the keyboard to see predictions</div>';
    }

    private showLoadingPredictions() {
        this.predictionsEl.innerHTML = '<div class="no-predictions">Processing...</div>';
    }

    private showPredictions(predictions: Array<{word: string, score: number}>) {
        if (predictions.length === 0) {
            this.predictionsEl.innerHTML = '<div class="no-predictions">No predictions found</div>';
            return;
        }

        this.predictionsEl.innerHTML = predictions
            .map((pred, i) => `
                <div class="prediction ${i === 0 ? 'primary' : ''}" 
                     data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </div>
            `)
            .join('');
    }

    private showError() {
        this.predictionsEl.innerHTML = '<div class="no-predictions">Error processing swipe</div>';
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