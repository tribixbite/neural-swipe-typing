export interface KeyPosition {
    x: number;
    y: number;
    char: string;
}

export class KeyboardRenderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private width: number;
    private height: number;
    private scale: number = 1;
    private keys: KeyPosition[] = [];
    private tracePoints: Array<{x: number, y: number}> = [];
    
    // QWERTY layout matching training data (360x215 coordinate space)
    private readonly KEYBOARD_LAYOUT = {
        // First row (numbers) - y ~= 67
        '1': {x: 18, y: 67}, '2': {x: 54, y: 67}, '3': {x: 90, y: 67},
        '4': {x: 126, y: 67}, '5': {x: 162, y: 67}, '6': {x: 198, y: 67},
        '7': {x: 234, y: 67}, '8': {x: 270, y: 67}, '9': {x: 306, y: 67},
        '0': {x: 342, y: 67},
        
        // Second row (qwerty) - y ~= 111
        'q': {x: 18, y: 111}, 'w': {x: 54, y: 111}, 'e': {x: 90, y: 111},
        'r': {x: 126, y: 111}, 't': {x: 162, y: 111}, 'y': {x: 198, y: 111},
        'u': {x: 234, y: 111}, 'i': {x: 270, y: 111}, 'o': {x: 306, y: 111},
        'p': {x: 342, y: 111},
        
        // Third row (asdf) - y ~= 155
        'a': {x: 36, y: 155}, 's': {x: 72, y: 155}, 'd': {x: 108, y: 155},
        'f': {x: 144, y: 155}, 'g': {x: 180, y: 155}, 'h': {x: 216, y: 155},
        'j': {x: 252, y: 155}, 'k': {x: 288, y: 155}, 'l': {x: 324, y: 155},
        
        // Fourth row (zxcv) - y ~= 199
        'z': {x: 72, y: 199}, 'x': {x: 108, y: 199}, 'c': {x: 144, y: 199},
        'v': {x: 180, y: 199}, 'b': {x: 216, y: 199}, 'n': {x: 252, y: 199},
        'm': {x: 288, y: 199}
    };

    constructor(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) {
        this.canvas = canvas;
        this.ctx = ctx;
        this.width = canvas.width;
        this.height = canvas.height;
        this.updateScale();
        this.initializeKeys();
    }

    updateDimensions(width: number, height: number) {
        this.width = width;
        this.height = height;
        this.updateScale();
    }

    private updateScale() {
        // Calculate scale factor to map 360x215 coordinate space to canvas size
        this.scale = this.width / 360;
    }

    private initializeKeys() {
        this.keys = [];
        for (const [char, pos] of Object.entries(this.KEYBOARD_LAYOUT)) {
            this.keys.push({
                char,
                x: pos.x,
                y: pos.y
            });
        }
    }

    render() {
        // Clear canvas
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw keys
        this.drawKeys();
        
        // Draw trace if exists
        if (this.tracePoints.length > 0) {
            this.drawTraceInternal();
        }
    }

    private drawKeys() {
        const keySize = 30 * this.scale;
        const fontSize = 16 * this.scale;

        for (const key of this.keys) {
            const x = key.x * this.scale;
            const y = key.y * this.scale;

            // Key background
            this.ctx.fillStyle = '#333';
            this.ctx.fillRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize
            );

            // Key border
            this.ctx.strokeStyle = '#555';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize
            );

            // Key text
            this.ctx.fillStyle = '#fff';
            this.ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(key.char.toUpperCase(), x, y);
        }
    }

    drawTrace(points: Array<{x: number, y: number, t: number}>) {
        this.tracePoints = points.map(p => ({
            x: p.x * this.scale,
            y: p.y * this.scale
        }));
        this.render();
    }

    private drawTraceInternal() {
        if (this.tracePoints.length < 2) return;

        // Draw trace line
        this.ctx.strokeStyle = 'rgba(102, 126, 234, 0.8)';
        this.ctx.lineWidth = 3 * this.scale;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Create gradient effect
        const gradient = this.ctx.createLinearGradient(
            this.tracePoints[0].x,
            this.tracePoints[0].y,
            this.tracePoints[this.tracePoints.length - 1].x,
            this.tracePoints[this.tracePoints.length - 1].y
        );
        gradient.addColorStop(0, 'rgba(102, 126, 234, 0.4)');
        gradient.addColorStop(1, 'rgba(118, 75, 162, 0.8)');
        this.ctx.strokeStyle = gradient;

        // Draw the path
        this.ctx.beginPath();
        this.ctx.moveTo(this.tracePoints[0].x, this.tracePoints[0].y);
        
        for (let i = 1; i < this.tracePoints.length; i++) {
            this.ctx.lineTo(this.tracePoints[i].x, this.tracePoints[i].y);
        }
        
        this.ctx.stroke();

        // Draw points
        for (let i = 0; i < this.tracePoints.length; i++) {
            const point = this.tracePoints[i];
            const radius = (i === 0 || i === this.tracePoints.length - 1) ? 
                          5 * this.scale : 2 * this.scale;
            
            this.ctx.fillStyle = i === 0 ? 
                                'rgba(102, 234, 126, 0.8)' : 
                                (i === this.tracePoints.length - 1 ? 
                                 'rgba(234, 102, 102, 0.8)' : 
                                 'rgba(255, 255, 255, 0.5)');
            
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    clearTrace() {
        this.tracePoints = [];
        this.render();
    }

    getKeyAt(x: number, y: number): string | null {
        // Convert canvas coordinates to keyboard space
        const keyX = x / this.scale;
        const keyY = y / this.scale;
        
        // Find nearest key within threshold
        const threshold = 20; // pixels in keyboard space
        let nearestKey: string | null = null;
        let minDistance = threshold;

        for (const key of this.keys) {
            const distance = Math.sqrt(
                Math.pow(keyX - key.x, 2) + 
                Math.pow(keyY - key.y, 2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestKey = key.char;
            }
        }

        return nearestKey;
    }

    // Convert canvas coordinates to keyboard space (360x215)
    canvasToKeyboard(canvasX: number, canvasY: number): {x: number, y: number} {
        return {
            x: canvasX / this.scale,
            y: canvasY / this.scale
        };
    }

    // Get keyboard layout for predictor
    getKeyboardLayout() {
        return this.KEYBOARD_LAYOUT;
    }
}