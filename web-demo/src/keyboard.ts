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
    // Layout fills entire canvas without number row
    private readonly KEYBOARD_LAYOUT = {
        // First row (qwerty) - top third of canvas
        'q': {x: 18, y: 53}, 'w': {x: 54, y: 53}, 'e': {x: 90, y: 53},
        'r': {x: 126, y: 53}, 't': {x: 162, y: 53}, 'y': {x: 198, y: 53},
        'u': {x: 234, y: 53}, 'i': {x: 270, y: 53}, 'o': {x: 306, y: 53},
        'p': {x: 342, y: 53},
        
        // Second row (asdf) - middle third of canvas
        'a': {x: 36, y: 107}, 's': {x: 72, y: 107}, 'd': {x: 108, y: 107},
        'f': {x: 144, y: 107}, 'g': {x: 180, y: 107}, 'h': {x: 216, y: 107},
        'j': {x: 252, y: 107}, 'k': {x: 288, y: 107}, 'l': {x: 324, y: 107},
        
        // Third row (zxcv) - bottom third of canvas
        'z': {x: 72, y: 161}, 'x': {x: 108, y: 161}, 'c': {x: 144, y: 161},
        'v': {x: 180, y: 161}, 'b': {x: 216, y: 161}, 'n': {x: 252, y: 161},
        'm': {x: 288, y: 161}
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
        // Clear canvas - check dark mode
        const isDark = document.documentElement.classList.contains('dark');
        this.ctx.fillStyle = isDark ? '#111827' : '#f9fafb';  // gray-900 : gray-50
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw keys
        this.drawKeys();
        
        // Draw trace if exists
        if (this.tracePoints.length > 0) {
            this.drawTraceInternal();
        }
    }

    private drawKeys() {
        const keySize = 48 * this.scale;  // Larger keys for better touch targets
        const fontSize = 22 * this.scale;  // Better readability
        const isDark = document.documentElement.classList.contains('dark');

        for (const key of this.keys) {
            const x = key.x * this.scale;
            const y = key.y * this.scale;

            // Key background with rounded corners effect
            const gradient = this.ctx.createLinearGradient(
                x - keySize / 2, y - keySize / 2,
                x + keySize / 2, y + keySize / 2
            );
            
            if (isDark) {
                gradient.addColorStop(0, '#374151');  // gray-700
                gradient.addColorStop(1, '#1f2937');  // gray-800
            } else {
                gradient.addColorStop(0, '#ffffff');
                gradient.addColorStop(1, '#f3f4f6');  // gray-100
            }
            
            this.ctx.fillStyle = gradient;
            this.roundRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize,
                4 * this.scale
            );
            this.ctx.fill();

            // Key border
            this.ctx.strokeStyle = isDark ? '#4b5563' : '#d1d5db';  // gray-600 : gray-300
            this.ctx.lineWidth = 1.5;
            this.roundRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize,
                4 * this.scale
            );
            this.ctx.stroke();

            // Key text with better contrast
            this.ctx.fillStyle = isDark ? '#f3f4f6' : '#111827';  // gray-100 : gray-900
            this.ctx.font = `bold ${fontSize}px 'JetBrains Mono', 'SF Mono', 'Consolas', monospace`;
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

    // Helper method to draw rounded rectangles
    private roundRect(x: number, y: number, width: number, height: number, radius: number) {
        this.ctx.beginPath();
        this.ctx.moveTo(x + radius, y);
        this.ctx.lineTo(x + width - radius, y);
        this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        this.ctx.lineTo(x + width, y + height - radius);
        this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        this.ctx.lineTo(x + radius, y + height);
        this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        this.ctx.lineTo(x, y + radius);
        this.ctx.quadraticCurveTo(x, y, x + radius, y);
        this.ctx.closePath();
    }
}