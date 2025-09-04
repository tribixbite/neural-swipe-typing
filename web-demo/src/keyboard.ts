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
    private keyboardWidth: number = 360;
    private keyboardHeight: number = 215;
    private offsetX: number = 0;
    private offsetY: number = 0;
    private keys: KeyPosition[] = [];
    private tracePoints: Array<{x: number, y: number}> = [];
    private activeKey: string | null = null;
    
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
        console.log('KeyboardRenderer initialized with:', {
            canvasWidth: canvas.width,
            canvasHeight: canvas.height,
            expectedWidth: 360,
            expectedHeight: 215
        });
        this.updateScale();
        this.initializeKeys();
    }

    updateDimensions(width: number, height: number) {
        console.log('updateDimensions called with:', width, height);
        this.width = width;
        this.height = height;
        this.updateScale();
    }

    private updateScale() {
        // Canvas is always 360x215, so scale is always 1
        // Keys are drawn at their natural positions
        this.scale = 1;
        
        // No offsets needed
        this.offsetX = 0;
        this.offsetY = 0;
        this.keyboardWidth = 360;
        this.keyboardHeight = 215;
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
        // Clear canvas with dark keyboard background
        this.ctx.fillStyle = '#1a1a2e';  // Dark purple-ish background
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw keys
        this.drawKeys();
        
        // Draw trace if exists
        if (this.tracePoints.length > 0) {
            this.drawTraceInternal();
        }
    }

    private drawKeys() {
        const keySize = 32;  // Slightly larger keys
        const fontSize = 20;  // Larger font for better visibility

        for (const key of this.keys) {
            const x = key.x;
            const y = key.y;
            const isActive = this.activeKey === key.char;
            
            // Scale up active key more dramatically
            const currentKeySize = isActive ? keySize * 1.2 : keySize;
            const currentFontSize = isActive ? fontSize * 1.15 : fontSize;

            // Save context for shadows
            this.ctx.save();
            
            // Add shadow for depth effect
            this.ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
            this.ctx.shadowBlur = 4;
            this.ctx.shadowOffsetX = 0;
            this.ctx.shadowOffsetY = 2;

            // Key background - dark purple/gray
            this.ctx.fillStyle = isActive ? '#5865f2' : '#2d2d44';
            this.roundRect(
                x - currentKeySize / 2,
                y - currentKeySize / 2,
                currentKeySize,
                currentKeySize,
                6
            );
            this.ctx.fill();

            // Restore context to remove shadow
            this.ctx.restore();

            // No border for cleaner look
            
            // Key text - white for visibility
            this.ctx.save();
            
            // Text styling
            this.ctx.fillStyle = isActive ? '#ffffff' : 'rgba(255, 255, 255, 0.95)';
            this.ctx.font = `400 ${currentFontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            
            this.ctx.fillText(key.char.toUpperCase(), x, y);
            
            this.ctx.restore();
        }
    }

    drawTrace(points: Array<{x: number, y: number, t: number}>) {
        // Points are already in 360x215 space, no scaling needed
        this.tracePoints = points.map(p => ({
            x: p.x,
            y: p.y
        }));
        
        // Update active key based on latest point
        if (points.length > 0) {
            const lastPoint = points[points.length - 1];
            const nearestKey = this.getKeyAt(lastPoint.x, lastPoint.y);
            if (nearestKey !== this.activeKey) {
                this.activeKey = nearestKey;
            }
            this.addSwipeTrail(lastPoint);
        }
        
        this.render();
    }
    
    private addSwipeTrail(point: {x: number, y: number}) {
        // Create trail element with cloud effect
        const trail = document.createElement('div');
        trail.className = 'swipe-trail';
        
        // Convert from 360x215 to percentage of canvas element
        const rect = this.canvas.getBoundingClientRect();
        const percentX = (point.x / 360) * 100;
        const percentY = (point.y / 215) * 100;
        
        trail.style.left = `${percentX}%`;
        trail.style.top = `${percentY}%`;
        
        // Add to canvas parent container
        const container = this.canvas.parentElement;
        if (container) {
            trail.style.position = 'absolute';
            trail.style.pointerEvents = 'none';
            container.style.position = 'relative';
            container.appendChild(trail);
            
            // Remove after animation
            setTimeout(() => {
                if (trail.parentNode) {
                    trail.remove();
                }
            }, 1000);
        }
    }

    private drawTraceInternal() {
        if (this.tracePoints.length < 2) return;

        // Save context for effects
        this.ctx.save();

        // Add glow effect for the trace
        this.ctx.shadowColor = '#5865f2';  // Discord purple
        this.ctx.shadowBlur = 15;
        
        // Draw trace line with enhanced gradient
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Simple purple trace
        this.ctx.strokeStyle = '#5865f2';

        // Draw the main path
        this.ctx.beginPath();
        this.ctx.moveTo(this.tracePoints[0].x, this.tracePoints[0].y);
        
        // Use quadratic curves for smoother lines
        if (this.tracePoints.length === 2) {
            this.ctx.lineTo(this.tracePoints[1].x, this.tracePoints[1].y);
        } else {
            for (let i = 1; i < this.tracePoints.length - 1; i++) {
                const cp = this.tracePoints[i];
                const next = this.tracePoints[i + 1];
                const midX = (cp.x + next.x) / 2;
                const midY = (cp.y + next.y) / 2;
                this.ctx.quadraticCurveTo(cp.x, cp.y, midX, midY);
            }
            // Last segment
            const last = this.tracePoints[this.tracePoints.length - 1];
            this.ctx.lineTo(last.x, last.y);
        }
        
        this.ctx.stroke();
        this.ctx.restore();

        // Draw points with enhanced effects
        for (let i = 0; i < this.tracePoints.length; i++) {
            const point = this.tracePoints[i];
            const isEndpoint = i === 0 || i === this.tracePoints.length - 1;
            const radius = isEndpoint ? 6 : 2;
            
            if (isEndpoint) {
                // Draw glow for endpoints
                this.ctx.save();
                const glowGradient = this.ctx.createRadialGradient(
                    point.x, point.y, 0,
                    point.x, point.y, radius * 2
                );
                
                if (i === 0) {
                    glowGradient.addColorStop(0, 'rgba(34, 197, 94, 0.8)');
                    glowGradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
                } else {
                    glowGradient.addColorStop(0, 'rgba(168, 85, 247, 0.8)');
                    glowGradient.addColorStop(1, 'rgba(168, 85, 247, 0)');
                }
                
                this.ctx.fillStyle = glowGradient;
                this.ctx.beginPath();
                this.ctx.arc(point.x, point.y, radius * 2, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.restore();
            }
            
            // Draw the actual point
            this.ctx.fillStyle = i === 0 ? 
                                'rgb(34, 197, 94)' :     // green-500
                                (i === this.tracePoints.length - 1 ? 
                                 'rgb(168, 85, 247)' :    // purple-500
                                 'rgba(255, 255, 255, 0.6)');
            
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Add border to endpoints
            if (isEndpoint) {
                this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            }
        }
    }

    clearTrace() {
        this.tracePoints = [];
        this.activeKey = null;
        this.render();
    }

    getKeyAt(x: number, y: number): string | null {
        // x and y are already in keyboard space (360x215)
        // No conversion needed
        
        // Find nearest key within threshold
        const threshold = 20; // pixels in keyboard space
        let nearestKey: string | null = null;
        let minDistance = threshold;

        for (const key of this.keys) {
            const distance = Math.sqrt(
                Math.pow(x - key.x, 2) + 
                Math.pow(y - key.y, 2)
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