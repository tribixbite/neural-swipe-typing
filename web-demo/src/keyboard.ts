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
        // We need to maintain the keyboard layout proportions
        // Use the minimum scale to ensure all keys fit
        const scaleX = this.width / 360;
        const scaleY = this.height / 215;
        
        // Use uniform scaling based on width to maintain key positions
        // This ensures coordinate mapping stays consistent
        this.scale = scaleX;
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
        // Clear canvas with gradient background
        const isDark = document.documentElement.classList.contains('dark');
        
        // Create gradient background
        const bgGradient = this.ctx.createLinearGradient(0, 0, this.width, this.height);
        if (isDark) {
            bgGradient.addColorStop(0, '#1e293b');  // slate-800
            bgGradient.addColorStop(1, '#0f172a');  // slate-900
        } else {
            bgGradient.addColorStop(0, '#e2e8f0');  // slate-200
            bgGradient.addColorStop(1, '#cbd5e1');  // slate-300
        }
        this.ctx.fillStyle = bgGradient;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw keys
        this.drawKeys();
        
        // Draw trace if exists
        if (this.tracePoints.length > 0) {
            this.drawTraceInternal();
        }
    }

    private drawKeys() {
        const keySize = 30 * this.scale;  // Balanced size to prevent overlap
        const fontSize = 16 * this.scale;  // Proportional font size
        const isDark = document.documentElement.classList.contains('dark');

        for (const key of this.keys) {
            const x = key.x * this.scale;
            const y = key.y * this.scale;

            // Save context for shadows
            this.ctx.save();
            
            // Add shadow for depth effect
            this.ctx.shadowColor = isDark ? 'rgba(0, 0, 0, 0.5)' : 'rgba(0, 0, 0, 0.2)';
            this.ctx.shadowBlur = 4 * this.scale;
            this.ctx.shadowOffsetX = 0;
            this.ctx.shadowOffsetY = 2 * this.scale;

            // Key background with enhanced gradient
            const gradient = this.ctx.createLinearGradient(
                x - keySize / 2, y - keySize / 2,
                x + keySize / 2, y + keySize / 2
            );
            
            if (isDark) {
                gradient.addColorStop(0, '#475569');  // slate-600
                gradient.addColorStop(0.5, '#334155');  // slate-700
                gradient.addColorStop(1, '#1e293b');  // slate-800
            } else {
                gradient.addColorStop(0, '#ffffff');
                gradient.addColorStop(0.5, '#f8fafc');  // slate-50
                gradient.addColorStop(1, '#f1f5f9');  // slate-100
            }
            
            this.ctx.fillStyle = gradient;
            this.roundRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize,
                5 * this.scale
            );
            this.ctx.fill();

            // Restore context to remove shadow for border
            this.ctx.restore();

            // Key border with subtle gradient
            const borderGradient = this.ctx.createLinearGradient(
                x - keySize / 2, y - keySize / 2,
                x + keySize / 2, y + keySize / 2
            );
            if (isDark) {
                borderGradient.addColorStop(0, '#64748b');  // slate-500
                borderGradient.addColorStop(1, '#475569');  // slate-600
            } else {
                borderGradient.addColorStop(0, '#cbd5e1');  // slate-300
                borderGradient.addColorStop(1, '#94a3b8');  // slate-400
            }
            this.ctx.strokeStyle = borderGradient;
            this.ctx.lineWidth = 1;
            this.roundRect(
                x - keySize / 2,
                y - keySize / 2,
                keySize,
                keySize,
                5 * this.scale
            );
            this.ctx.stroke();

            // Key text with subtle shadow
            this.ctx.save();
            this.ctx.shadowColor = isDark ? 'rgba(0, 0, 0, 0.5)' : 'rgba(0, 0, 0, 0.3)';
            this.ctx.shadowBlur = 1;
            this.ctx.shadowOffsetY = 1;
            
            this.ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';  // slate-50 : slate-900
            this.ctx.font = `bold ${fontSize}px 'JetBrains Mono', 'SF Mono', 'Consolas', monospace`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(key.char.toUpperCase(), x, y);
            
            this.ctx.restore();
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

        // Save context for effects
        this.ctx.save();

        // Add glow effect for the trace
        this.ctx.shadowColor = 'rgba(99, 102, 241, 0.6)';  // indigo-500
        this.ctx.shadowBlur = 10 * this.scale;
        
        // Draw trace line with enhanced gradient
        this.ctx.lineWidth = 4 * this.scale;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Create multi-color gradient effect
        const gradient = this.ctx.createLinearGradient(
            this.tracePoints[0].x,
            this.tracePoints[0].y,
            this.tracePoints[this.tracePoints.length - 1].x,
            this.tracePoints[this.tracePoints.length - 1].y
        );
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.9)');   // green-500
        gradient.addColorStop(0.5, 'rgba(99, 102, 241, 0.9)'); // indigo-500
        gradient.addColorStop(1, 'rgba(168, 85, 247, 0.9)');   // purple-500
        this.ctx.strokeStyle = gradient;

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
            const radius = isEndpoint ? 6 * this.scale : 2 * this.scale;
            
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