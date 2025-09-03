import { KeyboardRenderer } from './keyboard';

export interface SwipePoint {
    x: number;
    y: number;
    t: number;
}

type EventCallback = (data: any) => void;

export class SwipeTracker {
    private canvas: HTMLCanvasElement;
    private keyboard: KeyboardRenderer;
    private isTracking: boolean = false;
    private points: SwipePoint[] = [];
    private startTime: number = 0;
    private callbacks: Map<string, EventCallback[]> = new Map();
    
    constructor(canvas: HTMLCanvasElement, keyboard: KeyboardRenderer) {
        this.canvas = canvas;
        this.keyboard = keyboard;
        this.setupEventListeners();
    }

    private setupEventListeners() {
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        this.canvas.addEventListener('touchcancel', this.handleTouchEnd.bind(this), { passive: false });
        
        // Mouse events for desktop
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseUp.bind(this));
        
        // Prevent default touch behaviors
        this.canvas.addEventListener('gesturestart', (e) => e.preventDefault());
        this.canvas.addEventListener('gesturechange', (e) => e.preventDefault());
        this.canvas.addEventListener('gestureend', (e) => e.preventDefault());
    }

    private getCanvasCoordinates(e: MouseEvent | TouchEvent): { x: number; y: number } {
        const canvas = this.canvas;
        const rect = canvas.getBoundingClientRect();

        // Determine the correct clientX/Y for mouse vs. touch events
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

        // Get the computed styles to account for border and padding
        const style = window.getComputedStyle(canvas);
        const borderLeftWidth = parseFloat(style.borderLeftWidth) || 0;
        const borderTopWidth = parseFloat(style.borderTopWidth) || 0;
        const paddingLeft = parseFloat(style.paddingLeft) || 0;
        const paddingTop = parseFloat(style.paddingTop) || 0;

        // This is the actual rendered size of the content-box in CSS pixels
        const cssContentWidth = parseFloat(style.width);
        const cssContentHeight = parseFloat(style.height);

        // 1. Calculate pointer position relative to the canvas element's border-box
        const xRelativeToElement = clientX - rect.left;
        const yRelativeToElement = clientY - rect.top;

        // 2. Adjust for border and padding to get the position relative to the content-box
        const xRelativeToContent = xRelativeToElement - borderLeftWidth - paddingLeft;
        const yRelativeToContent = yRelativeToElement - borderTopWidth - paddingTop;

        // 3. Calculate the scaling ratio
        // canvas.width is the internal resolution (360)
        // cssContentWidth is the displayed size of the content area in CSS pixels
        const scaleX = canvas.width / cssContentWidth;
        const scaleY = canvas.height / cssContentHeight;

        // 4. Apply the scaling factor to get the final coordinates
        const canvasX = xRelativeToContent * scaleX;
        const canvasY = yRelativeToContent * scaleY;

        return { x: canvasX, y: canvasY };
    }

    private handleTouchStart(e: TouchEvent) {
        e.preventDefault();
        if (e.touches.length !== 1) return;
        
        const { x, y } = this.getCanvasCoordinates(e);
        this.startSwipe(x, y);
    }

    private handleTouchMove(e: TouchEvent) {
        e.preventDefault();
        if (!this.isTracking || e.touches.length !== 1) return;
        
        const { x, y } = this.getCanvasCoordinates(e);
        this.addPoint(x, y);
    }

    private handleTouchEnd(e: TouchEvent) {
        e.preventDefault();
        if (!this.isTracking) return;
        
        this.endSwipe();
    }

    private handleMouseDown(e: MouseEvent) {
        const { x, y } = this.getCanvasCoordinates(e);
        this.startSwipe(x, y);
    }

    private handleMouseMove(e: MouseEvent) {
        if (!this.isTracking) return;
        
        const { x, y } = this.getCanvasCoordinates(e);
        this.addPoint(x, y);
    }

    private handleMouseUp(e: MouseEvent) {
        if (!this.isTracking) return;
        
        this.endSwipe();
    }

    private startSwipe(x: number, y: number) {
        this.isTracking = true;
        this.points = [];
        this.startTime = Date.now();
        
        // x and y are already in keyboard coordinates (360x215 space)
        this.points.push({
            x: x,
            y: y,
            t: 0
        });
        
        this.emit('swipeStart', this.points);
    }

    private addPoint(x: number, y: number) {
        if (!this.isTracking) return;
        
        // x and y are already in keyboard coordinates (360x215 space)
        const elapsed = Date.now() - this.startTime;
        
        // Filter out points that are too close (noise reduction)
        if (this.points.length > 0) {
            const lastPoint = this.points[this.points.length - 1];
            const distance = Math.sqrt(
                Math.pow(x - lastPoint.x, 2) + 
                Math.pow(y - lastPoint.y, 2)
            );
            
            // Skip if too close (less than 2 pixels in keyboard space)
            if (distance < 2) return;
        }
        
        this.points.push({
            x: x,
            y: y,
            t: elapsed
        });
        
        this.emit('swipeMove', this.points);
    }

    private endSwipe() {
        if (!this.isTracking) return;
        
        this.isTracking = false;
        
        // Smooth the trajectory if needed
        const smoothedPoints = this.smoothTrajectory(this.points);
        
        this.emit('swipeEnd', smoothedPoints);
        this.points = [];
    }

    private smoothTrajectory(points: SwipePoint[]): SwipePoint[] {
        if (points.length < 3) return points;
        
        // Simple moving average smoothing
        const smoothed: SwipePoint[] = [];
        const windowSize = 3;
        
        for (let i = 0; i < points.length; i++) {
            if (i === 0 || i === points.length - 1) {
                // Keep first and last points unchanged
                smoothed.push(points[i]);
            } else {
                // Average with neighbors
                let sumX = 0, sumY = 0, count = 0;
                
                for (let j = Math.max(0, i - Math.floor(windowSize / 2)); 
                     j <= Math.min(points.length - 1, i + Math.floor(windowSize / 2)); 
                     j++) {
                    sumX += points[j].x;
                    sumY += points[j].y;
                    count++;
                }
                
                smoothed.push({
                    x: sumX / count,
                    y: sumY / count,
                    t: points[i].t
                });
            }
        }
        
        return smoothed;
    }

    // Event emitter methods
    on(event: string, callback: EventCallback) {
        if (!this.callbacks.has(event)) {
            this.callbacks.set(event, []);
        }
        this.callbacks.get(event)!.push(callback);
    }

    private emit(event: string, data: any) {
        const callbacks = this.callbacks.get(event);
        if (callbacks) {
            callbacks.forEach(cb => cb(data));
        }
    }
}