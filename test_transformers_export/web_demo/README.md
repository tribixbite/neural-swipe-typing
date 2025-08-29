# Neural Swipe Typing Web Demo

This demo shows how to use the converted neural swipe typing model with transformer.js.

## Setup

1. Ensure the ONNX model was successfully converted
2. Serve this directory with a web server (due to CORS restrictions)
3. Open `index.html` in a browser

## Usage

```bash
# Serve with Python
python -m http.server 8000

# Or with Node.js
npx http-server .

# Then open http://localhost:8000
```

## Model Details

- Architecture: v3_nearest_and_traj_transformer_bigger
- Input: Trajectory features (x, y, velocity, acceleration) + keyboard features
- Output: Character probabilities for word prediction

## Integration Notes

The actual integration would require:
1. Virtual keyboard component with touch tracking
2. Real-time trajectory feature extraction
3. Keyboard-aware distance calculations
4. Beam search for word generation
