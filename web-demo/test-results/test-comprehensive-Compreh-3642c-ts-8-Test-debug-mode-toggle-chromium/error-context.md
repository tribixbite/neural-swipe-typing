# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - banner [ref=e2]:
    - generic [ref=e4]:
      - generic [ref=e5]:
        - heading "Neural Swipe Typing" [level=1] [ref=e6]
        - paragraph [ref=e7]: ONNX Character-Level Model • 70.1% Accuracy
      - button [ref=e8] [cursor=pointer]:
        - img [ref=e9] [cursor=pointer]
  - main [ref=e11]:
    - generic [ref=e13]:
      - generic [ref=e14]:
        - generic [ref=e15]: "Status:"
        - generic [ref=e16]: Ready
      - generic [ref=e17]:
        - button "Clear" [ref=e18] [cursor=pointer]
        - 'button "Debug: OFF" [ref=e19] [cursor=pointer]'
    - generic [ref=e20]:
      - heading "Swipe Path" [level=2] [ref=e21]
      - generic [ref=e22]:
        - generic: Touch the keyboard to start swiping...
    - application "Swipe keyboard - draw gestures to input text" [ref=e26]
    - generic [ref=e27]:
      - heading "Predictions" [level=2] [ref=e28]
      - paragraph [ref=e30]: Swipe on the keyboard to see predictions
    - generic [ref=e31]:
      - paragraph [ref=e32]: "Model: Character-level Transformer (ONNX) | Sequence Length: 150 | Beam Size: 5"
      - paragraph [ref=e33]: Swipe across letters to form words. The model predicts based on your gesture pattern.
```