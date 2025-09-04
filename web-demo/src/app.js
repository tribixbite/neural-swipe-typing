// src/keyboard.ts
class KeyboardRenderer {
  canvas;
  ctx;
  width;
  height;
  scale = 1;
  keyboardWidth = 360;
  keyboardHeight = 215;
  offsetX = 0;
  offsetY = 0;
  keys = [];
  tracePoints = [];
  activeKey = null;
  KEYBOARD_LAYOUT = {
    q: { x: 18, y: 53 },
    w: { x: 54, y: 53 },
    e: { x: 90, y: 53 },
    r: { x: 126, y: 53 },
    t: { x: 162, y: 53 },
    y: { x: 198, y: 53 },
    u: { x: 234, y: 53 },
    i: { x: 270, y: 53 },
    o: { x: 306, y: 53 },
    p: { x: 342, y: 53 },
    a: { x: 36, y: 107 },
    s: { x: 72, y: 107 },
    d: { x: 108, y: 107 },
    f: { x: 144, y: 107 },
    g: { x: 180, y: 107 },
    h: { x: 216, y: 107 },
    j: { x: 252, y: 107 },
    k: { x: 288, y: 107 },
    l: { x: 324, y: 107 },
    z: { x: 72, y: 161 },
    x: { x: 108, y: 161 },
    c: { x: 144, y: 161 },
    v: { x: 180, y: 161 },
    b: { x: 216, y: 161 },
    n: { x: 252, y: 161 },
    m: { x: 288, y: 161 }
  };
  constructor(canvas, ctx) {
    this.canvas = canvas;
    this.ctx = ctx;
    this.width = canvas.width;
    this.height = canvas.height;
    console.log("KeyboardRenderer initialized with:", {
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      expectedWidth: 360,
      expectedHeight: 215
    });
    this.updateScale();
    this.initializeKeys();
  }
  updateDimensions(width, height) {
    console.log("updateDimensions called with:", width, height);
    this.width = width;
    this.height = height;
    this.updateScale();
  }
  updateScale() {
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.keyboardWidth = 360;
    this.keyboardHeight = 215;
  }
  initializeKeys() {
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
    this.ctx.fillStyle = "#1a1a2e";
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.drawKeys();
    if (this.tracePoints.length > 0) {
      this.drawTraceInternal();
    }
  }
  drawKeys() {
    const keySize = 32;
    const fontSize = 20;
    for (const key of this.keys) {
      const x = key.x;
      const y = key.y;
      const isActive = this.activeKey === key.char;
      const currentKeySize = isActive ? keySize * 1.2 : keySize;
      const currentFontSize = isActive ? fontSize * 1.15 : fontSize;
      this.ctx.save();
      this.ctx.shadowColor = "rgba(0, 0, 0, 0.3)";
      this.ctx.shadowBlur = 4;
      this.ctx.shadowOffsetX = 0;
      this.ctx.shadowOffsetY = 2;
      this.ctx.fillStyle = isActive ? "#5865f2" : "#2d2d44";
      this.roundRect(x - currentKeySize / 2, y - currentKeySize / 2, currentKeySize, currentKeySize, 6);
      this.ctx.fill();
      this.ctx.restore();
      this.ctx.save();
      this.ctx.fillStyle = isActive ? "#ffffff" : "rgba(255, 255, 255, 0.95)";
      this.ctx.font = `400 ${currentFontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(key.char.toUpperCase(), x, y);
      this.ctx.restore();
    }
  }
  drawTrace(points) {
    this.tracePoints = points.map((p) => ({
      x: p.x,
      y: p.y
    }));
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
  addSwipeTrail(point) {
    const trail = document.createElement("div");
    trail.className = "swipe-trail";
    const rect = this.canvas.getBoundingClientRect();
    const percentX = point.x / 360 * 100;
    const percentY = point.y / 215 * 100;
    trail.style.left = `${percentX}%`;
    trail.style.top = `${percentY}%`;
    const container = this.canvas.parentElement;
    if (container) {
      trail.style.position = "absolute";
      trail.style.pointerEvents = "none";
      container.style.position = "relative";
      container.appendChild(trail);
      setTimeout(() => {
        if (trail.parentNode) {
          trail.remove();
        }
      }, 1000);
    }
  }
  drawTraceInternal() {
    if (this.tracePoints.length < 2)
      return;
    this.ctx.save();
    this.ctx.shadowColor = "#5865f2";
    this.ctx.shadowBlur = 15;
    this.ctx.lineWidth = 3;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.strokeStyle = "#5865f2";
    this.ctx.beginPath();
    this.ctx.moveTo(this.tracePoints[0].x, this.tracePoints[0].y);
    if (this.tracePoints.length === 2) {
      this.ctx.lineTo(this.tracePoints[1].x, this.tracePoints[1].y);
    } else {
      for (let i = 1;i < this.tracePoints.length - 1; i++) {
        const cp = this.tracePoints[i];
        const next = this.tracePoints[i + 1];
        const midX = (cp.x + next.x) / 2;
        const midY = (cp.y + next.y) / 2;
        this.ctx.quadraticCurveTo(cp.x, cp.y, midX, midY);
      }
      const last = this.tracePoints[this.tracePoints.length - 1];
      this.ctx.lineTo(last.x, last.y);
    }
    this.ctx.stroke();
    this.ctx.restore();
    for (let i = 0;i < this.tracePoints.length; i++) {
      const point = this.tracePoints[i];
      const isEndpoint = i === 0 || i === this.tracePoints.length - 1;
      const radius = isEndpoint ? 6 : 2;
      if (isEndpoint) {
        this.ctx.save();
        const glowGradient = this.ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius * 2);
        if (i === 0) {
          glowGradient.addColorStop(0, "rgba(34, 197, 94, 0.8)");
          glowGradient.addColorStop(1, "rgba(34, 197, 94, 0)");
        } else {
          glowGradient.addColorStop(0, "rgba(168, 85, 247, 0.8)");
          glowGradient.addColorStop(1, "rgba(168, 85, 247, 0)");
        }
        this.ctx.fillStyle = glowGradient;
        this.ctx.beginPath();
        this.ctx.arc(point.x, point.y, radius * 2, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.restore();
      }
      this.ctx.fillStyle = i === 0 ? "rgb(34, 197, 94)" : i === this.tracePoints.length - 1 ? "rgb(168, 85, 247)" : "rgba(255, 255, 255, 0.6)";
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      this.ctx.fill();
      if (isEndpoint) {
        this.ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
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
  getKeyAt(x, y) {
    const threshold = 20;
    let nearestKey = null;
    let minDistance = threshold;
    for (const key of this.keys) {
      const distance = Math.sqrt(Math.pow(x - key.x, 2) + Math.pow(y - key.y, 2));
      if (distance < minDistance) {
        minDistance = distance;
        nearestKey = key.char;
      }
    }
    return nearestKey;
  }
  canvasToKeyboard(canvasX, canvasY) {
    return {
      x: canvasX / this.scale,
      y: canvasY / this.scale
    };
  }
  getKeyboardLayout() {
    return this.KEYBOARD_LAYOUT;
  }
  roundRect(x, y, width, height, radius) {
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

// src/predictor.ts
class SwipePredictor {
  encoderSession;
  decoderSession;
  tokenizer;
  keyboardLayout;
  PAD_IDX = 0;
  UNK_IDX = 1;
  SOS_IDX = 2;
  EOS_IDX = 3;
  constructor() {}
  async loadEncoder(url) {
    console.log("Loading encoder model...");
    const options = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };
    this.encoderSession = await ort.InferenceSession.create(url, options);
    console.log("Encoder loaded");
  }
  async loadDecoder(url) {
    console.log("Loading decoder model...");
    const options = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };
    this.decoderSession = await ort.InferenceSession.create(url, options);
    console.log("Decoder loaded");
  }
  async loadSingleModel(url) {
    console.log("Loading combined model...");
    const options = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };
    this.encoderSession = await ort.InferenceSession.create(url, options);
    this.decoderSession = this.encoderSession;
    console.log("Combined model loaded");
  }
  async loadTokenizer(url) {
    console.log("Loading tokenizer...");
    const response = await fetch(url);
    const config = await response.json();
    this.tokenizer = {
      charToIdx: config.char_to_idx,
      idxToChar: config.idx_to_char
    };
    this.keyboardLayout = config.keyboard_layout || this.getDefaultKeyboardLayout();
    console.log("Tokenizer loaded");
  }
  getDefaultKeyboardLayout() {
    return {
      q: { x: 18, y: 53 },
      w: { x: 54, y: 53 },
      e: { x: 90, y: 53 },
      r: { x: 126, y: 53 },
      t: { x: 162, y: 53 },
      y: { x: 198, y: 53 },
      u: { x: 234, y: 53 },
      i: { x: 270, y: 53 },
      o: { x: 306, y: 53 },
      p: { x: 342, y: 53 },
      a: { x: 36, y: 107 },
      s: { x: 72, y: 107 },
      d: { x: 108, y: 107 },
      f: { x: 144, y: 107 },
      g: { x: 180, y: 107 },
      h: { x: 216, y: 107 },
      j: { x: 252, y: 107 },
      k: { x: 288, y: 107 },
      l: { x: 324, y: 107 },
      z: { x: 72, y: 161 },
      x: { x: 108, y: 161 },
      c: { x: 144, y: 161 },
      v: { x: 180, y: 161 },
      b: { x: 216, y: 161 },
      n: { x: 252, y: 161 },
      m: { x: 288, y: 161 }
    };
  }
  padOrTruncatePoints(points, targetLength) {
    if (points.length >= targetLength) {
      return points.slice(0, targetLength);
    }
    const padded = [...points];
    const lastPoint = points[points.length - 1] || { x: 0, y: 0, t: 0 };
    while (padded.length < targetLength) {
      padded.push({ ...lastPoint });
    }
    return padded;
  }
  async predict(swipePoints, topK = 5) {
    if (!this.encoderSession || !this.decoderSession || !this.tokenizer) {
      throw new Error("Models not loaded");
    }
    console.log("Starting prediction with", swipePoints.length, "points");
    const FIXED_SEQ_LENGTH = 150;
    const paddedPoints = this.padOrTruncatePoints(swipePoints, FIXED_SEQ_LENGTH);
    const features = this.extractFeatures(paddedPoints);
    const nearestKeys = this.findNearestKeys(paddedPoints);
    const srcMask = new Uint8Array(FIXED_SEQ_LENGTH);
    for (let i = swipePoints.length;i < FIXED_SEQ_LENGTH; i++) {
      srcMask[i] = 1;
    }
    console.log("Features shape:", [1, FIXED_SEQ_LENGTH, 6]);
    console.log("Nearest keys shape:", [1, FIXED_SEQ_LENGTH]);
    console.log("Src mask shape:", [1, FIXED_SEQ_LENGTH]);
    const encoderInputs = {
      trajectory_features: new ort.Tensor("float32", features, [1, FIXED_SEQ_LENGTH, 6]),
      nearest_keys: new ort.Tensor("int64", nearestKeys, [1, FIXED_SEQ_LENGTH]),
      src_mask: new ort.Tensor("bool", srcMask, [1, FIXED_SEQ_LENGTH])
    };
    console.log("Running encoder...");
    try {
      const encoderOutputs = await this.encoderSession.run(encoderInputs);
      console.log("Encoder outputs:", Object.keys(encoderOutputs));
      const memory = encoderOutputs.encoder_output;
      console.log("Memory tensor shape:", memory.dims);
      console.log("Starting beam search decode...");
      const predictions = await this.beamSearchDecode(memory, topK);
      console.log("Predictions:", predictions);
      return predictions;
    } catch (error) {
      console.error("Encoder/Decoder error:", error);
      console.error("Error message:", error?.message);
      console.error("Error stack:", error?.stack);
      throw error;
    }
  }
  extractFeatures(points) {
    const features = [];
    for (let i = 0;i < points.length; i++) {
      const p = points[i];
      const x = p.x / 360;
      const y = p.y / 215;
      let vx = 0, vy = 0;
      if (i > 0) {
        const prev = points[i - 1];
        const dt = Math.max(p.t - prev.t, 1);
        vx = (p.x - prev.x) / dt;
        vy = (p.y - prev.y) / dt;
      }
      let ax = 0, ay = 0;
      if (i > 1) {
        const prev = points[i - 1];
        const prev2 = points[i - 2];
        const dt1 = Math.max(p.t - prev.t, 1);
        const dt2 = Math.max(prev.t - prev2.t, 1);
        const vxPrev = (prev.x - prev2.x) / dt2;
        const vyPrev = (prev.y - prev2.y) / dt2;
        ax = (vx - vxPrev) / dt1;
        ay = (vy - vyPrev) / dt1;
      }
      vx = Math.max(-1, Math.min(1, vx / 1000));
      vy = Math.max(-1, Math.min(1, vy / 1000));
      ax = Math.max(-1, Math.min(1, ax / 500));
      ay = Math.max(-1, Math.min(1, ay / 500));
      features.push(x, y, vx, vy, ax, ay);
    }
    return new Float32Array(features);
  }
  findNearestKeys(points) {
    const nearestKeys = [];
    for (const point of points) {
      let nearestKey = this.UNK_IDX;
      let minDist = Infinity;
      for (const [key, pos] of Object.entries(this.keyboardLayout)) {
        const dist = Math.sqrt(Math.pow(point.x - pos.x, 2) + Math.pow(point.y - pos.y, 2));
        if (dist < minDist) {
          minDist = dist;
          nearestKey = this.tokenizer.charToIdx[key] || this.UNK_IDX;
        }
      }
      nearestKeys.push(BigInt(nearestKey));
    }
    return new BigInt64Array(nearestKeys);
  }
  async beamSearchDecode(memory, beamSize) {
    const DECODER_SEQ_LENGTH = 20;
    const maxGeneratedTokens = 15;
    let beams = [{
      tokens: [this.SOS_IDX],
      score: 0,
      finished: false
    }];
    for (let step = 0;step < maxGeneratedTokens; step++) {
      const allCandidates = [];
      for (const beam of beams) {
        if (beam.finished) {
          allCandidates.push(beam);
          continue;
        }
        const paddedTokens = new BigInt64Array(DECODER_SEQ_LENGTH);
        for (let i = 0;i < beam.tokens.length && i < DECODER_SEQ_LENGTH; i++) {
          paddedTokens[i] = BigInt(beam.tokens[i]);
        }
        for (let i = beam.tokens.length;i < DECODER_SEQ_LENGTH; i++) {
          paddedTokens[i] = BigInt(this.PAD_IDX);
        }
        const tgtMask = new Uint8Array(DECODER_SEQ_LENGTH);
        for (let i = beam.tokens.length;i < DECODER_SEQ_LENGTH; i++) {
          tgtMask[i] = 1;
        }
        const srcMask = new Uint8Array(memory.dims[1]).fill(0);
        const decoderInputs = {
          memory,
          target_tokens: new ort.Tensor("int64", paddedTokens, [1, DECODER_SEQ_LENGTH]),
          target_mask: new ort.Tensor("bool", tgtMask, [1, DECODER_SEQ_LENGTH]),
          src_mask: new ort.Tensor("bool", srcMask, [1, memory.dims[1]])
        };
        let decoderOutputs;
        try {
          decoderOutputs = await this.decoderSession.run(decoderInputs);
        } catch (decodeError) {
          console.error("Decoder run failed:", decodeError);
          console.error("Error details:", decodeError?.message);
          throw decodeError;
        }
        const logits = decoderOutputs.logits;
        const logitsData = logits.data;
        const vocabSize = 30;
        const tokenPosition = Math.min(beam.tokens.length - 1, DECODER_SEQ_LENGTH - 1);
        const startIdx = tokenPosition * vocabSize;
        const endIdx = startIdx + vocabSize;
        const relevantLogits = logitsData.slice(startIdx, endIdx);
        const probs = this.softmax(relevantLogits);
        const topK = this.getTopK(probs, Math.min(beamSize, vocabSize));
        for (const { idx, prob } of topK) {
          const newTokens = [...beam.tokens, idx];
          const finished = idx === this.EOS_IDX;
          if (step < 3) {
            const char = this.tokenizer?.idxToChar[idx] || `<${idx}>`;
            console.log(`Step ${step}, beam: adding token ${idx} = "${char}", prob=${prob.toFixed(3)}`);
          }
          allCandidates.push({
            tokens: newTokens,
            score: beam.score + Math.log(prob),
            finished
          });
        }
      }
      beams = allCandidates.sort((a, b) => b.score - a.score).slice(0, beamSize);
      if (beams.every((b) => b.finished))
        break;
    }
    const predictions = beams.map((beam) => {
      const word = this.decodeTokens(beam.tokens);
      console.log(`Beam tokens: [${beam.tokens.join(",")}] -> "${word}"`);
      return {
        word,
        score: Math.exp(beam.score / beam.tokens.length)
      };
    });
    return predictions.filter((p) => p.word.length > 0);
  }
  softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expScores = Array.from(logits).map((l) => Math.exp(l - maxLogit));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return new Float32Array(expScores.map((e) => e / sumExp));
  }
  getTopK(probs, k) {
    const indexed = Array.from(probs).map((prob, idx) => ({ idx, prob }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
  }
  decodeTokens(tokens) {
    let word = "";
    for (const token of tokens) {
      if (token === this.EOS_IDX)
        break;
      if (token === this.SOS_IDX || token === this.PAD_IDX)
        continue;
      const char = this.tokenizer.idxToChar[token];
      if (char && !char.startsWith("<")) {
        word += char;
      }
    }
    return word;
  }
}

// src/swipe-tracker.ts
class SwipeTracker {
  canvas;
  keyboard;
  isTracking = false;
  points = [];
  startTime = 0;
  callbacks = new Map;
  constructor(canvas, keyboard) {
    this.canvas = canvas;
    this.keyboard = keyboard;
    this.setupEventListeners();
  }
  setupEventListeners() {
    this.canvas.addEventListener("touchstart", this.handleTouchStart.bind(this), { passive: false });
    this.canvas.addEventListener("touchmove", this.handleTouchMove.bind(this), { passive: false });
    this.canvas.addEventListener("touchend", this.handleTouchEnd.bind(this), { passive: false });
    this.canvas.addEventListener("touchcancel", this.handleTouchEnd.bind(this), { passive: false });
    this.canvas.addEventListener("mousedown", this.handleMouseDown.bind(this));
    this.canvas.addEventListener("mousemove", this.handleMouseMove.bind(this));
    this.canvas.addEventListener("mouseup", this.handleMouseUp.bind(this));
    this.canvas.addEventListener("mouseleave", this.handleMouseUp.bind(this));
    this.canvas.addEventListener("gesturestart", (e) => e.preventDefault());
    this.canvas.addEventListener("gesturechange", (e) => e.preventDefault());
    this.canvas.addEventListener("gestureend", (e) => e.preventDefault());
  }
  getCanvasCoordinates(e) {
    const rect = this.canvas.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
    const normalizedX = (clientX - rect.left) / rect.width;
    const normalizedY = (clientY - rect.top) / rect.height;
    const x = normalizedX * 360;
    const y = normalizedY * 215;
    return { x, y };
  }
  handleTouchStart(e) {
    e.preventDefault();
    if (e.touches.length !== 1)
      return;
    const { x, y } = this.getCanvasCoordinates(e);
    this.startSwipe(x, y);
  }
  handleTouchMove(e) {
    e.preventDefault();
    if (!this.isTracking || e.touches.length !== 1)
      return;
    const { x, y } = this.getCanvasCoordinates(e);
    this.addPoint(x, y);
  }
  handleTouchEnd(e) {
    e.preventDefault();
    if (!this.isTracking)
      return;
    this.endSwipe();
  }
  handleMouseDown(e) {
    const { x, y } = this.getCanvasCoordinates(e);
    this.startSwipe(x, y);
  }
  handleMouseMove(e) {
    if (!this.isTracking)
      return;
    const { x, y } = this.getCanvasCoordinates(e);
    this.addPoint(x, y);
  }
  handleMouseUp(e) {
    if (!this.isTracking)
      return;
    this.endSwipe();
  }
  startSwipe(x, y) {
    this.isTracking = true;
    this.points = [];
    this.startTime = Date.now();
    this.points.push({
      x,
      y,
      t: 0
    });
    this.emit("swipeStart", this.points);
  }
  addPoint(x, y) {
    if (!this.isTracking)
      return;
    const elapsed = Date.now() - this.startTime;
    if (this.points.length > 0) {
      const lastPoint = this.points[this.points.length - 1];
      const distance = Math.sqrt(Math.pow(x - lastPoint.x, 2) + Math.pow(y - lastPoint.y, 2));
      if (distance < 2)
        return;
    }
    this.points.push({
      x,
      y,
      t: elapsed
    });
    this.emit("swipeMove", this.points);
  }
  endSwipe() {
    if (!this.isTracking)
      return;
    this.isTracking = false;
    const smoothedPoints = this.smoothTrajectory(this.points);
    this.emit("swipeEnd", smoothedPoints);
    this.points = [];
  }
  smoothTrajectory(points) {
    if (points.length < 3)
      return points;
    const smoothed = [];
    const windowSize = 3;
    for (let i = 0;i < points.length; i++) {
      if (i === 0 || i === points.length - 1) {
        smoothed.push(points[i]);
      } else {
        let sumX = 0, sumY = 0, count = 0;
        for (let j = Math.max(0, i - Math.floor(windowSize / 2));j <= Math.min(points.length - 1, i + Math.floor(windowSize / 2)); j++) {
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
  on(event, callback) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event).push(callback);
  }
  emit(event, data) {
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      callbacks.forEach((cb) => cb(data));
    }
  }
}

// src/app.ts
class SwipeTypingApp {
  canvas;
  ctx;
  keyboard;
  predictor;
  swipeTracker;
  predictionsEl;
  loadingEl;
  loadingProgressEl;
  swipeCharsEl;
  statusEl;
  debugMode = false;
  constructor() {
    this.canvas = document.getElementById("keyboard-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.predictionsEl = document.getElementById("predictions");
    this.loadingEl = document.getElementById("loading");
    this.loadingProgressEl = document.getElementById("loading-progress");
    this.swipeCharsEl = document.getElementById("swipe-chars");
    this.statusEl = document.getElementById("status");
    this.keyboard = new KeyboardRenderer(this.canvas, this.ctx);
    this.predictor = new SwipePredictor;
    this.swipeTracker = new SwipeTracker(this.canvas, this.keyboard);
    this.init();
  }
  async init() {
    try {
      this.setupCanvas();
      window.addEventListener("resize", () => this.setupCanvas());
      await this.loadModels();
      this.setupEventHandlers();
      this.keyboard.render();
      this.loadingEl.style.display = "none";
    } catch (error) {
      console.error("Failed to initialize app:", error);
      this.loadingProgressEl.textContent = "Failed to load models. Please refresh.";
      this.loadingProgressEl.style.color = "#ff5252";
    }
  }
  setupCanvas() {
    this.canvas.width = 360;
    this.canvas.height = 215;
    this.keyboard.updateDimensions(360, 215);
    this.keyboard.render();
  }
  async loadModels() {
    const basePath = window.location.hostname === "localhost" ? "" : ".";
    this.loadingProgressEl.textContent = "Loading encoder model...";
    await this.predictor.loadEncoder(`${basePath}/models/swipe_model_character_quant.onnx`);
    this.loadingProgressEl.textContent = "Loading decoder model...";
    await this.predictor.loadDecoder(`${basePath}/models/swipe_decoder_character_quant.onnx`);
    this.loadingProgressEl.textContent = "Loading tokenizer...";
    await this.predictor.loadTokenizer(`${basePath}/models/tokenizer_config.json`);
    this.loadingProgressEl.textContent = "Models loaded successfully!";
  }
  setupEventHandlers() {
    let loggedKeys = new Set;
    let swipedChars = [];
    this.swipeTracker.on("swipeStart", () => {
      this.keyboard.clearTrace();
      this.clearPredictions();
      loggedKeys.clear();
      swipedChars = [];
      this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.7); font-size: 16px;">Swiping...</span>';
      this.updateStatus("Swiping");
    });
    this.swipeTracker.on("swipeMove", (points) => {
      this.keyboard.drawTrace(points);
      if (points.length > 0) {
        const lastPoint = points[points.length - 1];
        const key = this.keyboard.getKeyAt(lastPoint.x, lastPoint.y);
        if (key && !loggedKeys.has(key)) {
          console.log(`Key: ${key.toUpperCase()}`);
          loggedKeys.add(key);
          swipedChars.push(key.toUpperCase());
          this.swipeCharsEl.innerHTML = swipedChars.join(" → ");
        }
      }
    });
    this.swipeTracker.on("swipeEnd", async (points) => {
      if (points.length < 3) {
        this.keyboard.clearTrace();
        this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.7); font-size: 16px;">Too short - try again</span>';
        this.updateStatus("Ready");
        return;
      }
      this.showLoadingPredictions();
      this.updateStatus("Processing");
      try {
        const predictions = await this.predictor.predict(points, 5);
        this.showPredictions(predictions);
        this.updateStatus("Ready");
        if (this.debugMode) {
          console.log("Swipe points:", points);
          console.log("Predictions:", predictions);
        }
      } catch (error) {
        console.error("Prediction error:", error);
        console.error("Error stack:", error?.stack);
        console.error("Error message:", error?.message);
        this.showError();
        this.updateStatus("Error");
      }
      setTimeout(() => {
        this.keyboard.clearTrace();
      }, 1000);
    });
    document.getElementById("clear-btn")?.addEventListener("click", () => {
      this.keyboard.clearTrace();
      this.clearPredictions();
      this.swipeCharsEl.innerHTML = '<span style="color: rgba(255,255,255,0.6); font-size: 14px;">Touch the keyboard to start swiping...</span>';
      this.updateStatus("Ready");
    });
    document.getElementById("debug-btn")?.addEventListener("click", () => {
      this.debugMode = !this.debugMode;
      const btn = document.getElementById("debug-btn");
      btn.textContent = this.debugMode ? "Debug: ON" : "Debug: OFF";
    });
    this.predictionsEl.addEventListener("click", (e) => {
      const target = e.target;
      if (target.tagName === "BUTTON") {
        const word = target.textContent?.trim();
        if (word) {
          this.selectWord(word);
        }
      }
    });
  }
  clearPredictions() {
    this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.6); text-align: center; width: 100%; font-size: 14px;">Swipe on the keyboard to see predictions</p>';
  }
  showLoadingPredictions() {
    this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.8); text-align: center; width: 100%; animation: pulse 2s infinite;">Processing...</p>';
  }
  showPredictions(predictions) {
    if (predictions.length === 0) {
      this.predictionsEl.innerHTML = '<p style="color: rgba(255,255,255,0.6); text-align: center; width: 100%;">No predictions found</p>';
      return;
    }
    this.predictionsEl.innerHTML = predictions.map((pred, i) => `
                <button style="
                    background: ${i === 0 ? "rgba(255, 255, 255, 0.25)" : "rgba(255, 255, 255, 0.15)"};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    font-weight: ${i === 0 ? "600" : "400"};
                    font-family: monospace;"
                    onmouseover="this.style.background='rgba(255,255,255,0.3)'; this.style.transform='translateY(-2px)';"
                    onmouseout="this.style.background='${i === 0 ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.15)"}'; this.style.transform='translateY(0)';"
                    data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </button>
            `).join("");
  }
  showError() {
    this.predictionsEl.innerHTML = '<p style="color: #ff6b6b; text-align: center; width: 100%;">Error processing swipe</p>';
  }
  updateStatus(text) {
    this.statusEl.textContent = `Status: ${text}`;
  }
  selectWord(word) {
    console.log("Selected word:", word);
    const predictions = this.predictionsEl.querySelectorAll(".prediction");
    predictions.forEach((pred) => {
      if (pred.textContent === word) {
        pred.classList.add("selected");
        setTimeout(() => pred.classList.remove("selected"), 300);
      }
    });
  }
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => new SwipeTypingApp);
} else {
  new SwipeTypingApp;
}

//# debugId=0BB1E780EB7CE78164756E2164756E21
