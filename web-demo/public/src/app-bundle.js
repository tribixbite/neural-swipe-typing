// src/keyboard.ts
class KeyboardRenderer {
  canvas;
  ctx;
  width;
  height;
  scale = 1;
  keys = [];
  tracePoints = [];
  KEYBOARD_LAYOUT = {
    "1": { x: 18, y: 67 },
    "2": { x: 54, y: 67 },
    "3": { x: 90, y: 67 },
    "4": { x: 126, y: 67 },
    "5": { x: 162, y: 67 },
    "6": { x: 198, y: 67 },
    "7": { x: 234, y: 67 },
    "8": { x: 270, y: 67 },
    "9": { x: 306, y: 67 },
    "0": { x: 342, y: 67 },
    q: { x: 18, y: 111 },
    w: { x: 54, y: 111 },
    e: { x: 90, y: 111 },
    r: { x: 126, y: 111 },
    t: { x: 162, y: 111 },
    y: { x: 198, y: 111 },
    u: { x: 234, y: 111 },
    i: { x: 270, y: 111 },
    o: { x: 306, y: 111 },
    p: { x: 342, y: 111 },
    a: { x: 36, y: 155 },
    s: { x: 72, y: 155 },
    d: { x: 108, y: 155 },
    f: { x: 144, y: 155 },
    g: { x: 180, y: 155 },
    h: { x: 216, y: 155 },
    j: { x: 252, y: 155 },
    k: { x: 288, y: 155 },
    l: { x: 324, y: 155 },
    z: { x: 72, y: 199 },
    x: { x: 108, y: 199 },
    c: { x: 144, y: 199 },
    v: { x: 180, y: 199 },
    b: { x: 216, y: 199 },
    n: { x: 252, y: 199 },
    m: { x: 288, y: 199 }
  };
  constructor(canvas, ctx) {
    this.canvas = canvas;
    this.ctx = ctx;
    this.width = canvas.width;
    this.height = canvas.height;
    this.updateScale();
    this.initializeKeys();
  }
  updateDimensions(width, height) {
    this.width = width;
    this.height = height;
    this.updateScale();
  }
  updateScale() {
    this.scale = this.width / 360;
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
    this.ctx.fillStyle = "#1a1a1a";
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.drawKeys();
    if (this.tracePoints.length > 0) {
      this.drawTraceInternal();
    }
  }
  drawKeys() {
    const keySize = 30 * this.scale;
    const fontSize = 16 * this.scale;
    for (const key of this.keys) {
      const x = key.x * this.scale;
      const y = key.y * this.scale;
      this.ctx.fillStyle = "#333";
      this.ctx.fillRect(x - keySize / 2, y - keySize / 2, keySize, keySize);
      this.ctx.strokeStyle = "#555";
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x - keySize / 2, y - keySize / 2, keySize, keySize);
      this.ctx.fillStyle = "#fff";
      this.ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(key.char.toUpperCase(), x, y);
    }
  }
  drawTrace(points) {
    this.tracePoints = points.map((p) => ({
      x: p.x * this.scale,
      y: p.y * this.scale
    }));
    this.render();
  }
  drawTraceInternal() {
    if (this.tracePoints.length < 2)
      return;
    this.ctx.strokeStyle = "rgba(102, 126, 234, 0.8)";
    this.ctx.lineWidth = 3 * this.scale;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    const gradient = this.ctx.createLinearGradient(this.tracePoints[0].x, this.tracePoints[0].y, this.tracePoints[this.tracePoints.length - 1].x, this.tracePoints[this.tracePoints.length - 1].y);
    gradient.addColorStop(0, "rgba(102, 126, 234, 0.4)");
    gradient.addColorStop(1, "rgba(118, 75, 162, 0.8)");
    this.ctx.strokeStyle = gradient;
    this.ctx.beginPath();
    this.ctx.moveTo(this.tracePoints[0].x, this.tracePoints[0].y);
    for (let i = 1;i < this.tracePoints.length; i++) {
      this.ctx.lineTo(this.tracePoints[i].x, this.tracePoints[i].y);
    }
    this.ctx.stroke();
    for (let i = 0;i < this.tracePoints.length; i++) {
      const point = this.tracePoints[i];
      const radius = i === 0 || i === this.tracePoints.length - 1 ? 5 * this.scale : 2 * this.scale;
      this.ctx.fillStyle = i === 0 ? "rgba(102, 234, 126, 0.8)" : i === this.tracePoints.length - 1 ? "rgba(234, 102, 102, 0.8)" : "rgba(255, 255, 255, 0.5)";
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }
  clearTrace() {
    this.tracePoints = [];
    this.render();
  }
  getKeyAt(x, y) {
    const keyX = x / this.scale;
    const keyY = y / this.scale;
    const threshold = 20;
    let nearestKey = null;
    let minDistance = threshold;
    for (const key of this.keys) {
      const distance = Math.sqrt(Math.pow(keyX - key.x, 2) + Math.pow(keyY - key.y, 2));
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
}

// src/predictor.ts
class SwipePredictor {
  encoderSession;
  decoderSession;
  tokenizer;
  keyboardLayout;
  PAD_IDX = 0;
  EOS_IDX = 1;
  UNK_IDX = 2;
  SOS_IDX = 3;
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
      q: { x: 18, y: 111 },
      w: { x: 54, y: 111 },
      e: { x: 90, y: 111 },
      r: { x: 126, y: 111 },
      t: { x: 162, y: 111 },
      y: { x: 198, y: 111 },
      u: { x: 234, y: 111 },
      i: { x: 270, y: 111 },
      o: { x: 306, y: 111 },
      p: { x: 342, y: 111 },
      a: { x: 36, y: 155 },
      s: { x: 72, y: 155 },
      d: { x: 108, y: 155 },
      f: { x: 144, y: 155 },
      g: { x: 180, y: 155 },
      h: { x: 216, y: 155 },
      j: { x: 252, y: 155 },
      k: { x: 288, y: 155 },
      l: { x: 324, y: 155 },
      z: { x: 72, y: 199 },
      x: { x: 108, y: 199 },
      c: { x: 144, y: 199 },
      v: { x: 180, y: 199 },
      b: { x: 216, y: 199 },
      n: { x: 252, y: 199 },
      m: { x: 288, y: 199 }
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
    const FIXED_SEQ_LENGTH = 50;
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
    console.log("Beam search - memory shape:", memory.dims);
    for (let step = 0;step < maxGeneratedTokens; step++) {
      console.log(`Beam search step ${step}`);
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
        console.log("Decoder input shapes:");
        console.log("  target_tokens:", [1, DECODER_SEQ_LENGTH]);
        console.log("  target_mask:", [1, DECODER_SEQ_LENGTH]);
        console.log("  src_mask:", [1, memory.dims[1]]);
        console.log("  Current token count:", beam.tokens.length);
        const decoderInputs = {
          memory,
          target_tokens: new ort.Tensor("int64", paddedTokens, [1, DECODER_SEQ_LENGTH]),
          target_mask: new ort.Tensor("bool", tgtMask, [1, DECODER_SEQ_LENGTH]),
          src_mask: new ort.Tensor("bool", srcMask, [1, memory.dims[1]])
        };
        let decoderOutputs;
        try {
          console.log("Running decoder...");
          decoderOutputs = await this.decoderSession.run(decoderInputs);
          console.log("Decoder outputs:", Object.keys(decoderOutputs));
          const logits2 = decoderOutputs.logits;
          console.log("Logits shape:", logits2.dims);
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
        console.log(`Getting logits for position ${tokenPosition}, indices ${startIdx}-${endIdx}`);
        const probs = this.softmax(relevantLogits);
        const topK = this.getTopK(probs, Math.min(beamSize, vocabSize));
        for (const { idx, prob } of topK) {
          allCandidates.push({
            tokens: [...beam.tokens, idx],
            score: beam.score + Math.log(prob),
            finished: idx === this.EOS_IDX
          });
        }
      }
      beams = allCandidates.sort((a, b) => b.score - a.score).slice(0, beamSize);
      if (beams.every((b) => b.finished))
        break;
    }
    const predictions = beams.map((beam) => ({
      word: this.decodeTokens(beam.tokens),
      score: Math.exp(beam.score / beam.tokens.length)
    }));
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
  handleTouchStart(e) {
    e.preventDefault();
    if (e.touches.length !== 1)
      return;
    const touch = e.touches[0];
    const rect = this.canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    this.startSwipe(x, y);
  }
  handleTouchMove(e) {
    e.preventDefault();
    if (!this.isTracking || e.touches.length !== 1)
      return;
    const touch = e.touches[0];
    const rect = this.canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    this.addPoint(x, y);
  }
  handleTouchEnd(e) {
    e.preventDefault();
    if (!this.isTracking)
      return;
    this.endSwipe();
  }
  handleMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.startSwipe(x, y);
  }
  handleMouseMove(e) {
    if (!this.isTracking)
      return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.addPoint(x, y);
  }
  handleMouseUp(e) {
    if (!this.isTracking)
      return;
    this.endSwipe();
  }
  startSwipe(canvasX, canvasY) {
    this.isTracking = true;
    this.points = [];
    this.startTime = Date.now();
    const keyboardCoords = this.keyboard.canvasToKeyboard(canvasX, canvasY);
    this.points.push({
      x: keyboardCoords.x,
      y: keyboardCoords.y,
      t: 0
    });
    this.emit("swipeStart", this.points);
  }
  addPoint(canvasX, canvasY) {
    if (!this.isTracking)
      return;
    const keyboardCoords = this.keyboard.canvasToKeyboard(canvasX, canvasY);
    const elapsed = Date.now() - this.startTime;
    if (this.points.length > 0) {
      const lastPoint = this.points[this.points.length - 1];
      const distance = Math.sqrt(Math.pow(keyboardCoords.x - lastPoint.x, 2) + Math.pow(keyboardCoords.y - lastPoint.y, 2));
      if (distance < 2)
        return;
    }
    this.points.push({
      x: keyboardCoords.x,
      y: keyboardCoords.y,
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
  debugMode = false;
  constructor() {
    this.canvas = document.getElementById("keyboard-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.predictionsEl = document.getElementById("predictions");
    this.loadingEl = document.getElementById("loading");
    this.loadingProgressEl = document.getElementById("loading-progress");
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
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.width * (215 / 360);
    this.keyboard.updateDimensions(this.canvas.width, this.canvas.height);
    this.keyboard.render();
  }
  async loadModels() {
    this.loadingProgressEl.textContent = "Loading encoder model...";
    await this.predictor.loadEncoder("/models/swipe_model_character.onnx");
    this.loadingProgressEl.textContent = "Loading decoder model...";
    await this.predictor.loadDecoder("/models/swipe_decoder_character.onnx");
    this.loadingProgressEl.textContent = "Loading tokenizer...";
    await this.predictor.loadTokenizer("/models/tokenizer_config.json");
    this.loadingProgressEl.textContent = "Models loaded successfully!";
  }
  setupEventHandlers() {
    this.swipeTracker.on("swipeStart", () => {
      this.keyboard.clearTrace();
      this.clearPredictions();
    });
    this.swipeTracker.on("swipeMove", (points) => {
      this.keyboard.drawTrace(points);
    });
    this.swipeTracker.on("swipeEnd", async (points) => {
      if (points.length < 3) {
        this.keyboard.clearTrace();
        return;
      }
      this.showLoadingPredictions();
      try {
        const predictions = await this.predictor.predict(points, 5);
        this.showPredictions(predictions);
        if (this.debugMode) {
          console.log("Swipe points:", points);
          console.log("Predictions:", predictions);
        }
      } catch (error) {
        console.error("Prediction error:", error);
        console.error("Error stack:", error?.stack);
        console.error("Error message:", error?.message);
        this.showError();
      }
      setTimeout(() => {
        this.keyboard.clearTrace();
      }, 1000);
    });
    document.getElementById("clear-btn")?.addEventListener("click", () => {
      this.keyboard.clearTrace();
      this.clearPredictions();
    });
    document.getElementById("debug-btn")?.addEventListener("click", () => {
      this.debugMode = !this.debugMode;
      const btn = document.getElementById("debug-btn");
      btn.textContent = this.debugMode ? "Debug: ON" : "Debug: OFF";
      btn.style.background = this.debugMode ? "#f44336" : "#667eea";
    });
    this.predictionsEl.addEventListener("click", (e) => {
      const target = e.target;
      if (target.classList.contains("prediction")) {
        const word = target.textContent;
        if (word) {
          this.selectWord(word);
        }
      }
    });
  }
  clearPredictions() {
    this.predictionsEl.innerHTML = '<div class="no-predictions">Swipe on the keyboard to see predictions</div>';
  }
  showLoadingPredictions() {
    this.predictionsEl.innerHTML = '<div class="no-predictions">Processing...</div>';
  }
  showPredictions(predictions) {
    if (predictions.length === 0) {
      this.predictionsEl.innerHTML = '<div class="no-predictions">No predictions found</div>';
      return;
    }
    this.predictionsEl.innerHTML = predictions.map((pred, i) => `
                <div class="prediction ${i === 0 ? "primary" : ""}" 
                     data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </div>
            `).join("");
  }
  showError() {
    this.predictionsEl.innerHTML = '<div class="no-predictions">Error processing swipe</div>';
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

//# debugId=25737B8E9235B6BE64756E2164756E21
