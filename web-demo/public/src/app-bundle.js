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
    this.updateScale();
    this.initializeKeys();
  }
  updateDimensions(width, height) {
    this.width = width;
    this.height = height;
    this.updateScale();
  }
  updateScale() {
    const scaleX = this.width / 360;
    const scaleY = this.height / 215;
    this.scale = scaleX;
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
    const isDark = document.documentElement.classList.contains("dark");
    const bgGradient = this.ctx.createLinearGradient(0, 0, this.width, this.height);
    if (isDark) {
      bgGradient.addColorStop(0, "#1e293b");
      bgGradient.addColorStop(1, "#0f172a");
    } else {
      bgGradient.addColorStop(0, "#f1f5f9");
      bgGradient.addColorStop(1, "#e2e8f0");
    }
    this.ctx.fillStyle = bgGradient;
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.drawKeys();
    if (this.tracePoints.length > 0) {
      this.drawTraceInternal();
    }
  }
  drawKeys() {
    const keySize = 30 * this.scale;
    const fontSize = 16 * this.scale;
    const isDark = document.documentElement.classList.contains("dark");
    for (const key of this.keys) {
      const x = key.x * this.scale;
      const y = key.y * this.scale;
      this.ctx.save();
      this.ctx.shadowColor = isDark ? "rgba(0, 0, 0, 0.5)" : "rgba(0, 0, 0, 0.2)";
      this.ctx.shadowBlur = 4 * this.scale;
      this.ctx.shadowOffsetX = 0;
      this.ctx.shadowOffsetY = 2 * this.scale;
      const gradient = this.ctx.createLinearGradient(x - keySize / 2, y - keySize / 2, x + keySize / 2, y + keySize / 2);
      if (isDark) {
        gradient.addColorStop(0, "#475569");
        gradient.addColorStop(0.5, "#334155");
        gradient.addColorStop(1, "#1e293b");
      } else {
        gradient.addColorStop(0, "#ffffff");
        gradient.addColorStop(0.5, "#f8fafc");
        gradient.addColorStop(1, "#f1f5f9");
      }
      this.ctx.fillStyle = gradient;
      this.roundRect(x - keySize / 2, y - keySize / 2, keySize, keySize, 5 * this.scale);
      this.ctx.fill();
      this.ctx.restore();
      const borderGradient = this.ctx.createLinearGradient(x - keySize / 2, y - keySize / 2, x + keySize / 2, y + keySize / 2);
      if (isDark) {
        borderGradient.addColorStop(0, "#64748b");
        borderGradient.addColorStop(1, "#475569");
      } else {
        borderGradient.addColorStop(0, "#cbd5e1");
        borderGradient.addColorStop(1, "#94a3b8");
      }
      this.ctx.strokeStyle = borderGradient;
      this.ctx.lineWidth = 1;
      this.roundRect(x - keySize / 2, y - keySize / 2, keySize, keySize, 5 * this.scale);
      this.ctx.stroke();
      this.ctx.save();
      this.ctx.fillStyle = isDark ? "#ffffff" : "#000000";
      this.ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      if (isDark) {
        this.ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
        this.ctx.shadowBlur = 2;
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 1;
      }
      this.ctx.fillText(key.char.toUpperCase(), x, y);
      this.ctx.restore();
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
    this.ctx.save();
    this.ctx.shadowColor = "rgba(99, 102, 241, 0.6)";
    this.ctx.shadowBlur = 10 * this.scale;
    this.ctx.lineWidth = 4 * this.scale;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    const gradient = this.ctx.createLinearGradient(this.tracePoints[0].x, this.tracePoints[0].y, this.tracePoints[this.tracePoints.length - 1].x, this.tracePoints[this.tracePoints.length - 1].y);
    gradient.addColorStop(0, "rgba(34, 197, 94, 0.9)");
    gradient.addColorStop(0.5, "rgba(99, 102, 241, 0.9)");
    gradient.addColorStop(1, "rgba(168, 85, 247, 0.9)");
    this.ctx.strokeStyle = gradient;
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
      const radius = isEndpoint ? 6 * this.scale : 2 * this.scale;
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
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const isMobile = window.innerWidth < 640;
    const targetAspectRatio = isMobile ? 1.4 : 360 / 215;
    if (isMobile) {
      this.canvas.style.width = "100%";
      const desiredHeight = rect.width / targetAspectRatio;
      if (desiredHeight <= rect.height) {
        this.canvas.width = rect.width;
        this.canvas.height = desiredHeight;
        this.canvas.style.height = desiredHeight + "px";
      } else {
        this.canvas.height = rect.height;
        this.canvas.width = rect.height * targetAspectRatio;
        this.canvas.style.height = "100%";
        this.canvas.style.width = rect.height * targetAspectRatio + "px";
      }
    } else {
      if (rect.width / rect.height > targetAspectRatio) {
        this.canvas.height = rect.height;
        this.canvas.width = rect.height * targetAspectRatio;
      } else {
        this.canvas.width = rect.width;
        this.canvas.height = rect.width / targetAspectRatio;
      }
    }
    this.keyboard.updateDimensions(this.canvas.width, this.canvas.height);
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
      this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Swiping...</span>';
      this.statusEl.textContent = "Swiping";
      this.statusEl.className = "text-sm font-semibold text-blue-600 dark:text-blue-400";
    });
    this.swipeTracker.on("swipeMove", (points) => {
      this.keyboard.drawTrace(points);
      if (points.length > 0) {
        const lastPoint = points[points.length - 1];
        const scale = this.canvas.width / 360;
        const canvasX = lastPoint.x * scale;
        const canvasY = lastPoint.y * scale;
        const key = this.keyboard.getKeyAt(canvasX, canvasY);
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
        this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Too short - try again</span>';
        this.statusEl.textContent = "Ready";
        this.statusEl.className = "text-sm font-semibold text-green-600 dark:text-green-400";
        return;
      }
      this.showLoadingPredictions();
      this.statusEl.textContent = "Processing";
      this.statusEl.className = "text-sm font-semibold text-yellow-600 dark:text-yellow-400";
      try {
        const predictions = await this.predictor.predict(points, 5);
        this.showPredictions(predictions);
        this.statusEl.textContent = "Ready";
        this.statusEl.className = "text-sm font-semibold text-green-600 dark:text-green-400";
        if (this.debugMode) {
          console.log("Swipe points:", points);
          console.log("Predictions:", predictions);
        }
      } catch (error) {
        console.error("Prediction error:", error);
        console.error("Error stack:", error?.stack);
        console.error("Error message:", error?.message);
        this.showError();
        this.statusEl.textContent = "Error";
        this.statusEl.className = "text-sm font-semibold text-red-600 dark:text-red-400";
      }
      setTimeout(() => {
        this.keyboard.clearTrace();
      }, 1000);
    });
    document.getElementById("clear-btn")?.addEventListener("click", () => {
      this.keyboard.clearTrace();
      this.clearPredictions();
      this.swipeCharsEl.innerHTML = '<span class="text-gray-400 dark:text-gray-600 text-base">Touch the keyboard to start swiping...</span>';
      this.statusEl.textContent = "Ready";
      this.statusEl.className = "text-sm font-semibold text-green-600 dark:text-green-400";
    });
    document.getElementById("debug-btn")?.addEventListener("click", () => {
      this.debugMode = !this.debugMode;
      const btn = document.getElementById("debug-btn");
      btn.textContent = this.debugMode ? "Debug: ON" : "Debug: OFF";
      if (this.debugMode) {
        btn.className = "px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-all duration-150 active:scale-95";
      } else {
        btn.className = "px-3 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 text-sm font-medium rounded-lg transition-all duration-150 active:scale-95";
      }
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
    this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full">Swipe on the keyboard to see predictions</p>';
  }
  showLoadingPredictions() {
    this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full animate-pulse">Processing...</p>';
  }
  showPredictions(predictions) {
    if (predictions.length === 0) {
      this.predictionsEl.innerHTML = '<p class="text-gray-500 dark:text-gray-400 text-center w-full">No predictions found</p>';
      return;
    }
    this.predictionsEl.innerHTML = predictions.map((pred, i) => `
                <button class="px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg font-mono text-xs sm:text-sm font-semibold transition-all duration-150
                              ${i === 0 ? "bg-indigo-600 hover:bg-indigo-700 text-white shadow-md" : "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 text-gray-900 dark:text-gray-100"}"
                        data-score="${pred.score.toFixed(3)}">
                    ${pred.word}
                </button>
            `).join("");
  }
  showError() {
    this.predictionsEl.innerHTML = '<p class="text-red-500 dark:text-red-400 text-center w-full">Error processing swipe</p>';
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

//# debugId=C1AE7B3D6010C64764756E2164756E21
//# sourceMappingURL=app.js.map
