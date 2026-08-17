// The drawing path. Two renderers, both consuming protocol data only.
//
// The measurements price this whole path at 1.6-2.2% of a frame at every grid
// size, against 98% for the simulation step, so it is written for legibility
// rather than for speed.
//
// Two decisions here are about honesty rather than looks, and both are the
// figure lesson this project keeps relearning — a figure carries a claim and
// gets looked at, not exit-code checked:
//
//   * The y-range never SHRINKS during a run. An axis that follows the data down
//     draws a signal decaying to nothing exactly like one at full strength.
//   * A stochastic trace is decimated by per-column min/max, never by strides.
//     Taking every k-th point throws away the scatter, and the scatter is the
//     whole subject: a run drawn by strides looks smoother than it is, which is
//     a claim about the model.

// The categorical slots, in fixed order, from the validated reference palette.
// Assigned by series index and never cycled or re-assigned when a series is
// hidden -- colour follows the entity, not its rank, so toggling m2 off must not
// repaint p1.
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"];
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"];

export function seriesColors() {
  const dark =
    document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  return dark ? SERIES_DARK : SERIES_LIGHT;
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.body).getPropertyValue(name).trim();
  return value || fallback;
}

function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

function tickLabel(value, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(Math.abs(step || 1))) + 0);
  if (Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-3)) {
    return value.toExponential(1);
  }
  return value.toFixed(Math.min(decimals, 4));
}

export class TraceChart {
  /**
   * @param canvas   the target canvas
   * @param options  {keys, xMax, xLabel, yLabel, title, colorOffset}
   *
   * ``colorOffset`` lets two panels of one run share a hue per entity: the
   * repressilator's mRNA panel and protein panel both start at slot 0, so gene
   * 1's mRNA and gene 1's protein are the same colour in both. Colour follows
   * the entity, not the panel.
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.title = options.title ?? "";
    this.colorOffset = options.colorOffset ?? 0;
    this.keys = options.keys ?? [];
    this.visible = new Set(this.keys);
    this.xLabel = options.xLabel ?? "time";
    this.yLabel = options.yLabel ?? "";
    this.xMax = options.xMax ?? 1;
    this.replicates = [];
    this.limit = null;
    this.yLo = 0;
    this.yHi = 0;
    this.hasData = false;
  }

  reset(options = {}) {
    Object.assign(this, {
      keys: options.keys ?? this.keys,
      xMax: options.xMax ?? this.xMax,
      replicates: [],
      limit: null,
      yLo: 0,
      yHi: 0,
      hasData: false,
      _columnCount: null,
    });
    this.visible = new Set(this.keys);
  }

  setLimit(limit) {
    if (!limit || !limit.available) {
      this.limit = null;
      return;
    }
    this.limit = { t: limit.t, series: limit.series };
    for (const key of this.keys) this._grow(limit.series[key]);
  }

  /**
   * Append a drained chunk for one replicate. Points are never re-sent.
   *
   * The min/max envelope is folded in HERE, not at draw time. The data is
   * append-only and the x-scale is fixed for the whole run, so a point's column
   * never changes once assigned — which makes a redraw cost O(columns) instead
   * of O(points). Measured on a 400 000-event trace with four replicates and six
   * series: 43 ms per redraw recomputing from scratch, and the p95 was a second.
   * Drawing must not get more expensive the longer a run goes, or a demo that
   * streams gets worse exactly as it gets interesting.
   */
  append(index, chunk) {
    while (this.replicates.length <= index) {
      this.replicates.push({ t: [], series: {}, envelope: {} });
    }
    const store = this.replicates[index];
    const columns = this._columns();
    store.t.push(...chunk.t);
    for (const [key, values] of Object.entries(chunk.series)) {
      // A chart draws only the keys it was given. Two panels over one run is how
      // series of different magnitude are shown -- never two y-scales on one
      // frame, which lets a reader compare two things that are not comparable.
      if (!this.keys.includes(key)) continue;
      (store.series[key] ??= []).push(...values);
      if (this.visible.has(key)) this._grow(values);
      const band = (store.envelope[key] ??= {
        lo: new Float64Array(columns).fill(Infinity),
        hi: new Float64Array(columns).fill(-Infinity),
      });
      const offset = store.t.length - values.length;
      for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (value === null || !Number.isFinite(value)) continue;
        const column = this._columnOf(store.t[offset + i], columns);
        if (value < band.lo[column]) band.lo[column] = value;
        if (value > band.hi[column]) band.hi[column] = value;
      }
    }
    if (chunk.t.length) this.hasData = true;
  }

  /** Envelope resolution. Fixed for a run so a point's column never moves. */
  _columns() {
    this._columnCount ??= Math.max(200, Math.round(this.canvas.clientWidth || 900));
    return this._columnCount;
  }

  _columnOf(t, columns) {
    const fraction = this.xMax > 0 ? t / this.xMax : 0;
    return Math.min(columns - 1, Math.max(0, Math.round(fraction * (columns - 1))));
  }

  _grow(values) {
    if (!values) return;
    for (const value of values) {
      if (value === null || !Number.isFinite(value)) continue;
      // Monotone: the range only ever widens. See the header.
      if (value < this.yLo) this.yLo = value;
      if (value > this.yHi) this.yHi = value;
    }
  }

  toggle(key, on) {
    if (on) this.visible.add(key);
    else this.visible.delete(key);
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (this.canvas.width !== width * dpr || this.canvas.height !== height * dpr) {
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const ink = cssVar("--ink", "#16181d");
    const muted = cssVar("--muted", "#5b616e");
    const line = cssVar("--line", "#dfe2ea");
    const colors = seriesColors();

    const pad = { top: this.title ? 26 : 14, right: 60, bottom: 34, left: 56 };
    const plotW = Math.max(10, width - pad.left - pad.right);
    const plotH = Math.max(10, height - pad.top - pad.bottom);

    // A hair of headroom so the topmost point is not welded to the frame.
    const span = this.yHi - this.yLo || 1;
    const yLo = this.yLo - span * 0.04;
    const yHi = this.yHi + span * 0.04;
    const xOf = (t) => pad.left + (t / this.xMax) * plotW;
    const yOf = (v) => pad.top + plotH - ((v - yLo) / (yHi - yLo)) * plotH;

    // --- recessive grid and axes ---------------------------------------
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.strokeStyle = line;
    ctx.fillStyle = muted;
    ctx.lineWidth = 1;

    const yTicks = niceTicks(yLo, yHi, 5);
    const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : yHi - yLo;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tick of yTicks) {
      const y = Math.round(yOf(tick)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(tickLabel(tick, yStep), pad.left - 8, y);
    }

    const xTicks = niceTicks(0, this.xMax, 6);
    const xStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : this.xMax;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tick of xTicks) {
      const x = Math.round(xOf(tick)) + 0.5;
      ctx.fillText(tickLabel(tick, xStep), x, pad.top + plotH + 8);
    }

    if (this.title) {
      ctx.fillStyle = ink;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(this.title, pad.left, 16);
      ctx.font = "11px ui-monospace, Consolas, monospace";
    }

    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(this.xLabel, pad.left + plotW / 2, height - 12);
    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(this.yLabel, 0, 0);
    ctx.restore();

    if (!this.hasData && !this.limit) {
      ctx.fillStyle = muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("no data yet", pad.left + plotW / 2, pad.top + plotH / 2);
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    // --- the stochastic replicates -------------------------------------
    // Thin and translucent: there may be several per key and the point is the
    // spread around the limit, not any one of them.
    for (const store of this.replicates) {
      for (const [k, key] of this.keys.entries()) {
        if (!this.visible.has(key)) continue;
        const band = store.envelope[key];
        if (!band) continue;
        ctx.strokeStyle = colors[(k + this.colorOffset) % colors.length];
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1;
        this._strokeEnvelope(ctx, band, yOf, plotW, pad.left);
        ctx.globalAlpha = 1;
      }
    }

    // --- the deterministic limit ----------------------------------------
    if (this.limit) {
      for (const [k, key] of this.keys.entries()) {
        if (!this.visible.has(key)) continue;
        const values = this.limit.series[key];
        if (!values) continue;
        ctx.strokeStyle = colors[(k + this.colorOffset) % colors.length];
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < values.length; i++) {
          const v = values[i];
          if (v === null || !Number.isFinite(v)) continue;
          const x = xOf(this.limit.t[i]);
          const y = yOf(v);
          if (started) ctx.lineTo(x, y);
          else {
            ctx.moveTo(x, y);
            started = true;
          }
        }
        ctx.stroke();
      }
    }
    ctx.restore();

    // --- direct labels --------------------------------------------------
    // Three of the light-mode slots sit below 3:1 against the surface, so the
    // palette's relief rule applies: identity must not rest on colour alone.
    const source = this.limit ?? this.replicates[0];
    if (source) {
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      // Collect first, then de-collide by spreading within the frame. Nudging
      // each label down as it is placed pushes the last ones off the bottom,
      // which is how the first version of this ended up with six labels in a
      // pile below the axis.
      const wanted = [];
      for (const [k, key] of this.keys.entries()) {
        if (!this.visible.has(key)) continue;
        const values = source.series?.[key] ?? source[key];
        if (!values || !values.length) continue;
        let last = null;
        for (let i = values.length - 1; i >= 0 && last === null; i--) {
          if (values[i] !== null && Number.isFinite(values[i])) last = values[i];
        }
        if (last === null) continue;
        wanted.push({ key, k, y: yOf(last) });
      }
      wanted.sort((a, b) => a.y - b.y);
      const gap = 13;
      const top = pad.top + 6;
      const bottom = pad.top + plotH - 6;
      for (let i = 0; i < wanted.length; i++) {
        wanted[i].y = Math.max(wanted[i].y, top + i * gap);
      }
      for (let i = wanted.length - 1; i >= 0; i--) {
        wanted[i].y = Math.min(wanted[i].y, bottom - (wanted.length - 1 - i) * gap);
      }
      for (const label of wanted) {
        ctx.fillStyle = colors[(label.k + this.colorOffset) % colors.length];
        ctx.fillText(label.key, pad.left + plotW + 7, label.y);
      }
    }

    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.25;
    ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW, plotH);
    ctx.globalAlpha = 1;
  }

  /**
   * Stroke a pre-folded min/max envelope.
   *
   * Min/max per column, never a stride. Taking every k-th point of a 400 000-event
   * SSA trace draws a tidier trace than the model produces, and the untidiness is
   * the subject: the whole picture is how much the replicates scatter around
   * their limit. The envelope keeps the visible amplitude of that scatter exactly.
   */
  _strokeEnvelope(ctx, band, yOf, plotW, left) {
    const columns = band.lo.length;
    const scale = plotW / (columns - 1 || 1);
    ctx.beginPath();
    let started = false;
    for (let column = 0; column < columns; column++) {
      if (band.lo[column] === Infinity) continue;
      const x = left + column * scale;
      const yTop = yOf(band.hi[column]);
      const yBottom = yOf(band.lo[column]);
      if (!started) {
        ctx.moveTo(x, yTop);
        started = true;
      } else {
        ctx.lineTo(x, yTop);
      }
      if (yBottom !== yTop) ctx.lineTo(x, yBottom);
    }
    if (started) ctx.stroke();
  }
}

export class FieldView {
  /** A 2-D field as an image, plus the colour range it was drawn against. */
  constructor(canvas, scaleCanvas = null) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scaleCanvas = scaleCanvas;
    this.offscreen = document.createElement("canvas");
    this.offscreenCtx = this.offscreen.getContext("2d");
  }

  /**
   * @param rgba    Uint8ClampedArray of length width*height*4, from the bridge
   * @param meta    {width, height, vmin, vmax, field, t}
   */
  put(rgba, meta) {
    const { width, height } = meta;
    if (this.offscreen.width !== width || this.offscreen.height !== height) {
      this.offscreen.width = width;
      this.offscreen.height = height;
    }
    this.offscreenCtx.putImageData(new ImageData(rgba, width, height), 0, 0);

    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Nearest-neighbour: a smoothed field invents structure between cells, and
    // the cell IS the discretization the model is defined on.
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.drawImage(this.offscreen, 0, 0, w, h);
  }

  /**
   * The scale bar is drawn from a strip the BRIDGE produced through the same
   * colormap code as the image, so the two cannot disagree. A scale bar drawn
   * independently is a second implementation of the mapping, and the picture
   * would keep looking right while meaning something else.
   */
  putScale(rgba, meta) {
    if (!this.scaleCanvas) return;
    const ctx = this.scaleCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = this.scaleCanvas.clientWidth;
    const h = this.scaleCanvas.clientHeight;
    this.scaleCanvas.width = w * dpr;
    this.scaleCanvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const strip = document.createElement("canvas");
    strip.width = meta.width;
    strip.height = meta.height;
    strip.getContext("2d").putImageData(new ImageData(rgba, meta.width, meta.height), 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(strip, 0, 0, w, h);
  }
}
