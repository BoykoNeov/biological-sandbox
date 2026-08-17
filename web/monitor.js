// The event-loop monitor. This is the instrument, and it is the third one the
// project tried.
//
// The obvious way to measure "is the page frozen" is to watch
// requestAnimationFrame for gaps. It cannot work: rAF is SUSPENDED entirely in a
// background tab, so the slice's first monitor reported `worst freeze 1840 ms`
// for the worker arm -- identical to the main-thread arm -- because neither was
// being measured. A check that reports "blocked" for the arm designed not to
// block is not a check.
//
// A MessageChannel ping-pong is a *task*, not a timer, so it is throttled
// neither by background-tab timer clamping nor by rAF suspension. And
// event-loop latency is precisely what "the page is frozen" means. rAF is kept
// as a second opinion, valid only while the tab is genuinely visible.

const loopTicks = [];
const frameTimes = [];

const channel = new MessageChannel();
channel.port1.onmessage = () => {
  loopTicks.push(performance.now());
  channel.port2.postMessage(0);
};
channel.port2.postMessage(0);

(function tick(t) {
  frameTimes.push(t);
  requestAnimationFrame(tick);
})(performance.now());

const hiddenSpans = [];
let hiddenFrom = document.hidden ? performance.now() : null;
document.addEventListener("visibilitychange", () => {
  const t = performance.now();
  if (document.hidden) hiddenFrom = t;
  else if (hiddenFrom !== null) {
    hiddenSpans.push([hiddenFrom, t]);
    hiddenFrom = null;
  }
});

function wasHidden(t0, t1) {
  if (hiddenFrom !== null && hiddenFrom < t1) return true;
  return hiddenSpans.some(([a, b]) => a < t1 && b > t0);
}

function gapsIn(stamps, t0, t1) {
  const inside = stamps.filter((t) => t >= t0 && t <= t1);
  let worst = 0;
  for (let i = 1; i < inside.length; i++) worst = Math.max(worst, inside[i] - inside[i - 1]);
  if (inside.length) {
    worst = Math.max(worst, inside[0] - t0, t1 - inside[inside.length - 1]);
  }
  return { count: inside.length, worst: inside.length ? worst : null };
}

// The clock's own granularity, printed rather than assumed. Outside a
// cross-origin-isolated page performance.now() is coarsened to 0.1 ms, and a
// single sub-millisecond reading is then a rounding artifact. web/serve.py sends
// the COOP/COEP headers that lift that, and this reports whether they took.
export const CLOCK_MS = (() => {
  const deltas = new Set();
  for (let i = 0; i < 5000; i++) {
    const a = performance.now();
    deltas.add(performance.now() - a);
  }
  const positive = [...deltas].filter((d) => d > 0).sort((a, b) => a - b);
  return positive.length ? positive[0] : 0;
})();

export const CROSS_ORIGIN_ISOLATED = self.crossOriginIsolated === true;

export function windowFrom(t0, t1 = performance.now()) {
  const span_s = (t1 - t0) / 1000;
  const hidden = wasHidden(t0, t1);
  const loop = gapsIn(loopTicks, t0, t1);
  const raf = gapsIn(frameTimes, t0, t1);
  return {
    span_s,
    hidden,
    loop_ticks: loop.count,
    // Zero loop turns is NOT a dead instrument. It is the strongest reading
    // available: the thread never reached its task queue, so it was blocked for
    // the whole window. The slice's first version printed this case as
    // "[monitor produced nothing]", which reads like a failure and is in fact
    // the headline result.
    blocked_ms: loop.count > 0 ? loop.worst : (t1 - t0),
    blocked_is_lower_bound: loop.count === 0,
    frames: raf.count,
    fps: raf.count / span_s,
    raf_worst_ms: raf.worst,
    raf_valid: !hidden && raf.count > 0,
  };
}

export async function watch(fn) {
  const t0 = performance.now();
  const value = await fn();
  return { value, window: windowFrom(t0) };
}

export function describeWindow(w) {
  const blocked = w.blocked_is_lower_bound
    ? `blocked >= ${w.blocked_ms.toFixed(0)} ms — the entire window, the event loop never got a turn`
    : `blocked ${w.blocked_ms.toFixed(0)} ms (${w.loop_ticks} loop turns)`;
  const paint = w.raf_valid
    ? `, paint ${w.fps.toFixed(0)} fps / worst frame ${w.raf_worst_ms.toFixed(0)} ms`
    : w.hidden
      ? ", paint n/a (tab hidden — rAF is suspended, not slow)"
      : ", paint n/a (no frames)";
  return `${w.span_s.toFixed(2)} s   ${blocked}${paint}`;
}
