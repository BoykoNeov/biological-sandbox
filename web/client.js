// Main-thread side of the message protocol. See web/protocol.md.
//
// One promise per request, keyed by id; `progress` events are routed to the
// pending request that produced them rather than to a global handler, so a page
// running two things at once cannot cross the streams.

export class WorkerClient {
  constructor(url = "./worker.js") {
    this.worker = new Worker(url);
    this.pending = new Map();
    this.nextId = 0;
    this.worker.onmessage = (event) => this._receive(event);
    this.worker.onerror = (event) => {
      // A worker that dies takes every in-flight request with it. Rejecting them
      // explicitly turns "the page stopped updating" into a message.
      const error = new Error("worker error: " + (event.message || event.type));
      for (const entry of this.pending.values()) entry.reject(error);
      this.pending.clear();
    };
  }

  _receive(event) {
    const { id, payload, error, traceback, event: kind, rgba } = event.data;
    const entry = this.pending.get(id);
    if (!entry) return;
    if (kind === "progress") {
      if (entry.onProgress) entry.onProgress(payload);
      return;
    }
    this.pending.delete(id);
    if (error) {
      const failure = new Error(error);
      failure.traceback = traceback;
      entry.reject(failure);
    } else {
      entry.resolve(rgba === undefined ? payload : { ...payload, rgba });
    }
  }

  send(cmd, arg = {}, onProgress = null) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, cmd, arg });
    });
  }

  terminate() {
    this.worker.terminate();
  }
}

// A tolerant number formatter for the readouts: the values here span
// milliseconds to millions of steps, and a fixed number of decimals is wrong
// somewhere in that range.
export function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) return value.toExponential(2);
  return Number(value.toPrecision(digits)).toString();
}
