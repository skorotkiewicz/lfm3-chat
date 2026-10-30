'use strict';

// One decision per pause. Speech invalidates a pending decision, so a late
// classifier result can never cut off a continuation of the same utterance.
class TurnGate {
  constructor() { this.configure('silence', 900, 2500); this.reset(); }
  configure(mode, pauseMs, holdMs) {
    this.mode = mode;
    this.pauseMs = pauseMs;
    this.holdMs = Math.max(holdMs, pauseMs);
  }
  reset() { this.sequence = 0; this.pending = null; this.checked = false; this.holding = false; }
  voiced() { this.pending = null; this.checked = false; this.holding = false; }
  update(quietMs, durationMs) {
    if (durationMs < 500) return null;
    if (this.mode !== 'smart-turn-v3') return quietMs >= this.pauseMs ? {type: 'commit'} : null;
    if (quietMs >= this.holdMs) return {type: 'commit'};
    if (quietMs >= this.pauseMs && !this.checked) {
      this.checked = true;
      this.pending = ++this.sequence;
      return {type: 'pause', check: this.pending};
    }
    return null;
  }
  decide(check, complete) {
    if (this.pending !== check) return false;
    this.pending = null;
    this.holding = !complete;
    return complete;
  }
}

if (typeof module !== 'undefined') module.exports = {TurnGate};
