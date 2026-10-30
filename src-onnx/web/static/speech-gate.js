'use strict';

// Require several speech-like frames to start, then bridge short consonant gaps.
class SpeechGate {
  constructor() { this.reset(); }
  reset() { this.window = []; this.lastSpeech = -Infinity; this.lastUpdate = -Infinity; }
  update(probability, speechScore, now) {
    this.lastUpdate = now;
    // A speech/non-speech model alone can mistake instruments for speech.
    this.window.push(probability >= 0.65 && speechScore != null && speechScore >= 0.1);
    if (this.window.length > 6) this.window.shift();
    if (this.window.filter(Boolean).length >= 4 ||
        (now - this.lastSpeech < 160 && probability >= 0.35 && speechScore >= 0.1)) this.lastSpeech = now;
  }
  active(now) { return now - this.lastUpdate < 250 && now - this.lastSpeech < 160; }
}
if (typeof module !== 'undefined') module.exports = {SpeechGate};
