const {test} = require('node:test');
const assert = require('node:assert/strict');
const {SpeechGate} = require('../web/static/speech-gate.js');

test('isolated noise and a single high-confidence frame do not start speech', () => {
  const gate = new SpeechGate();
  [0.02, 0.05, 0.92, 0.08, 0.03].forEach((p, i) => gate.update(p, 0.8, i * 32));
  assert.equal(gate.active(160), false);
});

test('sustained speech activates and a short gap remains voiced', () => {
  const gate = new SpeechGate();
  [0.1, 0.82, 0.92, 0.89, 0.85].forEach((p, i) => gate.update(p, 0.8, i * 32));
  assert.equal(gate.active(128), true);
  gate.update(0.01, 0.8, 160);
  assert.equal(gate.active(190), true);
  assert.equal(gate.active(400), false);
});

test('reset drops stale speech and disconnects cannot leave it active', () => {
  const gate = new SpeechGate();
  [0.9, 0.9, 0.9, 0.9].forEach((p, i) => gate.update(p, 0.8, i * 32));
  assert.equal(gate.active(96), true);
  gate.reset();
  assert.equal(gate.active(100), false);
});

test('music is rejected even if the voice detector fires', () => {
  const gate = new SpeechGate();
  [0.9, 0.9, 0.9, 0.9, 0.9, 0.9].forEach((p, i) => gate.update(p, 0.001, i * 32));
  assert.equal(gate.active(160), false);
});

test('no classifier result cannot silently enable hands-free', () => {
  const gate = new SpeechGate();
  [0.9, 0.9, 0.9, 0.9].forEach((p, i) => gate.update(p, null, i * 32));
  assert.equal(gate.active(96), false);
});
