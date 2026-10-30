const {test} = require('node:test');
const assert = require('node:assert/strict');
const {TurnGate} = require('../shared/web/static/turn-gate.js');

function smartGate() { const gate = new TurnGate(); gate.configure('smart-turn-v3', 250, 2500); return gate; }

test('short pauses do not trigger a turn decision', () => {
  const gate = smartGate();
  assert.equal(gate.update(249, 2000), null);
  assert.deepEqual(gate.update(250, 2000), {type: 'pause', check: 1});
  assert.equal(gate.update(1000, 3000), null);
});

test('unfinished speech stays open, with a bounded silence fallback', () => {
  const gate = smartGate();
  gate.update(250, 2000);
  assert.equal(gate.decide(1, false), false);
  assert.equal(gate.holding, true);
  assert.equal(gate.update(2499, 4000), null);
  assert.deepEqual(gate.update(2500, 4000), {type: 'commit'});
});

test('speech invalidates a late completion decision without losing the utterance', () => {
  const gate = smartGate();
  gate.update(250, 2000);
  gate.voiced();
  assert.equal(gate.decide(1, true), false);
  assert.deepEqual(gate.update(250, 3000), {type: 'pause', check: 2});
  assert.equal(gate.decide(1, true), false);
  assert.equal(gate.decide(2, true), true);
});

test('a missing classifier response cannot leave the user waiting forever', () => {
  const gate = smartGate();
  gate.update(250, 2000);
  assert.deepEqual(gate.update(2500, 5000), {type: 'commit'});
});

test('without the model the original silence threshold still works', () => {
  const gate = new TurnGate();
  assert.equal(gate.update(899, 2000), null);
  assert.deepEqual(gate.update(900, 2000), {type: 'commit'});
});

test('very short sounds cannot trigger automatic submission', () => {
  const gate = smartGate();
  assert.equal(gate.update(2500, 400), null);
});
