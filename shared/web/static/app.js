
'use strict';
const $ = id => document.getElementById(id);
let sessionId = crypto.randomUUID();
let stream, context, source, worklet, mute, timer, socket, speechSocket, audioUrl;
let connected = false, busy = false, recording = false, manual = false, resetting = false;
let preRoll = [], recordedSamples = 0, frameCount = 0;
let lastVoice = 0, resumeAt = 0, modelReady = false, maxSeconds = 30;
let nextTurn = 0, activeTurn = 0, replyBody, replyMeta, replyChunks = [];
let playbackAt = 0;
const playing = new Set();
const turnGate = new TurnGate(), speechGate = new SpeechGate();
let micAnalyser, replyAnalyser, speakingSince = 0;
let speechDetectionMode = 'manual', noiseFloor = 0.002, speechWindow = [];
let waveformFrame;
let previewMode = false;

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function message(text, who, meta = '') {
  $('empty')?.remove();
  const box = document.createElement('div'); box.className = 'message ' + who;
  const label = document.createElement('div'); label.className = 'meta'; label.textContent = meta;
  if (meta) box.append(label);
  const body = document.createElement('div'); body.textContent = text; box.append(body);
  $('messages').append(box); $('messages').scrollTop = $('messages').scrollHeight;
  return {body, label};
}
function clearMessages() {
  const empty = document.createElement('p'); empty.className = 'muted'; empty.id = 'empty';
  empty.textContent = 'Your conversation will appear here.';
  $('messages').replaceChildren(empty);
}
function controls() {
  $('start').disabled = connected || !modelReady;
  $('start').hidden = connected;
  $('stop').disabled = !connected;
  $('stop').hidden = !connected;
  $('talk').disabled = !connected || resetting;
  $('talk').hidden = !connected;
  $('talk').textContent = recording ? 'Send' : busy || playing.size || !$('reply').paused ? 'Interrupt & speak' : 'Speak';
  $('talk').classList.toggle('active', recording);
  $('clear').disabled = resetting;
  const state = $('status').classList.contains('error') ? 'error' : !connected ? 'idle' :
    recording ? (turnGate.holding ? 'holding' : turnGate.pending ? 'checking' : 'recording') :
    playing.size ? 'speaking' : busy ? 'thinking' : 'listening';
  document.body.dataset.state = state;
  $('stateLabel').textContent = {idle: 'Ready when you are', error: 'Needs attention',
    recording: 'Listening to you', holding: 'Take your time', checking: 'Checking your pause',
    speaking: 'Speaking', thinking: 'Thinking', listening: 'Listening'}[state];
  $('connection').textContent = previewMode ? 'Preview' : connected ? 'Live' : modelReady ? 'Ready' : 'Offline';
  $('cameraPlaceholder').hidden = connected && !!stream?.getVideoTracks().length && $('useCamera').checked;
  $('cameraNote').hidden = !$('cameraPlaceholder').hidden;
  $('cameraTitle').textContent = connected ? 'Camera off' : 'Ready to talk?';
  $('cameraHelp').textContent = connected ? 'Your microphone is ready.' : 'Start a conversation to turn on your microphone and camera.';
}
function listening() {
  if (connected && !busy && !recording && !playing.size && $('reply').paused) {
    status($('automatic').checked ? 'Listening...' : 'Ready. Click Speak to record.');
  }
  controls();
}
function send(event) {
  if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error('Connection closed. Restart the camera and microphone.');
  if (socket.bufferedAmount > 1_000_000) throw new Error('Connection is too slow. Restart and try a shorter recording.');
  socket.send(event instanceof ArrayBuffer ? event : JSON.stringify(event));
}
function clearRecording() {
  recording = false; manual = false; preRoll = []; recordedSamples = 0; frameCount = 0;
  turnGate.reset(); speechWindow = [];
  controls();
}
function stopPlayback() {
  for (const node of playing) { node.onended = null; node.stop(); node.disconnect(); }
  playing.clear(); playbackAt = 0; $('reply').pause();
}
function clearReplay() {
  $('reply').removeAttribute('src'); $('reply').load();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null; replyChunks = [];
}
function cancelReply() {
  if (socket?.readyState === WebSocket.OPEN) send({type: 'cancel'});
  if (busy && replyMeta) replyMeta.textContent = 'LFM3 · interrupted';
  stopPlayback(); busy = false; resumeAt = 0;
}
function encodeSamples(samples) {
  const buffer = new ArrayBuffer(samples.length * 2), view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) view.setInt16(i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return buffer;
}
function sendSamples(samples) {
  send(encodeSamples(samples)); recordedSamples += samples.length;
}
function captureFrame() {
  const video = $('camera');
  if (!recording || !$('useCamera').checked || !video.videoWidth) return;
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 384 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  send({type: 'frame', time: recordedSamples / 16000, image: canvas.toDataURL('image/jpeg', 0.75).split(',')[1]});
  frameCount++;
}
function beginTurn(isManual) {
  if (!connected || recording || resetting) return;
  cancelReply();
  activeTurn = ++nextTurn;
  send({type: 'start', id: activeTurn, session_id: sessionId});
  clearReplay(); replyBody = replyMeta = null;
  recording = true; manual = isManual; recordedSamples = 0; frameCount = 0;
  turnGate.reset();
  const prior = isManual && speechDetectionMode !== 'volume' ? [] : preRoll; preRoll = [];
  for (const samples of prior) sendSamples(samples);
  lastVoice = performance.now();
  captureFrame(); status('Recording...'); controls();
}
function onSamples(samples) {
  let sum = 0; for (const x of samples) sum += x * x;
  const rms = Math.sqrt(sum / samples.length); $('level').value = rms;
  if (!connected || resetting) return;
  const now = performance.now();
  if (speechSocket?.readyState === WebSocket.OPEN && speechSocket.bufferedAmount < 64000) {
    speechSocket.send(encodeSamples(samples));
  }
  let voiced;
  if (speechDetectionMode === 'volume') {
    const threshold = Math.max(Number($('threshold').value), noiseFloor * 3);
    voiced = rms >= threshold;
    if (!recording && !busy && !playing.size && !voiced) noiseFloor = noiseFloor * 0.98 + rms * 0.02;
  } else {
    voiced = speechGate.active(now);
  }
  if (!recording) {
    const answering = busy || playing.size || !$('reply').paused;
    if (answering && !$('bargeIn').checked) { preRoll = []; return; }
    if (now < resumeAt && (speechDetectionMode !== 'volume' || !answering)) { preRoll = []; return; }
    // Ignore the first echo burst when playback starts; manual interruption
    // always remains available. A rolling window tolerates consonant gaps.
    const canInterrupt = !playing.size || now - speakingSince > 600;
    let sustained = true;
    if (speechDetectionMode === 'volume') {
      speechWindow.push(voiced ? 1 : 0);
      if (speechWindow.length > 8) speechWindow.shift();
      sustained = speechWindow.reduce((sum, value) => sum + value, 0) >= 6;
    }
    if ($('automatic').checked && voiced && (!answering || sustained && canInterrupt)) beginTurn(false);
    else {
      preRoll.push(samples);
      while (preRoll.length > (speechDetectionMode === 'volume' ? 8 : 48)) preRoll.shift();
      return;
    }
  }
  try {
    sendSamples(samples);
    if (voiced) {
      const wasWaiting = turnGate.pending !== null || turnGate.holding;
      lastVoice = now; turnGate.voiced();
      if (wasWaiting) { status('Recording...'); controls(); }
    }
    if (recordedSamples >= 16000 * (maxSeconds - 0.1)) commitTurn();
    else if (!manual) {
      const action = turnGate.update(now - lastVoice, recordedSamples / 16);
      if (action?.type === 'commit') commitTurn();
      else if (action?.type === 'pause') { send(action); status('Checking your pause...'); controls(); }
    }
  } catch (error) { failConnection(error); }
}
function commitTurn() {
  if (!recording) return;
  if (recordedSamples < 1600) { send({type: 'cancel'}); clearRecording(); listening(); return; }
  captureFrame();
  const duration = recordedSamples / 16000, count = $('useCamera').checked ? Math.min(6, frameCount) : 0;
  send({type: 'commit', use_camera: $('useCamera').checked});
  busy = true;
  if (speechDetectionMode === 'classifier') { speechGate.reset(); resumeAt = performance.now() + 350; }
  clearRecording(); status('Model is thinking...'); controls();
  message(`Spoken message · ${duration.toFixed(1)}s · ${count} camera frames`, 'user');
}
function ensureReply() {
  if (!replyBody) {
    const {body, label} = message('', 'assistant', 'LFM3 · streaming');
    replyBody = body; replyMeta = label;
  }
}
function playChunk(packet) {
  if (packet.byteLength < 6 || (packet.byteLength - 4) % 2) throw new Error('Invalid streamed audio packet.');
  const view = new DataView(packet);
  if (view.getUint32(0, true) !== activeTurn || !connected) return;
  ensureReply();
  const pcm = new Uint8Array(packet.slice(4)); replyChunks.push(pcm);
  const buffer = context.createBuffer(1, pcm.length / 2, 24000), data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = view.getInt16(4 + i * 2, true) / 32768;
  const node = context.createBufferSource(); node.buffer = buffer; node.connect(replyAnalyser);
  if (!playing.size) speakingSince = performance.now();
  playbackAt = Math.max(playbackAt, context.currentTime + 0.04);
  playing.add(node); node.start(playbackAt); playbackAt += buffer.duration;
  node.onended = () => {
    playing.delete(node); node.disconnect();
    if (!playing.size) { resumeAt = performance.now() + 350; listening(); }
  };
  status('Speaking...'); controls();
}
function saveReplay() {
  const length = replyChunks.reduce((n, part) => n + part.length, 0);
  if (!length) return;
  const header = new ArrayBuffer(44), view = new DataView(header);
  const word = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
  word(0, 'RIFF'); view.setUint32(4, 36 + length, true); word(8, 'WAVE'); word(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true); view.setUint32(28, 48000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  word(36, 'data'); view.setUint32(40, length, true);
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(new Blob([header, ...replyChunks], {type: 'audio/wav'}));
  $('reply').src = audioUrl;
}
async function connectSocket() {
  const target = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/live`);
  target.binaryType = 'arraybuffer'; socket = target; activeTurn = nextTurn = 0;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { reject(new Error('WebSocket connection timed out.')); target.close(); }, 15000);
    target.onmessage = event => {
      if (socket !== target) return;
      try {
        if (event.data instanceof ArrayBuffer) { playChunk(event.data); return; }
        const data = JSON.parse(event.data);
        if (data.type === 'ready') {
          clearTimeout(timeout);
          if (!data.ready) reject(new Error('Model is not loaded. Restart without --ui-only.'));
          else if (data.input_rate !== 16000 || data.output_rate !== 24000) reject(new Error('Unsupported audio sample rate.'));
          else {
            turnGate.configure(data.turn_detector || 'silence', data.pause_ms || 900, data.hold_ms || 2500);
            $('turnInfo').textContent = data.turn_detector === 'smart-turn-v3' ? 'Smart Turn v3' : 'Silence detection';
            resolve();
          }
          return;
        }
        if (data.id !== activeTurn) return;
        if (data.type === 'turn_decision' && recording && !manual) {
          const accepted = turnGate.pending === data.check;
          if (turnGate.decide(data.check, data.complete)) commitTurn();
          else if (accepted) { status('Take your time. Still listening...'); controls(); }
        }
        if (data.type === 'text') { ensureReply(); replyBody.textContent = data.text; }
        if (data.type === 'generating') status('Model is thinking...');
        if (data.type === 'done') {
          busy = false; ensureReply(); replyBody.textContent = data.text || '[Audio reply]';
          replyMeta.textContent = `LFM3 · ${data.seconds}s`; saveReplay(); listening();
        }
        if (data.type === 'reset') {
          resetting = false; busy = false; sessionId = crypto.randomUUID();
          clearMessages(); replyBody = replyMeta = null; clearReplay(); listening();
        }
        if (data.type === 'error') {
          busy = resetting = false; clearRecording(); stopPlayback();
          status(data.detail || 'The model request failed.', true); message(data.detail, 'error'); controls();
        }
      } catch (error) { reject(error); failConnection(error); }
    };
    target.onerror = () => { clearTimeout(timeout); reject(new Error('WebSocket connection failed.')); };
    target.onclose = () => {
      clearTimeout(timeout); reject(new Error('WebSocket connection closed.'));
      if (socket === target) { socket = null; failConnection(new Error('Connection closed. Start again to reconnect.')); }
    };
  });
}
async function connectSpeech() {
  const target = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/speech`);
  speechSocket = target;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { target.close(); reject(new Error('Speech detection timed out.')); }, 5000);
    let ready = false;
    target.onmessage = event => {
      const data = JSON.parse(event.data);
      if ('ready' in data) {
        clearTimeout(timeout);
        ready = data.ready;
        if (ready) resolve(); else reject(new Error('Speech detection is unavailable.'));
      } else if (Number.isFinite(data.probability)) {
        speechGate.update(data.probability, data.speech_score, performance.now());
      }
    };
    target.onerror = () => { clearTimeout(timeout); reject(new Error('Speech detection could not connect.')); };
    target.onclose = () => {
      clearTimeout(timeout);
      if (speechSocket !== target) return;
      speechSocket = null; speechGate.reset();
      $('automatic').checked = false; $('automatic').disabled = true;
      if (ready && connected) { status('Speech detection stopped. Use Speak to record.', true); controls(); }
      if (!ready) reject(new Error('Speech detection is unavailable.'));
    };
  });
}
function failConnection(error) {
  void stopMedia().then(() => status(error.message, true));
}
const workletCode = `
class MicCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(512); this.at = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (channels && channels.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let value = 0; for (const channel of channels) value += channel[i];
        this.buffer[this.at++] = value / channels.length;
        if (this.at === this.buffer.length) {
          this.port.postMessage(this.buffer, [this.buffer.buffer]);
          this.buffer = new Float32Array(512); this.at = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('mic-capture', MicCapture);`;

$('start').onclick = async () => {
  if (!window.isSecureContext || !navigator.mediaDevices) { status('Camera and mic require localhost or HTTPS.', true); return; }
  $('start').disabled = true; status('Connecting camera and microphone...');
  try {
    context = new AudioContext({sampleRate: 16000}); await context.resume();
    if (context.sampleRate !== 16000) throw new Error('This browser does not support a 16 kHz AudioContext.');
    await connectSocket();
    if (speechDetectionMode === 'classifier') {
      try { await connectSpeech(); $('automatic').disabled = false; }
      catch (_) { $('automatic').checked = false; $('automatic').disabled = true; }
    }
    noiseFloor = 0.002; speechWindow = [];
    stream = await navigator.mediaDevices.getUserMedia({
      video: $('useCamera').checked ? {width: {ideal: 640}, height: {ideal: 480}} : false,
      audio: {channelCount: 1, echoCancellation: {ideal: 'all'}, noiseSuppression: true, autoGainControl: true}
    });
    $('camera').srcObject = stream;
    const moduleUrl = URL.createObjectURL(new Blob([workletCode], {type: 'text/javascript'}));
    try { await context.audioWorklet.addModule(moduleUrl); } finally { URL.revokeObjectURL(moduleUrl); }
    source = context.createMediaStreamSource(stream); worklet = new AudioWorkletNode(context, 'mic-capture');
    micAnalyser = context.createAnalyser(); micAnalyser.fftSize = 256; source.connect(micAnalyser);
    replyAnalyser = context.createAnalyser(); replyAnalyser.fftSize = 256; replyAnalyser.connect(context.destination);
    worklet.port.onmessage = event => {
      try { onSamples(event.data); } catch (error) { failConnection(error); }
    };
    mute = context.createGain(); mute.gain.value = 0;
    source.connect(worklet); worklet.connect(mute); mute.connect(context.destination);
    connected = true;
    timer = setInterval(() => { try { captureFrame(); } catch (error) { failConnection(error); } }, 500);
    for (const track of stream.getTracks()) track.onended = () => failConnection(new Error('Camera or microphone disconnected.'));
    listening();
  } catch (error) { await stopMedia(); status('Could not start: ' + error.message, true); }
};
async function stopMedia() {
  connected = false; clearInterval(timer);
  const target = socket; socket = null; target?.close();
  const speechTarget = speechSocket; speechSocket = null; speechTarget?.close(); speechGate.reset();
  if (worklet) { worklet.port.onmessage = null; worklet.disconnect(); }
  source?.disconnect(); mute?.disconnect();
  micAnalyser?.disconnect(); replyAnalyser?.disconnect(); micAnalyser = replyAnalyser = null;
  for (const track of stream?.getTracks() || []) { track.onended = null; track.stop(); }
  stream = null; $('camera').srcObject = null; stopPlayback();
  const previous = context; context = null;
  if (previous && previous.state !== 'closed') await previous.close();
  source = worklet = mute = null; busy = resetting = false;
  clearRecording(); $('level').value = 0; controls();
}
$('stop').onclick = async () => { await stopMedia(); status('Stopped.'); };
$('talk').onclick = () => {
  try { if (recording) commitTurn(); else beginTurn(true); }
  catch (error) { failConnection(error); }
};
$('automatic').onchange = () => { if (!recording) listening(); };
$('useCamera').onchange = () => {
  for (const track of stream?.getVideoTracks() || []) track.enabled = $('useCamera').checked;
  controls();
  if (connected && $('useCamera').checked && !stream.getVideoTracks().length) status('Stop and restart to enable the camera.');
};
$('reply').onplay = () => {
  for (const node of playing) { node.onended = null; node.stop(); node.disconnect(); }
  playing.clear(); playbackAt = 0; preRoll = [];
  if (recording) { send({type: 'cancel'}); clearRecording(); }
  status('Playing reply...'); controls();
};
$('reply').onpause = $('reply').onended = () => { preRoll = []; resumeAt = performance.now() + 350; listening(); };
$('clear').onclick = async () => {
  if (resetting) return;
  try {
    cancelReply(); clearRecording();
    if (connected) {
      activeTurn = ++nextTurn; resetting = true; controls(); status('Clearing conversation...');
      send({type: 'reset', id: activeTurn, session_id: sessionId});
    } else {
      const response = await fetch('/api/session/' + sessionId, {method: 'DELETE'});
      if (!response.ok) throw new Error((await response.json()).detail);
      sessionId = crypto.randomUUID(); clearMessages(); clearReplay(); status('Conversation cleared.');
    }
  } catch (error) { resetting = false; status(error.message, true); controls(); }
};
window.addEventListener('pagehide', () => {
  cancelAnimationFrame(waveformFrame);
  socket?.close(); speechSocket?.close(); for (const track of stream?.getTracks() || []) track.stop();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
});
fetch('/api/status').then(r => r.json()).then(info => {
  modelReady = info.ready; maxSeconds = info.max_seconds;
  const mode = info.capabilities?.speech_detection;
  speechDetectionMode = ['volume', 'classifier', 'manual'].includes(mode) ? mode : 'manual';
  $('thresholdSetting').hidden = speechDetectionMode !== 'volume';
  $('automatic').disabled = speechDetectionMode === 'manual';
  if ($('automatic').disabled) $('automatic').checked = false;
  previewMode = !!info.preview; $('previewNotice').hidden = !previewMode;
  $('turnInfo').textContent = info.turn_detector === 'smart-turn-v3' ? 'Smart Turn' : 'Silence detection';
  $('modelInfo').textContent = info.backend === 'mlx' ? 'Running locally on the Apple GPU.' : 'Running locally.';
  status(info.ready ? 'Ready. Start your camera and microphone.' : 'UI-only mode. Restart without --ui-only to chat.');
  controls();
}).catch(error => status('Cannot reach server: ' + error.message, true));

// The waveform follows the microphone while listening and the actual scheduled
// reply audio while speaking. It needs no external fonts or browser model CDN.
const waveform = $('waveform'), drawing = waveform.getContext('2d');
function drawWaveform() {
  const width = waveform.clientWidth, height = waveform.clientHeight, ratio = window.devicePixelRatio || 1;
  if (waveform.width !== Math.round(width * ratio) || waveform.height !== Math.round(height * ratio)) {
    waveform.width = Math.round(width * ratio); waveform.height = Math.round(height * ratio);
  }
  drawing.setTransform(ratio, 0, 0, ratio, 0, 0); drawing.clearRect(0, 0, width, height);
  const analyser = playing.size ? replyAnalyser : micAnalyser;
  const bins = analyser ? new Uint8Array(analyser.frequencyBinCount) : null;
  if (bins) analyser.getByteFrequencyData(bins);
  drawing.fillStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim();
  const count = 44, gap = 4, barWidth = Math.max(2, (width - gap * (count - 1)) / count);
  for (let i = 0; i < count; i++) {
    const amplitude = bins ? bins[Math.floor(i * bins.length * 0.65 / count)] / 255 : 0;
    const barHeight = Math.max(3, amplitude * height * 0.9);
    drawing.globalAlpha = 0.35 + amplitude * 0.65;
    drawing.beginPath(); drawing.roundRect(i * (barWidth + gap), (height - barHeight) / 2, barWidth, barHeight, 2); drawing.fill();
  }
  drawing.globalAlpha = 1; waveformFrame = requestAnimationFrame(drawWaveform);
}
drawWaveform();
