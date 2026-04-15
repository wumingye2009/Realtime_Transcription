const stateEl = document.querySelector("#session-state");
const warningListEl = document.querySelector("#warnings");
const outputSelectEl = document.querySelector("#system-output-device");
const micCheckboxEl = document.querySelector("#microphone-enabled");
const micSelectEl = document.querySelector("#microphone-input-device");
const transcriptEl = document.querySelector("#transcript");
const processingStatusEl = document.querySelector("#processing-status");
const formEl = document.querySelector("#session-form");
const startBtnEl = document.querySelector("#start-btn");
const stopBtnEl = document.querySelector("#stop-btn");
const pauseBtnEl = document.querySelector("#pause-btn");
const resumeBtnEl = document.querySelector("#resume-btn");
let statusPollHandle = null;
let controlRequestInFlight = false;

function formatTimestamp(value) {
  const seconds = Math.floor(value % 60).toString().padStart(2, "0");
  const minutes = Math.floor((value / 60) % 60).toString().padStart(2, "0");
  const hours = Math.floor(value / 3600).toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function setState(nextState) {
  stateEl.textContent = nextState;
  updateControlAvailability(nextState);
}

function renderWarnings(warnings) {
  warningListEl.innerHTML = "";
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = warning;
    warningListEl.appendChild(item);
  }
}

function populateSelect(selectEl, devices, placeholder) {
  selectEl.innerHTML = "";

  if (!devices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = placeholder;
    selectEl.appendChild(option);
    return;
  }

  for (const device of devices) {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = formatDeviceLabel(device);
    selectEl.appendChild(option);
  }
}

function formatDeviceLabel(device) {
  const parts = [];
  parts.push(device.is_default ? `${device.name} (default)` : device.name);
  if (device.description) {
    parts.push(device.description);
  }
  if (device.hostapi) {
    parts.push(device.hostapi);
  }
  return parts.join(" - ");
}

async function loadDevices() {
  const response = await fetch("/api/devices");
  const data = await response.json();
  populateSelect(outputSelectEl, data.system_output_devices, "No output devices found");
  populateSelect(micSelectEl, data.microphone_input_devices, "No microphone devices found");
  renderWarnings(data.warnings || []);
}

function renderTranscript(segments) {
  if (!segments.length) {
    transcriptEl.innerHTML = "<p>No transcript segments yet.</p>";
    return;
  }

  transcriptEl.innerHTML = "";
  for (const segment of segments) {
    const block = document.createElement("div");
    block.className = "segment";
    block.innerHTML = `
      <div class="timestamp">[${formatTimestamp(segment.start)} - ${formatTimestamp(segment.end)}]</div>
      <div>${segment.text}</div>
    `;
    transcriptEl.appendChild(block);
  }
}

function renderProcessingStatus(state, diagnostics) {
  if (!processingStatusEl) {
    return;
  }

  if (state !== "stopping") {
    processingStatusEl.textContent = "";
    return;
  }

  const queued = diagnostics?.queued_audio_chunks ?? 0;
  const producerFinished = diagnostics?.producer_finished ? "capture stopped" : "stopping capture";
  const finalizing = diagnostics?.finalize_complete ? "finalizing complete" : "processing remaining audio";
  processingStatusEl.textContent = `Stopping... ${finalizing}. Queue backlog: ${queued} chunk(s). ${producerFinished}.`;
}

function collectPayload() {
  const exportTxt = document.querySelector("#export-txt").checked;
  return {
    system_output_device_id: outputSelectEl.value,
    microphone_enabled: micCheckboxEl.checked,
    microphone_input_device_id: micCheckboxEl.checked ? micSelectEl.value : null,
    language_mode: document.querySelector("#language-mode").value,
    output_dir: document.querySelector("#output-dir").value,
    export_formats: exportTxt ? ["md", "txt"] : ["md"],
  };
}

async function postControl(path, payload = undefined) {
  if (controlRequestInFlight) {
    return null;
  }

  controlRequestInFlight = true;
  setControlsDisabled(true);

  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : null,
  });

  const data = await response.json();
  controlRequestInFlight = false;
  setControlsDisabled(false);

  if (!response.ok) {
    alert(data.detail || "Request failed");
    return null;
  }

  setState(data.state);
  return data;
}

function setControlsDisabled(isDisabled) {
  startBtnEl.disabled = isDisabled;
  stopBtnEl.disabled = isDisabled;
  pauseBtnEl.disabled = isDisabled;
  resumeBtnEl.disabled = isDisabled;
}

function updateControlAvailability(state) {
  if (controlRequestInFlight) {
    return;
  }

  if (state === "running") {
    startBtnEl.disabled = true;
    stopBtnEl.disabled = false;
    pauseBtnEl.disabled = false;
    resumeBtnEl.disabled = true;
    return;
  }

  if (state === "paused") {
    startBtnEl.disabled = true;
    stopBtnEl.disabled = false;
    pauseBtnEl.disabled = true;
    resumeBtnEl.disabled = false;
    return;
  }

  if (state === "stopping") {
    startBtnEl.disabled = true;
    stopBtnEl.disabled = true;
    pauseBtnEl.disabled = true;
    resumeBtnEl.disabled = true;
    return;
  }

  startBtnEl.disabled = false;
  stopBtnEl.disabled = true;
  pauseBtnEl.disabled = true;
  resumeBtnEl.disabled = true;
}

micCheckboxEl.addEventListener("change", () => {
  micSelectEl.disabled = !micCheckboxEl.checked;
});

document.querySelector("#pause-btn").addEventListener("click", async () => {
  await postControl("/api/sessions/pause");
});

document.querySelector("#resume-btn").addEventListener("click", async () => {
  await postControl("/api/sessions/resume");
});

document.querySelector("#stop-btn").addEventListener("click", async () => {
  await postControl("/api/sessions/stop");
  await loadStatus();
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  transcriptEl.innerHTML = "<p>Waiting for transcript segments...</p>";
  await postControl("/api/sessions/start", collectPayload());
  await loadStatus();
});

async function loadStatus() {
  const response = await fetch("/api/sessions/current");
  const data = await response.json();
  setState(data.state);
  renderProcessingStatus(data.state, data.diagnostics || {});
  renderTranscript(data.transcript_segments || []);
}

function startStatusPolling() {
  if (statusPollHandle !== null) {
    window.clearInterval(statusPollHandle);
  }

  statusPollHandle = window.setInterval(async () => {
    await loadStatus();
  }, 1000);
}

window.addEventListener("DOMContentLoaded", async () => {
  micSelectEl.disabled = !micCheckboxEl.checked;
  await loadDevices();
  await loadStatus();
  startStatusPolling();
});
