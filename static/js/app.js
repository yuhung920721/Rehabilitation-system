const actionText = document.querySelector("#actionText");
const feedbackText = document.querySelector("#feedbackText");
const totalCount = document.querySelector("#totalCount");
const targetCount = document.querySelector("#targetCount");
const scoreText = document.querySelector("#scoreText");
const angleText = document.querySelector("#angleText");
const stageText = document.querySelector("#stageText");
const progressText = document.querySelector("#progressText");
const progressFill = document.querySelector("#progressFill");
const s1Lamp = document.querySelector("#s1Lamp");
const s2Lamp = document.querySelector("#s2Lamp");
const connectionStatus = document.querySelector("#connectionStatus");
const targetInput = document.querySelector("#targetInput");
const setTargetButton = document.querySelector("#setTargetButton");
const resetButton = document.querySelector("#resetButton");

function setLamp(element, isDone) {
  element.classList.toggle("done", Boolean(isDone));
}

function updateStatus(status) {
  actionText.textContent = status.action_text ?? "--";
  feedbackText.textContent = status.feedback ?? "";
  totalCount.textContent = status.total ?? 0;
  targetCount.textContent = status.target_count ?? 5;
  scoreText.textContent = Number(status.score ?? 0).toFixed(1);
  angleText.textContent = status.angle === null || status.angle === undefined
    ? "--"
    : `${Number(status.angle).toFixed(1)}°`;
  stageText.textContent = status.rep_stage ?? "--";

  const progress = Number(status.progress ?? 0);
  progressText.textContent = `${progress.toFixed(1)}%`;
  progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;

  setLamp(s1Lamp, status.s1_done);
  setLamp(s2Lamp, status.s2_done);

  if (document.activeElement !== targetInput) {
    targetInput.value = status.target_count ?? 5;
  }

  if (status.model_ready) {
    connectionStatus.textContent = status.is_finished ? "訓練完成" : "偵測中";
    connectionStatus.className = "status-pill ready";
  } else {
    connectionStatus.textContent = "缺少模型";
    connectionStatus.className = "status-pill warning";
  }
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    updateStatus(await response.json());
  } catch (error) {
    connectionStatus.textContent = "連線中斷";
    connectionStatus.className = "status-pill warning";
  }
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.ok) {
    updateStatus(await response.json());
  }
}

setTargetButton.addEventListener("click", () => {
  postJson("/api/target", { target_count: Number(targetInput.value || 5) });
});

resetButton.addEventListener("click", () => {
  postJson("/api/reset");
});

fetchStatus();
setInterval(fetchStatus, 200);
