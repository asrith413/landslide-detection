import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

type SegmentResponse = {
  width: number;
  height: number;
  coverage: number;
  latencyMs: number;
  tiles: number;
  originalPng: string;
  maskPng: string;
  overlayPng: string;
};

type HistoryItem = {
  file: string;
  coverage: number;
  latencyMs: number;
  threshold: number;
  createdAt: number;
};

const modelLabel = document.querySelector<HTMLSpanElement>("#model-label")!;
const imageLabel = document.querySelector<HTMLSpanElement>("#image-label")!;
const threshold = document.querySelector<HTMLInputElement>("#threshold")!;
const thresholdValue = document.querySelector<HTMLOutputElement>("#threshold-value")!;
const runButton = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLSpanElement>("#status")!;
const metrics = document.querySelector<HTMLElement>("#metrics")!;
const views = document.querySelector<HTMLElement>("#views")!;
const historyList = document.querySelector<HTMLUListElement>("#history-list")!;

let modelPath = "";
let imagePath = "";
const historyKey = "landslide-history-v1";

function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function setStatus(text: string): void {
  status.textContent = text;
}

function refreshRunState(): void {
  runButton.disabled = !modelPath || !imagePath;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(historyKey);
    return raw ? (JSON.parse(raw) as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(items: HistoryItem[]): void {
  localStorage.setItem(historyKey, JSON.stringify(items.slice(0, 30)));
}

function renderHistory(): void {
  const items = loadHistory();
  historyList.replaceChildren();
  for (const item of items) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = item.file;
    const coverage = document.createElement("span");
    coverage.textContent = formatPercent(item.coverage);
    const latency = document.createElement("span");
    latency.textContent = `${item.latencyMs.toFixed(0)} ms`;
    const when = document.createElement("span");
    when.textContent = new Date(item.createdAt).toLocaleString();
    li.append(name, coverage, latency, when);
    historyList.append(li);
  }
}

async function pickModel(): Promise<void> {
  const selected = await open({
    multiple: false,
    filters: [{ name: "ONNX model", extensions: ["onnx"] }]
  });
  if (typeof selected === "string") {
    modelPath = selected;
    modelLabel.textContent = fileName(selected);
    refreshRunState();
  }
}

async function pickImage(): Promise<void> {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"] }]
  });
  if (typeof selected === "string") {
    imagePath = selected;
    imageLabel.textContent = fileName(selected);
    refreshRunState();
  }
}

async function run(): Promise<void> {
  if (!modelPath || !imagePath) {
    return;
  }
  runButton.disabled = true;
  setStatus("Running inference...");
  try {
    const result = await invoke<SegmentResponse>("segment_image", {
      modelPath,
      imagePath,
      threshold: Number(threshold.value)
    });
    document.querySelector<HTMLImageElement>("#view-original")!.src = result.originalPng;
    document.querySelector<HTMLImageElement>("#view-overlay")!.src = result.overlayPng;
    document.querySelector<HTMLImageElement>("#view-mask")!.src = result.maskPng;
    document.querySelector<HTMLElement>("#m-coverage")!.textContent = formatPercent(result.coverage);
    document.querySelector<HTMLElement>("#m-latency")!.textContent = `${result.latencyMs.toFixed(0)} ms`;
    document.querySelector<HTMLElement>("#m-tiles")!.textContent = String(result.tiles);
    document.querySelector<HTMLElement>("#m-size")!.textContent = `${result.width} x ${result.height}`;
    metrics.hidden = false;
    views.hidden = false;
    setStatus("Done");
    const items = loadHistory();
    items.unshift({
      file: fileName(imagePath),
      coverage: result.coverage,
      latencyMs: result.latencyMs,
      threshold: Number(threshold.value),
      createdAt: Date.now()
    });
    saveHistory(items);
    renderHistory();
  } catch (error) {
    setStatus(`Error: ${String(error)}`);
  } finally {
    refreshRunState();
  }
}

document.querySelector<HTMLButtonElement>("#pick-model")!.addEventListener("click", pickModel);
document.querySelector<HTMLButtonElement>("#pick-image")!.addEventListener("click", pickImage);
runButton.addEventListener("click", run);
threshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(threshold.value).toFixed(2);
});
document.querySelector<HTMLButtonElement>("#clear-history")!.addEventListener("click", () => {
  saveHistory([]);
  renderHistory();
});

thresholdValue.textContent = Number(threshold.value).toFixed(2);
renderHistory();
refreshRunState();
