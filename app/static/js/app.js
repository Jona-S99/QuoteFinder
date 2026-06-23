const state = {
  dbReady: false,
  selectedFiles: [],
};

const els = {
  chatArea: document.querySelector(".chat-area"),
  chatLock: document.querySelector("#chat-lock"),
  dbStatus: document.querySelector("#database-status"),
  fileDrop: document.querySelector("#file-drop"),
  documents: document.querySelector("#documents"),
  fileList: document.querySelector("#file-list"),
  uploadButton: document.querySelector("#upload-button"),
  convertButton: document.querySelector("#convert-button"),
  buildButton: document.querySelector("#build-button"),
  cleanButton: document.querySelector("#clean-button"),
  refreshModels: document.querySelector("#refresh-models"),
  embeddingModel: document.querySelector("#embedding-model"),
  parentSize: document.querySelector("#parent-size"),
  parentOverlap: document.querySelector("#parent-overlap"),
  childSize: document.querySelector("#child-size"),
  childOverlap: document.querySelector("#child-overlap"),
  topK: document.querySelector("#top-k"),
  topRanked: document.querySelector("#top-ranked"),
  llmModel: document.querySelector("#llm-model"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chat-form"),
  query: document.querySelector("#query"),
  sendButton: document.querySelector("#send-button"),
  retrievedCount: document.querySelector("#retrieved-count"),
  rerankedCount: document.querySelector("#reranked-count"),
  activityLog: document.querySelector("#activity-log"),
};

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
    button.disabled = true;
    return;
  }
  button.textContent = button.dataset.label || button.textContent;
  button.disabled = false;
}

function addLog(message) {
  const item = document.createElement("li");
  item.textContent = message;
  els.activityLog.prepend(item);
}

function setDbReady(ready) {
  state.dbReady = ready;
  els.chatArea.classList.toggle("locked", !ready);
  els.chatLock.classList.toggle("hidden", ready);
  els.query.disabled = !ready;
  els.sendButton.disabled = !ready;
  els.dbStatus.textContent = ready ? "Base lista" : "Sin base";
  els.dbStatus.classList.toggle("ready", ready);
}

function renderFiles() {
  els.fileList.replaceChildren();

  if (state.selectedFiles.length === 0) {
    const empty = document.createElement("span");
    empty.textContent = "No hay archivos seleccionados";
    els.fileList.append(empty);
    return;
  }

  state.selectedFiles.forEach((file) => {
    const pill = document.createElement("span");
    pill.textContent = file.name;
    els.fileList.append(pill);
  });
}

function setSelectedFiles(files) {
  const allowedExtensions = [".pdf", ".docx", ".md"];
  state.selectedFiles = Array.from(files).filter((file) => {
    const filename = file.name.toLowerCase();
    return allowedExtensions.some((extension) => filename.endsWith(extension));
  });
  renderFiles();

  if (files.length > 0 && state.selectedFiles.length === 0) {
    addLog("Arrastra solo archivos PDF, DOCX o Markdown.");
  }
}

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(paragraph);
  els.messages.append(article);
  els.messages.scrollTop = els.messages.scrollHeight;
  return article;
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return { detail: await response.text() };
}

function errorText(error) {
  if (typeof error === "string") return error;
  if (error?.detail) return error.detail;
  if (error?.message) return error.message;
  return "La accion no se pudo completar.";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await parseResponse(response);

  if (!response.ok) {
    throw new Error(errorText(data));
  }

  return data;
}

async function loadModels() {
  setBusy(els.refreshModels, true, "...");
  try {
    const data = await api("/api/v1/vector-db/embedding-models");
    const current = els.embeddingModel.value;

    const models = data.models.length > 0 ? data.models : [current];
    els.embeddingModel.replaceChildren();

    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      els.embeddingModel.append(option);
    });

    if (models.includes(current)) {
      els.embeddingModel.value = current;
    }

    addLog(`${models.length} modelos de embedding disponibles.`);
  } catch (error) {
    addLog(errorText(error));
  } finally {
    setBusy(els.refreshModels, false);
  }
}

async function uploadFiles() {
  if (state.selectedFiles.length === 0) {
    addLog("Selecciona al menos un archivo para subir.");
    return;
  }

  const formData = new FormData();
  state.selectedFiles.forEach((file) => formData.append("files", file));

  setBusy(els.uploadButton, true, "Subiendo");
  try {
    const data = await api("/api/v1/vector-db/upload", {
      method: "POST",
      body: formData,
    });
    addLog(`${data.message}. ${data.raw_files.length} archivos en raw.`);
  } catch (error) {
    addLog(errorText(error));
  } finally {
    setBusy(els.uploadButton, false);
  }
}

async function convertFiles() {
  setBusy(els.convertButton, true, "Convirtiendo");
  try {
    const data = await api("/api/v1/vector-db/convert", { method: "POST" });
    addLog(`${data.converted_count} documentos convertidos a Markdown.`);
  } catch (error) {
    addLog(errorText(error));
  } finally {
    setBusy(els.convertButton, false);
  }
}

function buildPayload() {
  return {
    embedding_model: els.embeddingModel.value.trim(),
    parent_size: Number(els.parentSize.value),
    parent_overlap: Number(els.parentOverlap.value),
    child_size: Number(els.childSize.value),
    child_overlap: Number(els.childOverlap.value),
  };
}

async function buildDb() {
  setBusy(els.buildButton, true, "Creando");
  try {
    const data = await api("/api/v1/vector-db/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    setDbReady(true);
    addLog(
      `${data.documents_count} documentos, ${data.chunks_count} chunks, ${data.parents_count} parents.`
    );
  } catch (error) {
    setDbReady(false);
    addLog(errorText(error));
  } finally {
    setBusy(els.buildButton, false);
  }
}

async function cleanDb() {
  setBusy(els.cleanButton, true, "Limpiando");
  try {
    const data = await api("/api/v1/vector-db/clean", { method: "POST" });
    setDbReady(false);
    els.retrievedCount.textContent = "0";
    els.rerankedCount.textContent = "0";
    addLog(data.message);
  } catch (error) {
    addLog(errorText(error));
  } finally {
    setBusy(els.cleanButton, false);
  }
}

async function askQuestion(event) {
  event.preventDefault();

  const query = els.query.value.trim();
  if (!query || !state.dbReady) return;

  appendMessage("user", query);
  els.query.value = "";
  const pending = appendMessage("assistant", "Buscando citas...");
  els.sendButton.disabled = true;

  try {
    const data = await api("/api/v1/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        embedding_model: els.embeddingModel.value.trim(),
        top_k: Number(els.topK.value),
        top_ranked: Number(els.topRanked.value),
        llm_model: els.llmModel.value.trim(),
      }),
    });

    pending.querySelector("p").textContent = data.answer;
    els.retrievedCount.textContent = data.retrieved_chunks_count;
    els.rerankedCount.textContent = data.reranked_chunks_count;
    addLog(`Consulta lista: ${data.reranked_chunks_count} fragmentos finales.`);
  } catch (error) {
    pending.classList.add("error");
    pending.querySelector("p").textContent = errorText(error);
  } finally {
    els.sendButton.disabled = false;
    els.query.focus();
  }
}

els.documents.addEventListener("change", () => {
  setSelectedFiles(els.documents.files);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.fileDrop.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.fileDrop.classList.remove("dragging");
  });
});

els.fileDrop.addEventListener("drop", (event) => {
  setSelectedFiles(event.dataTransfer.files);
});

els.uploadButton.addEventListener("click", uploadFiles);
els.convertButton.addEventListener("click", convertFiles);
els.buildButton.addEventListener("click", buildDb);
els.cleanButton.addEventListener("click", cleanDb);
els.refreshModels.addEventListener("click", loadModels);
els.chatForm.addEventListener("submit", askQuestion);
els.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    els.chatForm.requestSubmit();
  }
});

renderFiles();
setDbReady(false);
loadModels();
