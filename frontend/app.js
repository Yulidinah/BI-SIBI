const video          = document.getElementById("video");
const charResult     = document.getElementById("charResult");
const resultSub      = document.getElementById("resultSub");
const sentenceResult = document.getElementById("sentenceResult");
const startBtn       = document.getElementById("startBtn");
const stopBtn        = document.getElementById("stopBtn");
const spaceBtn       = document.getElementById("spaceBtn");
const backspaceBtn   = document.getElementById("backspaceBtn");
const clearBtn       = document.getElementById("clearBtn");
const modeAutoBtn    = document.getElementById("modeAutoBtn");
const modeHurufBtn   = document.getElementById("modeHurufBtn");
const modeKataBtn    = document.getElementById("modeKataBtn");
const modeLabel      = document.getElementById("modeLabel");
const liveBadge      = document.getElementById("liveBadge");
const scanLine       = document.getElementById("scanLine");
const vOverlay       = document.getElementById("vOverlay");
const progFill       = document.getElementById("progFill");
const progText       = document.getElementById("progText");
const statusBadge    = document.getElementById("statusBadge");
const statusText     = document.getElementById("statusText");

let detectionInterval = null;
let currentSentence   = "";
let detectionMode     = "auto";
let isLocked          = false;
let displayLockTimer  = null;
let isSending         = false;

const CANVAS_W    = 480;
const CANVAS_H    = 360;
const smallCanvas = document.createElement("canvas");
smallCanvas.width  = CANVAS_W;
smallCanvas.height = CANVAS_H;
const sCtx = smallCanvas.getContext("2d");

// ── UI helpers ──
function setProgress(pct, label) {
    if (progFill) progFill.style.width = Math.min(pct, 100) + "%";
    if (progText) progText.textContent = label || "—";
}

function setChar(text, active = false) {
    charResult.textContent = text;
    charResult.classList.toggle("active", active);
}

function setSub(text) {
    if (resultSub) resultSub.textContent = text;
}

function setStatus(text, live = false) {
    if (statusText) statusText.textContent = text;
    if (statusBadge) statusBadge.classList.toggle("live", live);
}

// ── Kamera ──
navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } }
})
.then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        setChar("—");
        setSub("Klik Mulai Deteksi untuk memulai");
    };
})
.catch(() => {
    setChar("!");
    setSub("Akses kamera ditolak — izinkan di browser");
});

// ── Start / Stop ──
startBtn.addEventListener("click", () => {
    startBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    liveBadge.classList.add("show");
    scanLine.classList.add("on");
    vOverlay.classList.add("hidden");
    setChar("—");
    setSub("Menginisialisasi…");
    setProgress(0, "Memulai deteksi…");
    setStatus("Mendeteksi", true);
    isSending = false;
    detectionInterval = setInterval(sendFrame, 100);
});

stopBtn.addEventListener("click", () => {
    stopBtn.classList.add("hidden");
    startBtn.classList.remove("hidden");
    liveBadge.classList.remove("show");
    scanLine.classList.remove("on");
    vOverlay.classList.remove("hidden");
    clearInterval(detectionInterval);
    detectionInterval = null;
    isSending = false;
    setChar("—");
    setSub("Deteksi dihentikan");
    setProgress(0, "—");
    setStatus("Siap");
});

// ── Mode ──
function setMode(mode) {
    detectionMode = mode;
    [modeAutoBtn, modeHurufBtn, modeKataBtn].forEach(b => b.classList.remove("active"));
    const map = {
        auto:  { btn: modeAutoBtn,  label: "AUTO" },
        huruf: { btn: modeHurufBtn, label: "HURUF & ANGKA" },
        kata:  { btn: modeKataBtn,  label: "KATA" },
    };
    map[mode].btn.classList.add("active");
    modeLabel.textContent = map[mode].label;
    isLocked = false;
    setProgress(0, "Mode berganti — buffer direset");
    fetch("http://127.0.0.1:8000/reset_buffer", { method: "POST" }).catch(() => {});
}

modeAutoBtn.addEventListener("click",  () => setMode("auto"));
modeHurufBtn.addEventListener("click", () => setMode("huruf"));
modeKataBtn.addEventListener("click",  () => setMode("kata"));

// ── Kalimat ──
spaceBtn.addEventListener("click", () => {
    if (currentSentence === "…") currentSentence = "";
    currentSentence += " ";
    sentenceResult.textContent = currentSentence;
});

backspaceBtn.addEventListener("click", () => {
    currentSentence = currentSentence.trimEnd().slice(0, -1);
    sentenceResult.textContent = currentSentence || "…";
});

clearBtn.addEventListener("click", async () => {
    currentSentence = "";
    sentenceResult.textContent = "…";
    isLocked = false;
    setChar("—");
    setSub("Kalimat dihapus");
    setProgress(0, "—");
    try { await fetch("http://127.0.0.1:8000/reset_buffer", { method: "POST" }); } catch {}
});

// ── Kirim frame ──
async function sendFrame() {
    if (isSending) return;
    if (!video.videoWidth) return;
    isSending = true;
    try {
        sCtx.setTransform(-1, 0, 0, 1, CANVAS_W, 0);
        sCtx.drawImage(video, 0, 0, CANVAS_W, CANVAS_H);
        sCtx.setTransform(1, 0, 0, 1, 0, 0);
        const image = smallCanvas.toDataURL("image/jpeg", 0.6);
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image, mode: detectionMode }),
            signal: AbortSignal.timeout(8000)
        });
        const data = await response.json();
        handlePrediction(data);
    } catch (err) {
        if (err.name !== "AbortError") {
            setChar("!");
            setSub("Server tidak terhubung");
            setProgress(0, "Pastikan backend berjalan");
        }
    } finally {
        isSending = false;
    }
}

// ── Handle prediksi ──
function handlePrediction(data) {
    const pred = data.prediction || "";

    if (pred.includes("Mengumpulkan")) {
        const match = pred.match(/(\d+)\/(\d+)/);
        if (match) {
            const cur = parseInt(match[1]);
            const max = parseInt(match[2]);
            if (cur > max) return;
            setProgress(Math.round((cur / max) * 100), pred);
            if (!displayLockTimer) { setChar("…"); setSub("Mengumpulkan frame gestur…"); }
        }
        return;
    }

    if (pred === "Menunggu Transisi Gerakan...") {
        setProgress(0, "Jeda transisi");
        if (!displayLockTimer) { setChar("—"); setSub("Siapkan gestur berikutnya"); }
        return;
    }

    if (!data.detected || pred === "-" || pred === "") {
        setProgress(0, "—");
        if (!displayLockTimer) { setChar("—"); setSub("Arahkan tangan ke kamera"); }
        return;
    }

    // Hasil valid
    setChar(pred, true);
    setSub("Terdeteksi");
    setProgress(100, "Prediksi berhasil ✓");

    if (displayLockTimer) clearTimeout(displayLockTimer);
    displayLockTimer = setTimeout(() => {
        setChar("—", false);
        setSub("Menunggu gestur berikutnya…");
        setProgress(0, "Siap mendeteksi");
        displayLockTimer = null;
    }, 2000);

    if (!isLocked) {
        isLocked = true;
        if (currentSentence === "…") currentSentence = "";
        if (pred.length > 1 && currentSentence.length > 0 && !currentSentence.endsWith(" ")) {
            currentSentence += " ";
        }
        currentSentence += pred;
        sentenceResult.textContent = currentSentence;
        const lockDuration = pred.length > 1 ? 4500 : 3500;
        setTimeout(() => { isLocked = false; }, lockDuration);
    }
}