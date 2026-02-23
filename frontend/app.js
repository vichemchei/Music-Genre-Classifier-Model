// ============================================================
// Config
// ============================================================
const API_BASE = 'http://localhost:5000';

const GENRE_EMOJIS = {
    blues: '🎷',
    classical: '🎻',
    country: '🤠',
    disco: '🪩',
    hiphop: '🎤',
    jazz: '🎹',
    metal: '🤘',
    pop: '🎵',
    reggae: '🌴',
    rock: '🎸',
};

const GENRE_COLORS = {
    blues: '#3b82f6',
    classical: '#a855f7',
    country: '#f59e0b',
    disco: '#ec4899',
    hiphop: '#ef4444',
    jazz: '#6366f1',
    metal: '#64748b',
    pop: '#06b6d4',
    reggae: '#22c55e',
    rock: '#f97316',
};


// ============================================================
// DOM References
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const statusBadge = $('#statusBadge');
const modeTabs = $$('.mode-tab');
const panels = $$('.input-panel');

// Upload
const dropZone = $('#dropZone');
const fileInput = $('#fileInput');
const fileInfo = $('#fileInfo');
const fileName = $('#fileName');
const fileSize = $('#fileSize');
const fileRemove = $('#fileRemove');
const btnPredict = $('#btnPredict');

// Record
const btnRecord = $('#btnRecord');
const recordLabel = $('#recordLabel');
const recordTimer = $('#recordTimer');
const timerText = $('#timerText');
const waveCanvas = $('#waveCanvas');

// System
const durationSlider = $('#durationSlider');
const durationValue = $('#durationValue');
const btnSystem = $('#btnSystem');

// Results
const resultsSection = $('#resultsSection');
const genreEmoji = $('#genreEmoji');
const genreName = $('#genreName');
const ringFill = $('#ringFill');
const confidenceValue = $('#confidenceValue');
const chartBars = $('#chartBars');

// Error
const errorToast = $('#errorToast');
const errorMessage = $('#errorMessage');
const errorClose = $('#errorClose');


// ============================================================
// State
// ============================================================
let selectedFile = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
let recordSeconds = 0;
let audioStream = null;
let audioContext = null;
let analyser = null;
let animFrameId = null;


// ============================================================
// Health Check
// ============================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            statusBadge.classList.add('online');
            statusBadge.querySelector('.status-text').textContent = 'Online';
        }
    } catch {
        statusBadge.classList.remove('online');
        statusBadge.querySelector('.status-text').textContent = 'Offline';
    }
}

checkHealth();
setInterval(checkHealth, 15000);


// ============================================================
// Tab Switching
// ============================================================
modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const mode = tab.dataset.mode;
        modeTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        panels.forEach(p => p.classList.remove('active'));
        $(`#panel${mode.charAt(0).toUpperCase() + mode.slice(1)}`).classList.add('active');
    });
});


// ============================================================
// File Upload
// ============================================================
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.style.display = 'flex';
    dropZone.style.display = 'none';
    btnPredict.disabled = false;
}

fileRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    dropZone.style.display = 'block';
    btnPredict.disabled = true;
});

btnPredict.addEventListener('click', async () => {
    if (!selectedFile) return;
    setLoading(btnPredict, true);
    try {
        const form = new FormData();
        form.append('file', selectedFile);
        const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form });
        const data = await res.json();
        if (res.ok) {
            showResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('Could not reach the server. Is it running?');
    } finally {
        setLoading(btnPredict, false);
    }
});


// ============================================================
// Microphone Recording
// ============================================================
btnRecord.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});

async function startRecording() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(audioStream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            clearInterval(recordingTimer);
            cancelAnimationFrame(animFrameId);

            waveCanvas.style.display = 'none';
            recordTimer.style.display = 'none';
            btnRecord.classList.remove('recording');
            recordLabel.textContent = 'Processing...';

            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            const bytes = await blob.arrayBuffer();

            try {
                const res = await fetch(`${API_BASE}/predict/record`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/octet-stream' },
                    body: bytes,
                });
                const data = await res.json();
                if (res.ok) {
                    showResults(data);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            } catch {
                showError('Could not reach the server.');
            }

            recordLabel.textContent = 'Click to start recording';
            cleanupAudio();
        };

        mediaRecorder.start();
        btnRecord.classList.add('recording');
        recordLabel.textContent = 'Recording... click to stop';
        recordSeconds = 0;
        timerText.textContent = '00:00';
        recordTimer.style.display = 'flex';

        // Timer
        recordingTimer = setInterval(() => {
            recordSeconds++;
            const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
            const s = String(recordSeconds % 60).padStart(2, '0');
            timerText.textContent = `${m}:${s}`;
        }, 1000);

        // Waveform
        startWaveform(audioStream);

    } catch (err) {
        showError('Microphone access denied. Please allow microphone access.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}

function cleanupAudio() {
    if (audioStream) {
        audioStream.getTracks().forEach(t => t.stop());
        audioStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

function startWaveform(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    waveCanvas.style.display = 'block';
    const ctx = waveCanvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        animFrameId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        const w = waveCanvas.width;
        const h = waveCanvas.height;
        ctx.clearRect(0, 0, w, h);

        const barW = (w / bufferLength) * 2.5;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const barH = (dataArray[i] / 255) * h;
            const hue = 260 + (i / bufferLength) * 60;
            ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.8)`;
            ctx.fillRect(x, h - barH, barW - 1, barH);
            x += barW;
        }
    }
    draw();
}


// ============================================================
// System Audio
// ============================================================
durationSlider.addEventListener('input', () => {
    durationValue.textContent = `${durationSlider.value}s`;
});

btnSystem.addEventListener('click', async () => {
    const duration = parseInt(durationSlider.value);
    setLoading(btnSystem, true);
    btnSystem.querySelector('.btn-text').textContent = `Listening for ${duration}s...`;

    try {
        const res = await fetch(`${API_BASE}/predict/system`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ duration }),
        });
        const data = await res.json();
        if (res.ok) {
            showResults(data);
        } else {
            showError(data.error || 'System audio capture failed');
        }
    } catch {
        showError('Could not reach the server. Is it running?');
    } finally {
        setLoading(btnSystem, false);
        btnSystem.querySelector('.btn-text').textContent = 'Start Listening';
    }
});


// ============================================================
// Results Display
// ============================================================
function showResults(data) {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Main genre
    const genre = data.genre;
    genreEmoji.textContent = GENRE_EMOJIS[genre] || '🎵';
    genreEmoji.style.animation = 'none';
    genreEmoji.offsetHeight; // reflow
    genreEmoji.style.animation = 'bounceIn 0.5s ease';

    genreName.textContent = genre;

    // Confidence ring
    const pct = Math.round(data.confidence * 100);
    confidenceValue.textContent = pct;
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference * (1 - data.confidence);
    ringFill.style.strokeDashoffset = offset;

    // Inject SVG gradient if not already present
    if (!document.getElementById('ringGradient')) {
        const svg = document.querySelector('.confidence-ring');
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#6c5ce7"/>
                <stop offset="100%" stop-color="#a855f7"/>
            </linearGradient>
        `;
        svg.prepend(defs);
    }

    // Bar chart
    chartBars.innerHTML = '';
    const topGenres = data.top_genres || [];
    const maxConf = topGenres.length > 0 ? topGenres[0].confidence : 1;

    topGenres.forEach((item, i) => {
        const row = document.createElement('div');
        row.className = `chart-row${i === 0 ? ' top' : ''}`;
        row.style.animationDelay = `${i * 0.05}s`;

        const barPct = maxConf > 0 ? (item.confidence / maxConf) * 100 : 0;
        const displayPct = (item.confidence * 100).toFixed(1);

        row.innerHTML = `
            <span class="chart-label">${GENRE_EMOJIS[item.genre] || '🎵'} ${item.genre}</span>
            <div class="chart-bar-wrap">
                <div class="chart-bar${i === 0 ? ' top' : ''}" style="width: 0%"></div>
            </div>
            <span class="chart-percent">${displayPct}%</span>
        `;
        chartBars.appendChild(row);

        // Animate bar width
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                row.querySelector('.chart-bar').style.width = `${barPct}%`;
            });
        });
    });
}


// ============================================================
// Error Handling
// ============================================================
function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.style.display = 'flex';
    errorToast.style.animation = 'none';
    errorToast.offsetHeight;
    errorToast.style.animation = 'toastIn 0.4s ease';

    setTimeout(() => {
        errorToast.style.display = 'none';
    }, 6000);
}

errorClose.addEventListener('click', () => {
    errorToast.style.display = 'none';
});


// ============================================================
// Utilities
// ============================================================
function setLoading(btn, loading) {
    if (loading) {
        btn.classList.add('loading');
        btn.disabled = true;
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}
