const input = document.getElementById('descInput');
const noiseScaleInput = document.getElementById('noiseScale');
const noiseValueDisplay = document.getElementById('noiseValue');
const generateBtn = document.getElementById('generateBtn');
const micBtn = document.getElementById('micBtn');
const resultImage = document.getElementById('resultImage');
const placeholder = document.getElementById('placeholder');
const loader = document.getElementById('loader');
const actions = document.getElementById('actions');
const downloadBtn = document.getElementById('downloadBtn');
const matchBtn = document.getElementById('matchBtn');
const matchOverlay = document.getElementById('matchOverlay');
const listeningIndicator = document.getElementById('listeningIndicator');

// State
let currentSketchBase64 = null;

// Noise Scale Slider
noiseScaleInput.addEventListener('input', (e) => {
    noiseValueDisplay.textContent = e.target.value;
});

function toggleOptions() {
    document.getElementById('advancedOptions').classList.toggle('hidden');
}

// Voice Recognition
if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';

    micBtn.addEventListener('click', () => {
        if (micBtn.classList.contains('recording')) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });

    recognition.onstart = () => {
        micBtn.classList.add('recording');
        listeningIndicator.classList.remove('hidden');
    };

    recognition.onend = () => {
        micBtn.classList.remove('recording');
        listeningIndicator.classList.add('hidden');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
    };
} else {
    micBtn.style.display = 'none'; // Not supported
}

// Generate
generateBtn.addEventListener('click', async () => {
    const desc = input.value.trim();
    if (!desc) return alert("Please enter a description or use voice input.");

    // UI Loading
    loader.classList.remove('hidden');
    placeholder.classList.add('hidden');
    resultImage.classList.add('hidden');
    actions.classList.add('hidden');

    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: desc,
                noise_scale: parseFloat(noiseScaleInput.value)
            })
        });

        const data = await res.json();

        // Show Result
        currentSketchBase64 = data.image; // Data URI
        resultImage.src = currentSketchBase64;
        resultImage.classList.remove('hidden');
        actions.classList.remove('hidden');
    } catch (err) {
        alert("Generation failed: " + err.message);
        placeholder.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
    }
});

// Download
downloadBtn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = currentSketchBase64;
    a.download = 'sketch.png';
    a.click();
});

// Match
matchBtn.addEventListener('click', async () => {
    if (!currentSketchBase64) return;

    // Convert base64 to blob
    const res = await fetch(currentSketchBase64);
    const blob = await res.blob();
    const formData = new FormData();
    formData.append('file', blob, 'sketch.png');

    // Show overlay loading state?
    // reuse placeholder or show separate loader

    try {
        const resp = await fetch('/match', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();

        displayMatch(data);
    } catch (err) {
        alert("Matching failed: " + err.message);
    }
});

function displayMatch(data) {
    const rec = data.match;
    // Update Elements
    // Note: photo_path is local server path (e.g. dataset/CUFS...), needs to be served.
    // Ideally we should have a static mount for dataset/ too.
    // For now, let's assume we can add that mount to app.py.

    // Path fix: dataset/CUFS/train/photos/xyz.jpg -> /static/photos/xyz.jpg ? 
    // We need to handle this in app.py.
    // Hack: we'll just show the result details first.

    // Fix path for serving
    // db record has "dataset/CUFS/train/photos/filename.jpg"
    // we mounted "dataset" at "/dataset"
    // so "dataset/..." becomes "/dataset/..." (relative to root, it matches)
    document.getElementById('dbPhoto').src = "/" + rec.photo_path;


    document.getElementById('criminalName').textContent = rec.name;
    document.getElementById('criminalID').textContent = rec.id || 'N/A';
    document.getElementById('criminalAge').textContent = rec.age;
    document.getElementById('criminalCrime').textContent = rec.crime;
    document.getElementById('criminalSentence').textContent = rec.sentence;
    document.getElementById('matchScore').textContent = Math.round(data.score * 100) + "% Match";

    document.getElementById('criminalRisk').textContent = rec.risk_level + " Risk";

    matchOverlay.classList.remove('hidden');
}

function closeMatch() {
    matchOverlay.classList.add('hidden');
}

// Global scope
window.toggleOptions = toggleOptions;
window.closeMatch = closeMatch;
