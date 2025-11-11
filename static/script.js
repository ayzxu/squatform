const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const videoPreview = document.getElementById('videoPreview');
const previewVideo = document.getElementById('previewVideo');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const resetBtn = document.getElementById('resetBtn');

// Upload area click handler
uploadArea.addEventListener('click', () => {
    videoInput.click();
});

// File input change handler
videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        handleFile(file);
    } else {
        showError('Please drop a video file');
    }
});

function handleFile(file) {
    // Check file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
        showError('File size exceeds 100MB limit');
        return;
    }
    
    // Create preview
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    
    uploadArea.style.display = 'none';
    videoPreview.style.display = 'block';
    hideError();
    hideResults();
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    const file = videoInput.files[0];
    if (!file) {
        showError('Please select a video file first');
        return;
    }
    
    // Show loading, hide other sections
    loading.style.display = 'block';
    videoPreview.style.display = 'none';
    hideError();
    hideResults();
    
    // Create form data
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        displayResults(data);
        
    } catch (err) {
        showError(err.message || 'Failed to analyze video. Please try again.');
    } finally {
        loading.style.display = 'none';
    }
});

function displayResults(data) {
    // Set overall score
    document.getElementById('overallScore').textContent = data.overall_score;
    document.getElementById('ratingBadge').textContent = data.rating;
    
    // Display angle information
    const angleInfo = document.getElementById('angleInfo');
    const angleText = document.getElementById('angleText');
    
    if (data.video_angle) {
        if (data.video_angle.warning) {
            angleInfo.style.display = 'block';
            angleText.textContent = data.video_angle.warning;
            angleText.style.color = '#ff6b6b';
        } else if (data.video_angle.is_ideal) {
            angleInfo.style.display = 'block';
            angleText.textContent = '✓ Side view detected - optimal angle for analysis';
            angleText.style.color = '#51cf66';
        } else {
            angleInfo.style.display = 'none';
        }
    } else {
        angleInfo.style.display = 'none';
    }
    
    // Set feedback
    document.getElementById('feedbackText').textContent = data.feedback;
    
    // Display metrics
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = '';
    
    const metrics = [
        { key: 'knee_tracking', name: 'Knee Tracking' },
        { key: 'back_angle', name: 'Back Angle' },
        { key: 'depth', name: 'Depth' },
        { key: 'alignment', name: 'Alignment' }
    ];
    
    metrics.forEach(metric => {
        const metricData = data.breakdown[metric.key];
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-header">
                <span class="metric-name">${metric.name}</span>
                <span class="metric-score">${metricData.score}/100</span>
            </div>
            <div class="metric-feedback">${metricData.feedback}</div>
        `;
        metricsGrid.appendChild(card);
    });
    
    // Show results
    results.style.display = 'block';
}

function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

function hideResults() {
    results.style.display = 'none';
    // Also hide angle info when hiding results
    document.getElementById('angleInfo').style.display = 'none';
}

// Reset button handler
resetBtn.addEventListener('click', () => {
    videoInput.value = '';
    previewVideo.src = '';
    uploadArea.style.display = 'block';
    videoPreview.style.display = 'none';
    hideResults();
    hideError();
});

