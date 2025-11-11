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
    
    // Display snapshots if available
    console.log('Full data received:', data);
    if (data.snapshots) {
        console.log('Snapshots found in data:', data.snapshots);
        displaySnapshots(data.snapshots);
    } else {
        console.log('No snapshots in data');
    }
    
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

function displaySnapshots(snapshots) {
    const snapshotsSection = document.getElementById('snapshotsSection');
    const snapshotsGrid = document.getElementById('snapshotsGrid');
    
    console.log('Snapshots received:', snapshots);
    
    if (!snapshots || Object.keys(snapshots).length === 0) {
        console.log('No snapshots to display');
        snapshotsSection.style.display = 'none';
        return;
    }
    
    snapshotsGrid.innerHTML = '';
    
    // Define order and icons for snapshots
    const snapshotOrder = ['start', 'mid_descent', 'bottom', 'mid_ascent', 'end'];
    const snapshotIcons = {
        'start': '🏁',
        'mid_descent': '⬇️',
        'bottom': '⬇️',
        'mid_ascent': '⬆️',
        'end': '🏁'
    };
    
    snapshotOrder.forEach(key => {
        if (snapshots[key]) {
            const snapshot = snapshots[key];
            const card = document.createElement('div');
            card.className = 'snapshot-card';
            card.style.cursor = 'pointer';
            
            // Build angle info text
            let angleInfo = '';
            if (snapshot.angles) {
                const angles = snapshot.angles;
                if (angles.back_angle !== null) {
                    angleInfo += `Back: ${angles.back_angle.toFixed(1)}°`;
                }
                if (angles.left_knee_angle !== null || angles.right_knee_angle !== null) {
                    if (angleInfo) angleInfo += ' | ';
                    const kneeAngles = [];
                    if (angles.left_knee_angle !== null) kneeAngles.push(`L: ${angles.left_knee_angle.toFixed(1)}°`);
                    if (angles.right_knee_angle !== null) kneeAngles.push(`R: ${angles.right_knee_angle.toFixed(1)}°`);
                    angleInfo += `Knees: ${kneeAngles.join(', ')}`;
                }
            }
            
            card.innerHTML = `
                <div class="snapshot-image-container">
                    <img src="${snapshot.image}" alt="${snapshot.label}" class="snapshot-image">
                    <div class="snapshot-badge">${snapshotIcons[key] || '📸'}</div>
                </div>
                <div class="snapshot-label">${snapshot.label}</div>
                <div class="snapshot-frame-info">Frame ${snapshot.frame_idx + 1}</div>
                ${angleInfo ? `<div class="snapshot-angles">${angleInfo}</div>` : ''}
            `;
            
            // Add click handler for lightbox
            card.addEventListener('click', () => {
                openLightbox(snapshot);
            });
            
            snapshotsGrid.appendChild(card);
        }
    });
    
    snapshotsSection.style.display = 'block';
}

let currentRotation = 0; // Track current rotation angle
let originalImageWidth = 0;
let originalImageHeight = 0;

function openLightbox(snapshot) {
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxInfo = document.getElementById('lightboxInfo');
    const lightboxImageContainer = document.querySelector('.lightbox-image-container');
    
    lightboxImage.src = snapshot.image;
    lightboxImage.alt = snapshot.label;
    
    // Reset rotation when opening new image
    currentRotation = 0;
    lightboxImage.style.transform = 'rotate(0deg)';
    
    // Wait for image to load to get dimensions
    lightboxImage.onload = function() {
        originalImageWidth = lightboxImage.naturalWidth;
        originalImageHeight = lightboxImage.naturalHeight;
        updateContainerSize();
    };
    
    // Build info HTML
    let infoHTML = `<h3>${snapshot.label}</h3>`;
    infoHTML += `<p><strong>Frame:</strong> ${snapshot.frame_idx + 1}</p>`;
    
    if (snapshot.angles) {
        infoHTML += '<div class="lightbox-angles">';
        if (snapshot.angles.back_angle !== null) {
            infoHTML += `<p><strong>Back Angle:</strong> ${snapshot.angles.back_angle.toFixed(1)}°</p>`;
        }
        if (snapshot.angles.left_knee_angle !== null) {
            infoHTML += `<p><strong>Left Knee Angle:</strong> ${snapshot.angles.left_knee_angle.toFixed(1)}°</p>`;
        }
        if (snapshot.angles.right_knee_angle !== null) {
            infoHTML += `<p><strong>Right Knee Angle:</strong> ${snapshot.angles.right_knee_angle.toFixed(1)}°</p>`;
        }
        infoHTML += '</div>';
    }
    
    lightboxInfo.innerHTML = infoHTML;
    lightbox.style.display = 'flex';
    
    // Prevent body scroll when lightbox is open
    document.body.style.overflow = 'hidden';
    
    // Update container size after info is rendered (small delay to ensure DOM is updated)
    setTimeout(() => {
        updateContainerSize();
    }, 100);
}

function updateContainerSize() {
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxImageContainer = document.querySelector('.lightbox-image-container');
    const lightboxContent = document.querySelector('.lightbox-content');
    const lightboxInfo = document.getElementById('lightboxInfo');
    
    if (!lightboxImageContainer || !lightboxContent || originalImageWidth === 0 || originalImageHeight === 0) {
        return;
    }
    
    // Calculate dimensions based on rotation
    let displayWidth, displayHeight;
    
    if (currentRotation === 90 || currentRotation === 270) {
        // Swapped dimensions
        displayWidth = originalImageHeight;
        displayHeight = originalImageWidth;
    } else {
        // Normal dimensions
        displayWidth = originalImageWidth;
        displayHeight = originalImageHeight;
    }
    
    // Calculate aspect ratio
    const aspectRatio = displayWidth / displayHeight;
    
    // Estimate info section height (roughly 150-200px depending on content)
    const infoHeight = lightboxInfo ? lightboxInfo.offsetHeight : 150;
    const padding = 40; // Top and bottom padding of lightbox-content
    const margin = 15; // Margin between image and info
    
    // Set max dimensions - account for info section and padding
    const maxTotalHeight = window.innerHeight * 0.9;
    const maxImageHeight = maxTotalHeight - infoHeight - padding - margin;
    const maxWidth = window.innerWidth * 0.85;
    
    let finalWidth, finalHeight;
    
    // Calculate based on height constraint first
    if (displayHeight > maxImageHeight) {
        finalHeight = maxImageHeight;
        finalWidth = finalHeight * aspectRatio;
    } else {
        finalHeight = displayHeight;
        finalWidth = displayWidth;
    }
    
    // Then check width constraint
    if (finalWidth > maxWidth) {
        finalWidth = maxWidth;
        finalHeight = finalWidth / aspectRatio;
        
        // Re-check height after width adjustment
        if (finalHeight > maxImageHeight) {
            finalHeight = maxImageHeight;
            finalWidth = finalHeight * aspectRatio;
        }
    }
    
    // Update container size
    lightboxImageContainer.style.width = `${finalWidth}px`;
    lightboxImageContainer.style.height = `${finalHeight}px`;
    lightboxImageContainer.style.minWidth = `${finalWidth}px`;
    lightboxImageContainer.style.minHeight = `${finalHeight}px`;
    
    // Ensure lightbox content fits within viewport
    const totalContentHeight = finalHeight + (lightboxInfo ? lightboxInfo.offsetHeight : 150) + padding + margin;
    if (totalContentHeight > maxTotalHeight) {
        // Adjust image size to fit
        const availableImageHeight = maxTotalHeight - (lightboxInfo ? lightboxInfo.offsetHeight : 150) - padding - margin;
        if (availableImageHeight < finalHeight) {
            finalHeight = availableImageHeight;
            finalWidth = finalHeight * aspectRatio;
            lightboxImageContainer.style.width = `${finalWidth}px`;
            lightboxImageContainer.style.height = `${finalHeight}px`;
        }
    }
}

function rotateLightboxImage() {
    currentRotation = (currentRotation + 90) % 360;
    const lightboxImage = document.getElementById('lightboxImage');
    lightboxImage.style.transform = `rotate(${currentRotation}deg)`;
    lightboxImage.style.transition = 'transform 0.3s ease';
    
    // Update container size after rotation (wait for transform to apply and info to be measured)
    setTimeout(() => {
        updateContainerSize();
    }, 100); // Delay to ensure transform is applied and DOM is updated
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Lightbox event listeners (set up immediately since script is at bottom of HTML)
const lightbox = document.getElementById('lightbox');
const lightboxClose = document.getElementById('lightboxClose');
const lightboxRotateBtn = document.getElementById('lightboxRotateBtn');

if (lightboxClose) {
    lightboxClose.addEventListener('click', closeLightbox);
}

if (lightboxRotateBtn) {
    lightboxRotateBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent closing lightbox when clicking rotate button
        rotateLightboxImage();
    });
}

// Close on background click
if (lightbox) {
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });
}

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox && lightbox.style.display === 'flex') {
        closeLightbox();
    }
});

function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

function hideResults() {
    results.style.display = 'none';
    // Also hide angle info and snapshots when hiding results
    document.getElementById('angleInfo').style.display = 'none';
    document.getElementById('snapshotsSection').style.display = 'none';
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

