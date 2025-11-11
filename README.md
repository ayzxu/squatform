# Squat Form Analyzer

A web application that analyzes squat form from uploaded videos using pose estimation and provides a detailed rating (0-100) with feedback.

## Features

- **Pose Detection**: Uses MediaPipe to detect key body points (hips, knees, ankles, shoulders)
- **Form Analysis**: Evaluates four key metrics:
  - **Knee Tracking**: Checks if knees stay aligned over toes
  - **Back Angle**: Measures torso angle relative to vertical
  - **Depth**: Verifies if hips go below knees at bottom of squat
  - **Alignment**: Analyzes hip-knee-ankle alignment
- **Rating System**: Provides overall score (0-100) with letter grade (A-F)
- **Detailed Feedback**: Offers specific, actionable feedback for each metric

## Installation

1. Install Python 3.8-3.12 (since MediaPipe does not support Python 3.13+)

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload a video file (MP4, AVI, MOV, MKV, or WEBM, max 100MB)

4. Click "Analyze Squat Form" to get your results

## How It Works

1. **Video Upload**: The video is uploaded to the server
2. **Pose Detection**: MediaPipe processes each frame to extract body keypoints
3. **Form Analysis**: The system analyzes key metrics at the bottom of the squat
4. **Rating Calculation**: Individual metric scores are weighted and combined into an overall score
5. **Results Display**: Score, feedback, and detailed metrics are shown in the web interface

## Project Structure

```
CV/
├── app.py                 # Flask web application
├── pose_detector.py       # Pose estimation using MediaPipe
├── form_analyzer.py       # Squat form analysis logic
├── rating_calculator.py   # Score calculation and feedback
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
└── uploads/              # Uploaded videos (created automatically)
```

## Technical Details

- **Backend**: Flask web framework
- **Pose Estimation**: MediaPipe Pose (lightweight, fast, accurate)
- **Video Processing**: OpenCV for frame extraction
- **Analysis**: Custom algorithms based on biomechanical principles

## Notes

- The analysis works best with side-view videos of squats
- Ensure good lighting and clear visibility of the person
- The person should be fully visible in the frame
- Multiple squats in one video will analyze the first complete squat detected

