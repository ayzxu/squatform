from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from form_analyzer import FormAnalyzer
from rating_calculator import RatingCalculator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video and return results."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the video
        analyzer = FormAnalyzer()
        analysis_results = analyzer.analyze_squat(filepath)
        
        # Calculate overall rating
        rating_calc = RatingCalculator()
        final_results = rating_calc.calculate_overall_rating(analysis_results)
        
        # Add video info and angle information
        final_results['video_filename'] = filename
        if 'video_angle' in analysis_results:
            final_results['video_angle'] = analysis_results['video_angle']
        if 'angle_warning' in analysis_results:
            final_results['angle_warning'] = analysis_results['angle_warning']
        # Add snapshots if available
        if 'snapshots' in analysis_results:
            final_results['snapshots'] = analysis_results['snapshots']
        
        # Clean up uploaded file (optional - you might want to keep it)
        # os.remove(filepath)
        
        return jsonify(final_results)
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

