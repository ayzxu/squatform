import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum

class ViewAngle(Enum):
    """Camera view angle types."""
    SIDE_VIEW = "side_view"  # 90° perpendicular (ideal)
    FRONT_VIEW = "front_view"  # 0° (facing camera)
    BACK_VIEW = "back_view"  # 180° (back to camera)
    ANGLED_VIEW = "angled_view"  # 30-60° or 120-150° (diagonal)
    UNKNOWN = "unknown"

class PersonOrientation(Enum):
    """Person's facing direction in side view."""
    FACING_LEFT = "facing_left"
    FACING_RIGHT = "facing_right"
    UNKNOWN = "unknown"

class AngleNormalizer:
    """Detects video angle and normalizes keypoints for consistent analysis."""
    
    def __init__(self):
        self.detected_angle = None
        self.detected_orientation = None
        self.rotation_angle = 0.0
    
    def detect_and_normalize(self, frames_keypoints: List[Dict]) -> List[Dict]:
        """
        Detect video angle and normalize all keypoints.
        
        Args:
            frames_keypoints: List of keypoint dictionaries for each frame
            
        Returns:
            Normalized list of keypoint dictionaries
        """
        if len(frames_keypoints) == 0:
            return frames_keypoints
        
        # Use first few frames (standing position) for angle detection
        sample_frames = frames_keypoints[:min(10, len(frames_keypoints))]
        
        # Detect view angle
        self.detected_angle = self._detect_view_angle(sample_frames)
        
        # Detect person orientation (for side views)
        if self.detected_angle == ViewAngle.SIDE_VIEW:
            self.detected_orientation = self._detect_orientation(sample_frames)
        
        # Normalize keypoints
        normalized_frames = self._normalize_keypoints(frames_keypoints)
        
        return normalized_frames
    
    def _detect_view_angle(self, sample_frames: List[Dict]) -> ViewAngle:
        """
        Detect the camera view angle based on keypoint relationships.
        
        Uses shoulder width vs depth, and visibility patterns.
        """
        if not sample_frames:
            return ViewAngle.UNKNOWN
        
        # Get average shoulder positions
        shoulder_widths = []
        shoulder_depths = []
        
        for frame in sample_frames:
            left_shoulder = frame.get('left_shoulder')
            right_shoulder = frame.get('right_shoulder')
            
            if left_shoulder and right_shoulder:
                # Horizontal distance (width)
                width = abs(right_shoulder[0] - left_shoulder[0])
                shoulder_widths.append(width)
                
                # Vertical distance (depth indicator in side view)
                depth = abs(right_shoulder[1] - left_shoulder[1])
                shoulder_depths.append(depth)
        
        if not shoulder_widths:
            return ViewAngle.UNKNOWN
        
        avg_width = np.mean(shoulder_widths)
        avg_depth = np.mean(shoulder_depths) if shoulder_depths else 0
        
        # Calculate width-to-depth ratio
        # Side view: width is small, depth is small (shoulders overlap)
        # Front/back view: width is large, depth is small
        # Angled view: width is medium, depth is medium
        
        width_depth_ratio = avg_width / (avg_depth + 0.001)  # Avoid division by zero
        
        # Check ankle visibility pattern
        ankle_visibility = self._check_ankle_visibility(sample_frames)
        
        # Determine view angle
        if avg_width < 0.05:  # Very narrow shoulders = side view
            return ViewAngle.SIDE_VIEW
        elif width_depth_ratio > 10:  # Very wide shoulders = front/back view
            # Distinguish front vs back by checking nose visibility
            if self._check_nose_visibility(sample_frames):
                return ViewAngle.FRONT_VIEW
            else:
                return ViewAngle.BACK_VIEW
        elif 0.05 <= avg_width <= 0.15:  # Medium width = angled view
            return ViewAngle.ANGLED_VIEW
        else:
            return ViewAngle.UNKNOWN
    
    def _check_ankle_visibility(self, sample_frames: List[Dict]) -> Dict[str, float]:
        """Check visibility of ankles to help determine angle."""
        left_visible = sum(1 for f in sample_frames if f.get('left_ankle') is not None)
        right_visible = sum(1 for f in sample_frames if f.get('right_ankle') is not None)
        
        return {
            'left': left_visible / len(sample_frames) if sample_frames else 0,
            'right': right_visible / len(sample_frames) if sample_frames else 0
        }
    
    def _check_nose_visibility(self, sample_frames: List[Dict]) -> bool:
        """Check if nose is visible (indicates front view)."""
        nose_visible = sum(1 for f in sample_frames if f.get('nose') is not None)
        return nose_visible / len(sample_frames) > 0.5 if sample_frames else False
    
    def _detect_orientation(self, sample_frames: List[Dict]) -> PersonOrientation:
        """
        Detect if person is facing left or right in side view.
        
        Uses nose position relative to shoulders.
        """
        if not sample_frames:
            return PersonOrientation.UNKNOWN
        
        nose_positions = []
        shoulder_centers = []
        
        for frame in sample_frames:
            nose = frame.get('nose')
            left_shoulder = frame.get('left_shoulder')
            right_shoulder = frame.get('right_shoulder')
            
            if nose and left_shoulder and right_shoulder:
                nose_positions.append(nose[0])  # x-coordinate
                shoulder_center = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_centers.append(shoulder_center)
        
        if not nose_positions:
            return PersonOrientation.UNKNOWN
        
        # Compare nose x-position to shoulder center
        # If nose is to the left of shoulders, person is facing left
        avg_nose_x = np.mean(nose_positions)
        avg_shoulder_x = np.mean(shoulder_centers)
        
        if avg_nose_x < avg_shoulder_x - 0.02:  # Nose significantly left
            return PersonOrientation.FACING_LEFT
        elif avg_nose_x > avg_shoulder_x + 0.02:  # Nose significantly right
            return PersonOrientation.FACING_RIGHT
        else:
            return PersonOrientation.UNKNOWN
    
    def _normalize_keypoints(self, frames_keypoints: List[Dict]) -> List[Dict]:
        """
        Normalize keypoints based on detected angle and orientation.
        
        For side views: Flip horizontally if facing right to standardize to facing left
        For angled views: Attempt to rotate/transform coordinates
        """
        normalized = []
        
        for frame in frames_keypoints:
            normalized_frame = frame.copy()
            
            if self.detected_angle == ViewAngle.SIDE_VIEW:
                # Normalize side view: flip if facing right
                if self.detected_orientation == PersonOrientation.FACING_RIGHT:
                    normalized_frame = self._flip_horizontal(normalized_frame)
            
            elif self.detected_angle == ViewAngle.ANGLED_VIEW:
                # For angled views, try to estimate rotation and correct
                normalized_frame = self._correct_angled_view(normalized_frame)
            
            elif self.detected_angle in [ViewAngle.FRONT_VIEW, ViewAngle.BACK_VIEW]:
                # Front/back views are not ideal, but we can still try to analyze
                # by using depth estimation or warning the user
                pass  # Keep as-is but will warn in analysis
            
            normalized.append(normalized_frame)
        
        return normalized
    
    def _flip_horizontal(self, keypoints: Dict) -> Dict:
        """Flip keypoints horizontally (mirror image)."""
        flipped = {}
        
        for key, value in keypoints.items():
            if value is not None:
                # Flip x-coordinate: x_new = 1 - x_old
                flipped[key] = (1.0 - value[0], value[1])
            else:
                flipped[key] = None
        
        # Swap left/right keypoints
        swap_pairs = [
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow'),
            ('left_wrist', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle'),
        ]
        
        for left_key, right_key in swap_pairs:
            if left_key in flipped and right_key in flipped:
                flipped[left_key], flipped[right_key] = flipped[right_key], flipped[left_key]
        
        return flipped
    
    def _correct_angled_view(self, keypoints: Dict) -> Dict:
        """
        Attempt to correct angled view by estimating rotation angle.
        
        Uses ankle positions to estimate ground plane and correct perspective.
        """
        # This is a simplified approach - could be enhanced with more sophisticated
        # perspective correction algorithms
        
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')
        
        if left_ankle and right_ankle:
            # Estimate rotation based on ankle alignment
            # In a perfect side view, ankles should have similar y-coordinates
            ankle_y_diff = abs(left_ankle[1] - right_ankle[1])
            
            # If difference is large, there's significant rotation
            # We could apply a correction, but for now, just return as-is
            # A more sophisticated approach would use homography transformation
            pass
        
        return keypoints
    
    def get_angle_info(self) -> Dict:
        """Get information about detected angle and orientation."""
        return {
            'view_angle': self.detected_angle.value if self.detected_angle else None,
            'orientation': self.detected_orientation.value if self.detected_orientation else None,
            'is_ideal': self.detected_angle == ViewAngle.SIDE_VIEW,
            'warning': self._get_angle_warning()
        }
    
    def _get_angle_warning(self) -> Optional[str]:
        """Get warning message if angle is not ideal."""
        if self.detected_angle == ViewAngle.FRONT_VIEW:
            return "Front view detected. Side view recommended for accurate analysis."
        elif self.detected_angle == ViewAngle.BACK_VIEW:
            return "Back view detected. Side view recommended for accurate analysis."
        elif self.detected_angle == ViewAngle.ANGLED_VIEW:
            return "Angled view detected. Side view (90°) recommended for best results."
        elif self.detected_angle == ViewAngle.UNKNOWN:
            return "Could not determine video angle. Side view recommended."
        return None

