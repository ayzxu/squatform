import numpy as np
from typing import List, Dict, Optional, Tuple
from pose_detector import PoseDetector
from angle_normalizer import AngleNormalizer

class FormAnalyzer:
    """Analyzes squat form based on pose keypoints."""
    
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.angle_normalizer = AngleNormalizer()
    
    def analyze_squat(self, video_path: str) -> Dict:
        """
        Analyze squat form from video.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary containing analysis results with metrics and scores
        """
        # Extract keypoints from video
        frames_keypoints = self.pose_detector.process_video(video_path)
        
        if len(frames_keypoints) == 0:
            return {
                'error': 'No frames detected in video',
                'score': 0
            }
        
        # Detect angle and normalize keypoints
        normalized_keypoints = self.angle_normalizer.detect_and_normalize(frames_keypoints)
        angle_info = self.angle_normalizer.get_angle_info()
        
        # Find the bottom of the squat (lowest hip position)
        bottom_frame_idx = self._find_bottom_frame(normalized_keypoints)
        
        # Calculate metrics using normalized keypoints
        knee_tracking_score, knee_tracking_feedback = self._analyze_knee_tracking(
            normalized_keypoints, bottom_frame_idx
        )
        
        back_angle_score, back_angle_feedback = self._analyze_back_angle(
            normalized_keypoints, bottom_frame_idx
        )
        
        depth_score, depth_feedback = self._analyze_depth(
            normalized_keypoints, bottom_frame_idx
        )
        
        alignment_score, alignment_feedback = self._analyze_alignment(
            normalized_keypoints, bottom_frame_idx
        )
        
        result = {
            'knee_tracking': {
                'score': knee_tracking_score,
                'feedback': knee_tracking_feedback
            },
            'back_angle': {
                'score': back_angle_score,
                'feedback': back_angle_feedback
            },
            'depth': {
                'score': depth_score,
                'feedback': depth_feedback
            },
            'alignment': {
                'score': alignment_score,
                'feedback': alignment_feedback
            },
            'bottom_frame_idx': bottom_frame_idx,
            'total_frames': len(normalized_keypoints),
            # Add angle information
            'video_angle': angle_info
        }
        
        # Add warning if angle is not ideal
        if angle_info.get('warning'):
            result['angle_warning'] = angle_info['warning']
        
        return result
    
    def _find_bottom_frame(self, frames_keypoints: List[Dict]) -> int:
        """
        Find the frame where the squat is at its lowest point.
        
        Args:
            frames_keypoints: List of keypoint dictionaries for each frame
            
        Returns:
            Index of the frame with lowest hip position
        """
        hip_heights = []
        
        for frame_kp in frames_keypoints:
            left_hip = frame_kp.get('left_hip')
            right_hip = frame_kp.get('right_hip')
            
            if left_hip and right_hip:
                # Use y-coordinate (higher y = lower on screen)
                avg_y = (left_hip[1] + right_hip[1]) / 2
                hip_heights.append(avg_y)
            elif left_hip:
                hip_heights.append(left_hip[1])
            elif right_hip:
                hip_heights.append(right_hip[1])
            else:
                hip_heights.append(float('inf'))
        
        if not hip_heights or all(h == float('inf') for h in hip_heights):
            return len(frames_keypoints) // 2  # Default to middle frame
        
        return np.argmax(hip_heights)  # Highest y = lowest position
    
    def _analyze_knee_tracking(self, frames_keypoints: List[Dict], bottom_idx: int) -> Tuple[float, str]:
        """
        Analyze if knees track over toes (lateral deviation).
        
        Args:
            frames_keypoints: List of keypoint dictionaries
            bottom_idx: Index of bottom frame
            
        Returns:
            Tuple of (score 0-100, feedback message)
        """
        if bottom_idx >= len(frames_keypoints):
            return 0.0, "Could not detect squat bottom position"
        
        bottom_frame = frames_keypoints[bottom_idx]
        
        # Get keypoints at bottom of squat
        left_knee = bottom_frame.get('left_knee')
        right_knee = bottom_frame.get('right_knee')
        left_ankle = bottom_frame.get('left_ankle')
        right_ankle = bottom_frame.get('right_ankle')
        
        if not (left_knee and left_ankle) and not (right_knee and right_ankle):
            return 0.0, "Could not detect knee/ankle positions"
        
        deviations = []
        
        # Analyze left side
        if left_knee and left_ankle:
            # Calculate horizontal deviation (knee x - ankle x)
            knee_ankle_deviation = abs(left_knee[0] - left_ankle[0])
            deviations.append(knee_ankle_deviation)
        
        # Analyze right side
        if right_knee and right_ankle:
            knee_ankle_deviation = abs(right_knee[0] - right_ankle[0])
            deviations.append(knee_ankle_deviation)
        
        if not deviations:
            return 0.0, "Could not calculate knee tracking"
        
        avg_deviation = np.mean(deviations)
        
        # Score: 0-0.05 deviation = 100, 0.05-0.1 = 80-100, 0.1-0.15 = 60-80, >0.15 = 0-60
        if avg_deviation < 0.05:
            score = 100.0
            feedback = "Excellent knee tracking - knees stay aligned over toes"
        elif avg_deviation < 0.1:
            score = 100 - (avg_deviation - 0.05) * 400  # Linear from 100 to 80
            feedback = f"Good knee tracking with minor deviation ({avg_deviation*100:.1f}%)"
        elif avg_deviation < 0.15:
            score = 80 - (avg_deviation - 0.1) * 400  # Linear from 80 to 60
            feedback = f"Moderate knee tracking issues - knees deviate {avg_deviation*100:.1f}% from toes"
        else:
            score = max(0, 60 - (avg_deviation - 0.15) * 400)  # Linear from 60 to 0
            feedback = f"Poor knee tracking - significant deviation ({avg_deviation*100:.1f}%) detected. Focus on keeping knees over toes."
        
        return min(100, max(0, score)), feedback
    
    def _analyze_back_angle(self, frames_keypoints: List[Dict], bottom_idx: int) -> Tuple[float, str]:
        """
        Analyze back angle (torso angle relative to vertical).
        
        Args:
            frames_keypoints: List of keypoint dictionaries
            bottom_idx: Index of bottom frame
            
        Returns:
            Tuple of (score 0-100, feedback message)
        """
        if bottom_idx >= len(frames_keypoints):
            return 0.0, "Could not detect squat bottom position"
        
        bottom_frame = frames_keypoints[bottom_idx]
        
        # Get shoulder and hip positions
        left_shoulder = bottom_frame.get('left_shoulder')
        right_shoulder = bottom_frame.get('right_shoulder')
        left_hip = bottom_frame.get('left_hip')
        right_hip = bottom_frame.get('right_hip')
        
        if not (left_shoulder or right_shoulder) or not (left_hip or right_hip):
            return 0.0, "Could not detect shoulder/hip positions"
        
        # Calculate average positions
        if left_shoulder and right_shoulder:
            shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder:
            shoulder_x, shoulder_y = left_shoulder
        else:
            shoulder_x, shoulder_y = right_shoulder
        
        if left_hip and right_hip:
            hip_x = (left_hip[0] + right_hip[0]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
        elif left_hip:
            hip_x, hip_y = left_hip
        else:
            hip_x, hip_y = right_hip
        
        # Calculate angle from vertical
        # Vector from hip to shoulder
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        
        if abs(dy) < 0.01:  # Avoid division by zero
            return 50.0, "Could not calculate back angle accurately"
        
        # Angle from vertical (in radians, then convert to degrees)
        angle_rad = np.arctan2(abs(dx), abs(dy))
        angle_deg = np.degrees(angle_rad)
        
        # Ideal back angle is around 15-30 degrees from vertical
        # Score based on how close to ideal range
        if 15 <= angle_deg <= 30:
            score = 100.0
            feedback = f"Excellent back angle ({angle_deg:.1f}°) - maintains good posture"
        elif 10 <= angle_deg < 15 or 30 < angle_deg <= 35:
            score = 85.0
            feedback = f"Good back angle ({angle_deg:.1f}°) - slightly outside ideal range"
        elif 5 <= angle_deg < 10 or 35 < angle_deg <= 40:
            score = 70.0
            feedback = f"Moderate back angle issue ({angle_deg:.1f}°) - consider adjusting torso position"
        elif angle_deg < 5:
            score = 50.0
            feedback = f"Too upright ({angle_deg:.1f}°) - lean forward slightly to maintain balance"
        else:  # angle_deg > 40
            score = max(0, 50 - (angle_deg - 40) * 2.5)
            feedback = f"Excessive forward lean ({angle_deg:.1f}°) - focus on keeping chest up and back straight"
        
        return min(100, max(0, score)), feedback
    
    def _analyze_depth(self, frames_keypoints: List[Dict], bottom_idx: int) -> Tuple[float, str]:
        """
        Analyze if hips go below knees at bottom of squat.
        
        Args:
            frames_keypoints: List of keypoint dictionaries
            bottom_idx: Index of bottom frame
            
        Returns:
            Tuple of (score 0-100, feedback message)
        """
        if bottom_idx >= len(frames_keypoints):
            return 0.0, "Could not detect squat bottom position"
        
        bottom_frame = frames_keypoints[bottom_idx]
        
        # Get hip and knee positions at bottom
        left_hip = bottom_frame.get('left_hip')
        right_hip = bottom_frame.get('right_hip')
        left_knee = bottom_frame.get('left_knee')
        right_knee = bottom_frame.get('right_knee')
        
        if not (left_hip or right_hip) or not (left_knee or right_knee):
            return 0.0, "Could not detect hip/knee positions"
        
        # Calculate average positions
        if left_hip and right_hip:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        elif left_hip:
            hip_y = left_hip[1]
        else:
            hip_y = right_hip[1]
        
        if left_knee and right_knee:
            knee_y = (left_knee[1] + right_knee[1]) / 2
        elif left_knee:
            knee_y = left_knee[1]
        else:
            knee_y = right_knee[1]
        
        # Check if hip is below knee (higher y value = lower on screen)
        depth_achieved = hip_y > knee_y
        
        if depth_achieved:
            # Calculate how much below (as percentage)
            depth_percentage = ((hip_y - knee_y) / knee_y) * 100
            
            if depth_percentage > 5:
                score = 100.0
                feedback = f"Excellent depth - hips well below knees ({depth_percentage:.1f}% below)"
            elif depth_percentage > 2:
                score = 90.0
                feedback = f"Good depth - hips below knees ({depth_percentage:.1f}% below)"
            else:
                score = 80.0
                feedback = f"Adequate depth - hips just below knees ({depth_percentage:.1f}% below)"
        else:
            # Calculate how far above
            depth_shortage = ((knee_y - hip_y) / knee_y) * 100
            
            if depth_shortage < 2:
                score = 60.0
                feedback = f"Shallow squat - hips at knee level. Go deeper for full range of motion."
            elif depth_shortage < 5:
                score = 40.0
                feedback = f"Shallow squat - hips {depth_shortage:.1f}% above knees. Focus on achieving parallel or below."
            else:
                score = 20.0
                feedback = f"Very shallow squat - hips {depth_shortage:.1f}% above knees. Need significant improvement in depth."
        
        return min(100, max(0, score)), feedback
    
    def _analyze_alignment(self, frames_keypoints: List[Dict], bottom_idx: int) -> Tuple[float, str]:
        """
        Analyze hip-knee-ankle alignment.
        
        Args:
            frames_keypoints: List of keypoint dictionaries
            bottom_idx: Index of bottom frame
            
        Returns:
            Tuple of (score 0-100, feedback message)
        """
        if bottom_idx >= len(frames_keypoints):
            return 0.0, "Could not detect squat bottom position"
        
        bottom_frame = frames_keypoints[bottom_idx]
        
        # Get keypoints
        left_hip = bottom_frame.get('left_hip')
        right_hip = bottom_frame.get('right_hip')
        left_knee = bottom_frame.get('left_knee')
        right_knee = bottom_frame.get('right_knee')
        left_ankle = bottom_frame.get('left_ankle')
        right_ankle = bottom_frame.get('right_ankle')
        
        alignment_scores = []
        feedbacks = []
        
        # Analyze left side alignment
        if left_hip and left_knee and left_ankle:
            # Calculate angle at knee (hip-knee-ankle)
            angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            if angle is not None:
                # Ideal angle is around 90-100 degrees at bottom of squat
                if 85 <= angle <= 105:
                    alignment_scores.append(100.0)
                    feedbacks.append("Left side: Excellent alignment")
                elif 75 <= angle < 85 or 105 < angle <= 115:
                    alignment_scores.append(85.0)
                    feedbacks.append("Left side: Good alignment")
                elif 65 <= angle < 75 or 115 < angle <= 125:
                    alignment_scores.append(70.0)
                    feedbacks.append("Left side: Moderate alignment issues")
                else:
                    alignment_scores.append(50.0)
                    feedbacks.append(f"Left side: Poor alignment (angle: {angle:.1f}°)")
        
        # Analyze right side alignment
        if right_hip and right_knee and right_ankle:
            angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            if angle is not None:
                if 85 <= angle <= 105:
                    alignment_scores.append(100.0)
                    feedbacks.append("Right side: Excellent alignment")
                elif 75 <= angle < 85 or 105 < angle <= 115:
                    alignment_scores.append(85.0)
                    feedbacks.append("Right side: Good alignment")
                elif 65 <= angle < 75 or 115 < angle <= 125:
                    alignment_scores.append(70.0)
                    feedbacks.append("Right side: Moderate alignment issues")
                else:
                    alignment_scores.append(50.0)
                    feedbacks.append(f"Right side: Poor alignment (angle: {angle:.1f}°)")
        
        if not alignment_scores:
            return 0.0, "Could not calculate alignment"
        
        avg_score = np.mean(alignment_scores)
        combined_feedback = " | ".join(feedbacks)
        
        return avg_score, combined_feedback
    
    def _calculate_angle(self, point1: Tuple[float, float], 
                        vertex: Tuple[float, float], 
                        point2: Tuple[float, float]) -> Optional[float]:
        """
        Calculate angle at vertex formed by three points.
        
        Args:
            point1: First point
            vertex: Vertex point (where angle is measured)
            point2: Second point
            
        Returns:
            Angle in degrees or None if calculation fails
        """
        try:
            # Vectors from vertex to points
            vec1 = np.array([point1[0] - vertex[0], point1[1] - vertex[1]])
            vec2 = np.array([point2[0] - vertex[0], point2[1] - vertex[1]])
            
            # Calculate angle using dot product
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return None
            
            cos_angle = dot_product / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
        except:
            return None

