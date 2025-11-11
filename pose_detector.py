import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional

class PoseDetector:
    """Detects human pose keypoints from video frames using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process_video(self, video_path: str, return_frames: bool = False) -> List[Dict[str, Optional[Tuple[float, float]]]]:
        """
        Process video and extract pose keypoints for each frame.
        
        Args:
            video_path: Path to the input video file
            return_frames: If True, also return frames with pose overlays
            
        Returns:
            List of dictionaries containing keypoints for each frame.
            Each dict has keys like 'left_hip', 'right_knee', etc.
            Values are (x, y) tuples normalized to [0, 1] or None if not detected.
            If return_frames is True, also returns a list of annotated frames.
        """
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        annotated_frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            # Extract keypoints
            frame_keypoints = self._extract_keypoints(results.pose_landmarks)
            frames_data.append(frame_keypoints)
            
            # If requested, draw pose on frame
            if return_frames:
                annotated_frame = frame.copy()
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                annotated_frames.append(annotated_frame)
        
        cap.release()
        
        if return_frames:
            return frames_data, annotated_frames
        return frames_data
    
    def get_annotated_frame(self, frame, landmarks):
        """Get a single frame with pose landmarks drawn on it."""
        annotated = frame.copy()
        if landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        return annotated
    
    def _extract_keypoints(self, landmarks) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Extract keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary with keypoint names and (x, y) coordinates
        """
        if landmarks is None:
            return self._empty_keypoints()
        
        # MediaPipe pose landmark indices
        keypoint_map = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
        keypoints = {}
        for name, idx in keypoint_map.items():
            landmark = landmarks.landmark[idx]
            if landmark.visibility > 0.5:  # Only include visible keypoints
                keypoints[name] = (landmark.x, landmark.y)
            else:
                keypoints[name] = None
        
        return keypoints
    
    def _empty_keypoints(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """Return empty keypoints dictionary."""
        return {
            'nose': None,
            'left_shoulder': None,
            'right_shoulder': None,
            'left_elbow': None,
            'right_elbow': None,
            'left_wrist': None,
            'right_wrist': None,
            'left_hip': None,
            'right_hip': None,
            'left_knee': None,
            'right_knee': None,
            'left_ankle': None,
            'right_ankle': None,
        }
    
    def get_average_keypoint(self, keypoints_list: List[Dict], keypoint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get average position of a keypoint across multiple frames.
        
        Args:
            keypoints_list: List of keypoint dictionaries
            keypoint_name: Name of the keypoint to average
            
        Returns:
            Average (x, y) position or None if not detected in any frame
        """
        valid_points = [
            kp[keypoint_name] for kp in keypoints_list
            if kp[keypoint_name] is not None
        ]
        
        if not valid_points:
            return None
        
        avg_x = np.mean([p[0] for p in valid_points])
        avg_y = np.mean([p[1] for p in valid_points])
        return (avg_x, avg_y)

