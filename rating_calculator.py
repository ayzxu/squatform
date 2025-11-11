from typing import Dict

class RatingCalculator:
    """Calculates overall squat form rating and generates feedback."""
    
    def __init__(self):
        # Weights for each metric (should sum to 1.0)
        self.weights = {
            'knee_tracking': 0.25,
            'back_angle': 0.25,
            'depth': 0.30,
            'alignment': 0.20
        }
    
    def calculate_overall_rating(self, analysis_results: Dict) -> Dict:
        """
        Calculate overall rating from individual metric scores.
        
        Args:
            analysis_results: Dictionary from FormAnalyzer.analyze_squat()
            
        Returns:
            Dictionary with overall_score, rating, and comprehensive feedback
        """
        if 'error' in analysis_results:
            return {
                'overall_score': 0,
                'rating': 'F',
                'feedback': analysis_results['error']
            }
        
        # Extract individual scores
        knee_tracking_score = analysis_results['knee_tracking']['score']
        back_angle_score = analysis_results['back_angle']['score']
        depth_score = analysis_results['depth']['score']
        alignment_score = analysis_results['alignment']['score']
        
        # Calculate weighted average
        overall_score = (
            knee_tracking_score * self.weights['knee_tracking'] +
            back_angle_score * self.weights['back_angle'] +
            depth_score * self.weights['depth'] +
            alignment_score * self.weights['alignment']
        )
        
        # Round to nearest integer and convert to native Python int
        overall_score = int(round(float(overall_score)))
        
        # Determine letter rating
        rating = self._get_letter_rating(overall_score)
        
        # Generate comprehensive feedback
        feedback = self._generate_comprehensive_feedback(
            overall_score, analysis_results
        )
        
        return {
            'overall_score': int(overall_score),
            'rating': rating,
            'feedback': feedback,
            'breakdown': {
                'knee_tracking': {
                    'score': int(round(float(knee_tracking_score))),
                    'weight': float(self.weights['knee_tracking']),
                    'feedback': analysis_results['knee_tracking']['feedback']
                },
                'back_angle': {
                    'score': int(round(float(back_angle_score))),
                    'weight': float(self.weights['back_angle']),
                    'feedback': analysis_results['back_angle']['feedback']
                },
                'depth': {
                    'score': int(round(float(depth_score))),
                    'weight': float(self.weights['depth']),
                    'feedback': analysis_results['depth']['feedback']
                },
                'alignment': {
                    'score': int(round(float(alignment_score))),
                    'weight': float(self.weights['alignment']),
                    'feedback': analysis_results['alignment']['feedback']
                }
            }
        }
    
    def _get_letter_rating(self, score: float) -> str:
        """Convert numeric score to letter rating."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_comprehensive_feedback(self, overall_score: float, 
                                        analysis_results: Dict) -> str:
        """Generate overall feedback message."""
        feedback_parts = []
        
        # Overall assessment
        if overall_score >= 90:
            feedback_parts.append("Excellent squat form! Your technique is very solid.")
        elif overall_score >= 80:
            feedback_parts.append("Good squat form with minor areas for improvement.")
        elif overall_score >= 70:
            feedback_parts.append("Decent squat form, but there are several areas to work on.")
        elif overall_score >= 60:
            feedback_parts.append("Your squat form needs improvement. Focus on the key areas below.")
        else:
            feedback_parts.append("Your squat form requires significant attention. Consider working with a trainer or reviewing proper technique.")
        
        # Identify weakest areas
        scores = {
            'Knee Tracking': analysis_results['knee_tracking']['score'],
            'Back Angle': analysis_results['back_angle']['score'],
            'Depth': analysis_results['depth']['score'],
            'Alignment': analysis_results['alignment']['score']
        }
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        weakest = sorted_scores[0]
        
        if weakest[1] < 70:
            feedback_parts.append(f"\nPriority Focus: {weakest[0]} is your weakest area (score: {weakest[1]:.0f}/100).")
            feedback_parts.append(f"  → {analysis_results[weakest[0].lower().replace(' ', '_')]['feedback']}")
        
        # Highlight strengths if any
        strongest = sorted_scores[-1]
        if strongest[1] >= 85:
            feedback_parts.append(f"\nStrength: {strongest[0]} is performing well (score: {strongest[1]:.0f}/100).")
        
        return "\n".join(feedback_parts)

