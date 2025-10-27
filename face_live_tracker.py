import cv2
import numpy as np
import mediapipe as mp
import logging
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time

logger = logging.getLogger(__name__)

class LiveFaceTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh untuk deteksi wajah yang presisi
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Define regions untuk pipi dan bibir
        self.cheek_regions = self._define_cheek_regions()
        self.lip_regions = self._define_lip_regions()
        
        # State untuk efek aktif
        self.active_effects = {
            "cheek_color": None,
            "lip_color": None
        }
        
        # Original image storage
        self.original_images = {}

    def _define_cheek_regions(self):
        """Define regions untuk kulit pipi"""
        return {
            "left_cheek": [117, 118, 119, 100, 47, 126, 209, 49, 131, 134, 51, 4, 5, 50, 101, 36, 137, 177, 123, 116],
            "right_cheek": [346, 347, 348, 329, 277, 355, 429, 279, 360, 363, 281, 4, 5, 280, 330, 266, 366, 397, 352, 345],
            "cheek_extension_left": [143, 111, 117, 118, 119, 100, 47, 126, 209],
            "cheek_extension_right": [372, 340, 346, 347, 348, 329, 277, 355, 429]
        }

    def _define_lip_regions(self):
        """Define regions untuk bibir (lipstick)"""
        return {
            "upper_lip": [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            "lower_lip": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
            "lip_corners": [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324],
            "lip_outer": [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
        }

    def get_face_landmarks(self, image):
        """Get precise facial landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            return results.multi_face_landmarks[0]
        except Exception as e:
            logger.error(f"Error in face landmarks: {str(e)}")
            return None

    def create_cheek_mask(self, image, landmarks):
        """Create mask khusus untuk area pipi"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            cheek_points = []
            
            # Collect points untuk kedua pipi
            for region_name, indices in self.cheek_regions.items():
                for idx in indices:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cheek_points.append([x, y])
            
            if len(cheek_points) > 2:
                # Create convex hull untuk area pipi
                hull = cv2.convexHull(np.array(cheek_points))
                
                # Fill the convex hull
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Refine mask untuk hasil yang lebih natural
                mask = self._refine_cheek_mask(mask, landmarks, h, w)
                
                # Smooth the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (21, 21), 5)
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating cheek mask: {str(e)}")
            return mask

    def create_lip_mask(self, image, landmarks):
        """Create mask khusus untuk area bibir"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            lip_points = []
            
            # Collect points untuk bibir
            for region_name, indices in self.lip_regions.items():
                for idx in indices:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        lip_points.append([x, y])
            
            if len(lip_points) > 2:
                # Create convex hull untuk bibir
                hull = cv2.convexHull(np.array(lip_points))
                
                # Fill the convex hull
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Smooth the mask untuk hasil lipstick yang natural
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (11, 11), 3)
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating lip mask: {str(e)}")
            return mask

    def _refine_cheek_mask(self, mask, landmarks, h, w):
        """Refine cheek mask untuk menghindari area mata dan hidung"""
        try:
            # Exclude area mata
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            exclusion_points = []
            for idx in left_eye_indices + right_eye_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    exclusion_points.append([x, y])
            
            if len(exclusion_points) > 2:
                exclusion_hull = cv2.convexHull(np.array(exclusion_points))
                cv2.fillConvexPoly(mask, exclusion_hull, 0)
                
        except Exception as e:
            logger.error(f"Error refining cheek mask: {str(e)}")
        
        return mask

    def apply_cheek_color(self, image, cheek_hex):
        """Apply blush/cheek color dengan blending natural"""
        try:
            # Convert hex to RGB
            cheek_hex = cheek_hex.lstrip('#')
            if len(cheek_hex) != 6:
                return image, "Invalid cheek color"
                
            target_rgb = tuple(int(cheek_hex[i:i+2], 16) for i in (0, 2, 4))
            
            # Convert image to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)
            
            # Get landmarks dan create cheek mask
            landmarks = self.get_face_landmarks(image)
            if not landmarks:
                return image, "No face detected for cheek color application"
            
            cheek_mask = self.create_cheek_mask(image, landmarks)
            
            if np.sum(cheek_mask) == 0:
                return image, "No cheek area detected"
            
            # Create high-quality mask for blending
            mask_float = cheek_mask.astype(float) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (25, 25), 7)
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Apply cheek color dengan blending yang natural
            result = self._blend_cheek_color(img_rgb, target_rgb, mask_float)
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            return result_bgr, "Cheek color applied naturally"
            
        except Exception as e:
            logger.error(f"Error applying cheek color: {str(e)}")
            return image, f"Application error: {str(e)}"

    def apply_lipstick(self, image, lipstick_hex):
        """Apply lipstick color dengan blending natural"""
        try:
            # Convert hex to RGB
            lipstick_hex = lipstick_hex.lstrip('#')
            if len(lipstick_hex) != 6:
                return image, "Invalid lipstick color"
                
            target_rgb = tuple(int(lipstick_hex[i:i+2], 16) for i in (0, 2, 4))
            
            # Convert image to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)
            
            # Get landmarks dan create lip mask
            landmarks = self.get_face_landmarks(image)
            if not landmarks:
                return image, "No face detected for lipstick application"
            
            lip_mask = self.create_lip_mask(image, landmarks)
            
            if np.sum(lip_mask) == 0:
                return image, "No lip area detected"
            
            # Create high-quality mask for blending
            mask_float = lip_mask.astype(float) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (9, 9), 2)
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Apply lipstick dengan blending yang natural
            result = self._blend_lipstick(img_rgb, target_rgb, mask_float)
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            return result_bgr, "Lipstick applied naturally"
            
        except Exception as e:
            logger.error(f"Error applying lipstick: {str(e)}")
            return image, f"Application error: {str(e)}"

    def _blend_cheek_color(self, original, target_rgb, mask):
        """Advanced blending untuk cheek color yang natural seperti blush"""
        try:
            result = original.copy()
            
            # Buat cheek color layer dengan opacity yang natural
            cheek_layer = np.ones_like(original)
            cheek_layer[:, :, 0] = target_rgb[0]  # R
            cheek_layer[:, :, 1] = target_rgb[1]  # G  
            cheek_layer[:, :, 2] = target_rgb[2]  # B
            
            # Soft light blending untuk efek natural
            for c in range(3):
                original_channel = original[:, :, c]
                cheek_channel = cheek_layer[:, :, c]
                
                # Soft light blending formula
                blended = np.where(original_channel < 128, 
                                 (2 * original_channel * cheek_channel) / 255,
                                 255 - 2 * (255 - original_channel) * (255 - cheek_channel) / 255)
                
                # Apply dengan mask
                result[:, :, c] = original_channel * (1 - mask[:, :, c]) + blended * mask[:, :, c]
            
            return np.clip(result, 0, 255)
            
        except Exception as e:
            logger.error(f"Error in cheek color blending: {str(e)}")
            return original

    def _blend_lipstick(self, original, target_rgb, mask):
        """Advanced blending untuk lipstick yang vibrant tapi natural"""
        try:
            result = original.copy()
            
            # Buat lipstick layer
            lipstick_layer = np.ones_like(original)
            lipstick_layer[:, :, 0] = target_rgb[0]  # R
            lipstick_layer[:, :, 1] = target_rgb[1]  # G  
            lipstick_layer[:, :, 2] = target_rgb[2]  # B
            
            # Color burn blending untuk lipstick yang vibrant
            for c in range(3):
                original_channel = original[:, :, c]
                lipstick_channel = lipstick_layer[:, :, c]
                
                # Color burn blending
                blended = 255 - (255 - original_channel) / (lipstick_channel + 1e-6) * 255
                blended = np.clip(blended, 0, 255)
                
                # Apply dengan mask
                result[:, :, c] = original_channel * (1 - mask[:, :, c]) + blended * mask[:, :, c]
            
            return np.clip(result, 0, 255)
            
        except Exception as e:
            logger.error(f"Error in lipstick blending: {str(e)}")
            return original

    def apply_combined_effects(self, image, cheek_hex=None, lipstick_hex=None):
        """Apply both cheek color and lipstick effects"""
        try:
            result = image.copy()
            messages = []
            
            # Apply cheek color jika ada
            if cheek_hex:
                result, cheek_msg = self.apply_cheek_color(result, cheek_hex)
                messages.append(cheek_msg)
            
            # Apply lipstick jika ada
            if lipstick_hex:
                result, lip_msg = self.apply_lipstick(result, lipstick_hex)
                messages.append(lip_msg)
            
            return result, " | ".join(messages) if messages else "No effects applied"
            
        except Exception as e:
            logger.error(f"Error applying combined effects: {str(e)}")
            return image, f"Combined effects error: {str(e)}"

    def process_live_frame(self, frame_data, cheek_color=None, lipstick_color=None):
        """Process live camera frame dengan efek real-time"""
        try:
            # Decode base64 image
            if isinstance(frame_data, str):
                # Remove data URL prefix if present
                if ',' in frame_data:
                    frame_data = frame_data.split(',')[1]
                
                img_data = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = frame_data
            
            if image is None:
                return None, "Invalid image data"
            
            # Apply effects
            if cheek_color or lipstick_color:
                result, message = self.apply_combined_effects(image, cheek_color, lipstick_color)
            else:
                result, message = image, "No effects applied"
            
            # Encode result back to base64
            _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return result_base64, message
            
        except Exception as e:
            logger.error(f"Error processing live frame: {str(e)}")
            return None, f"Processing error: {str(e)}"

# Global instance
live_face_tracker = LiveFaceTracker()

# Flask app untuk live processing
app = Flask(__name__)
CORS(app)

@app.route('/api/process-live-frame', methods=['POST'])
def process_live_frame():
    """Endpoint untuk processing frame live camera"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        frame_data = data['image']
        cheek_color = data.get('cheek_color')
        lipstick_color = data.get('lipstick_color')
        
        # Process frame
        result_base64, message = live_face_tracker.process_live_frame(
            frame_data, cheek_color, lipstick_color
        )
        
        if result_base64 is None:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'processed_image': f"data:image/jpeg;base64,{result_base64}",
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Error in process-live-frame endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)