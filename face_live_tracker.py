import cv2
import numpy as np
import mediapipe as mp
import logging
import base64

logger = logging.getLogger(__name__)

class OptimizedLiveFaceTracker:
    def __init__(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Cache untuk performa
            self.last_landmarks = None
            self.last_frame_hash = None
            
            logger.info("OptimizedLiveFaceTracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OptimizedLiveFaceTracker: {str(e)}")
            self.face_mesh = None

    def get_face_landmarks_optimized(self, image):
        """Optimized face landmarks detection dengan caching"""
        try:
            # Simple frame hash untuk deteksi perubahan
            current_hash = hash(image.tobytes())
            if (self.last_landmarks and 
                self.last_frame_hash == current_hash):
                return self.last_landmarks
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                self.last_landmarks = results.multi_face_landmarks[0]
                self.last_frame_hash = current_hash
            else:
                self.last_landmarks = None
                
            return self.last_landmarks
            
        except Exception as e:
            logger.error(f"Error in optimized face landmarks: {str(e)}")
            return None

    def process_frame_fast(self, frame_data, cheek_color=None, lipstick_color=None):
        """Fast frame processing untuk real-time effects"""
        try:
            # Decode base64 image
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Invalid image data"
            
            # Resize untuk performa
            h, w = image.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = 640
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            result = image.copy()
            
            if cheek_color or lipstick_color:
                result = self.apply_fast_effects(result, cheek_color, lipstick_color)
            
            # Encode result
            _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{result_base64}", "Effects applied"
            
        except Exception as e:
            logger.error(f"Error in fast frame processing: {str(e)}")
            return None, f"Error: {str(e)}"

    def apply_fast_effects(self, image, cheek_color, lipstick_color):
        """Fast effects application tanpa complex face detection"""
        try:
            result = image.copy()
            h, w = image.shape[:2]
            
            if cheek_color:
                cheek_hex = cheek_color.lstrip('#')
                cheek_rgb = tuple(int(cheek_hex[i:i+2], 16) for i in (0, 2, 4))
                cheek_bgr = (cheek_rgb[2], cheek_rgb[1], cheek_rgb[0])
                
                # Simple cheek overlay
                cheek_overlay = np.zeros_like(image)
                
                # Predefined cheek positions (simple approach)
                left_cheek = (int(w*0.2), int(h*0.45))
                right_cheek = (int(w*0.8), int(h*0.45))
                radius = int(min(w, h) * 0.1)
                
                cv2.circle(cheek_overlay, left_cheek, radius, cheek_bgr, -1)
                cv2.circle(cheek_overlay, right_cheek, radius, cheek_bgr, -1)
                
                # Simple blend
                alpha = 0.3
                result = cv2.addWeighted(result, 1, cheek_overlay, alpha, 0)
            
            if lipstick_color:
                lip_hex = lipstick_color.lstrip('#')
                lip_rgb = tuple(int(lip_hex[i:i+2], 16) for i in (0, 2, 4))
                lip_bgr = (lip_rgb[2], lip_rgb[1], lip_rgb[0])
                
                # Simple lip overlay
                lip_overlay = np.zeros_like(image)
                lip_center = (w//2, int(h*0.6))
                lip_size = (int(w*0.3), int(h*0.05))
                
                cv2.ellipse(lip_overlay, lip_center, lip_size, 0, 0, 360, lip_bgr, -1)
                
                # Simple blend
                alpha = 0.5
                result = cv2.addWeighted(result, 1, lip_overlay, alpha, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying fast effects: {str(e)}")
            return image

# Global instance
optimized_tracker = OptimizedLiveFaceTracker()