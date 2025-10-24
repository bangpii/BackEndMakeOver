import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class PrecisionSkinTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh untuk deteksi wajah yang presisi
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Initialize MediaPipe Selfie Segmentation untuk isolasi tubuh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # Model general (0=landscape, 1=general)
        )

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

    def create_precise_skin_mask(self, image, landmarks):
        """Create mask that only covers skin areas, excluding hair, eyes, mouth"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Define skin regions only (cheeks, forehead, nose bridge, chin)
            skin_indices = []
            
            # Cheeks - area kulit pipi
            left_cheek = [117, 118, 119, 100, 47, 126, 209, 49, 131, 134, 51, 4, 5]
            right_cheek = [346, 347, 348, 329, 277, 355, 429, 279, 360, 363, 281, 4, 5]
            
            # Forehead - dahi
            forehead = [10, 67, 69, 104, 108, 151, 337, 338, 297, 332, 284]
            
            # Nose bridge - batang hidung (hindari lubang hidung)
            nose_bridge = [168, 6, 197, 195, 5, 4, 1, 19, 94]
            
            # Chin - dagu
            chin = [200, 201, 194, 204, 202, 212, 216, 206, 92, 165, 167, 164]
            
            # Combine all skin indices
            skin_indices = left_cheek + right_cheek + forehead + nose_bridge + chin
            
            # Remove duplicates
            skin_indices = list(set(skin_indices))
            
            landmark_points = []
            for idx in skin_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append([x, y])
            
            if len(landmark_points) > 2:
                # Create convex hull for skin area only
                hull = cv2.convexHull(np.array(landmark_points))
                
                # Fill the convex hull
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Remove non-skin areas using selfie segmentation
                body_mask = self.get_body_segmentation_mask(image)
                
                # Combine face skin mask with body segmentation
                combined_mask = cv2.bitwise_and(mask, body_mask)
                
                # Smooth the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.GaussianBlur(combined_mask, (21, 21), 5)
                
                return combined_mask
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating skin mask: {str(e)}")
            return mask

    def get_body_segmentation_mask(self, image):
        """Get body segmentation mask to exclude hair and background"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(rgb_image)
            
            if results.segmentation_mask is not None:
                # Convert mask to binary
                body_mask = (results.segmentation_mask * 255).astype(np.uint8)
                
                # Enhance mask to better capture skin tones
                _, body_mask = cv2.threshold(body_mask, 200, 255, cv2.THRESH_BINARY)
                
                # Smooth the mask
                body_mask = cv2.GaussianBlur(body_mask, (15, 15), 5)
                
                return body_mask
            else:
                # Fallback: create a simple mask covering most of the image
                h, w = image.shape[:2]
                return np.ones((h, w), dtype=np.uint8) * 255
                
        except Exception as e:
            logger.error(f"Error in body segmentation: {str(e)}")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.uint8) * 255

    def detect_skin_color_ranges(self, img_rgb):
        """Detect skin color using multiple color spaces"""
        try:
            # YCrCb range for skin detection
            img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
            
            # HSV range for skin detection
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
            upper_hsv = np.array([25, 150, 255], dtype=np.uint8)
            mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return skin_mask
            
        except Exception as e:
            logger.error(f"Error in skin color detection: {str(e)}")
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    def analyze_skin_tone_precise(self, image):
        """Analyze skin tone with precise skin-only detection"""
        try:
            # Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Method 1: Face landmarks-based skin analysis
            landmarks = self.get_face_landmarks(image)
            if landmarks:
                skin_mask = self.create_precise_skin_mask(image, landmarks)
                if np.sum(skin_mask) > 1000:  # Ensure we have enough skin area
                    skin_pixels = rgb_image[skin_mask > 128]
                    if len(skin_pixels) > 100:
                        return self.calculate_skin_tone(skin_pixels), "Skin tone analyzed precisely"
            
            # Method 2: Color-based skin detection
            skin_mask = self.detect_skin_color_ranges(rgb_image)
            if np.sum(skin_mask) > 1000:
                skin_pixels = rgb_image[skin_mask > 128]
                if len(skin_pixels) > 100:
                    return self.calculate_skin_tone(skin_pixels), "Skin tone analyzed (color detection)"
            
            return None, "No sufficient skin area detected"
            
        except Exception as e:
            logger.error(f"Error in precise skin analysis: {str(e)}")
            return None, f"Analysis error: {str(e)}"

    def calculate_skin_tone(self, skin_pixels):
        """Calculate average skin tone from pixel samples"""
        try:
            # Convert to numpy array
            skin_samples = skin_pixels.reshape(-1, 3)
            
            # Remove outliers using IQR method
            Q1 = np.percentile(skin_samples, 25, axis=0)
            Q3 = np.percentile(skin_samples, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter outliers
            mask = np.all((skin_samples >= lower_bound) & (skin_samples <= upper_bound), axis=1)
            filtered_skin = skin_samples[mask]
            
            if len(filtered_skin) > 0:
                # Use median for robustness against outliers
                avg_skin_tone = np.median(filtered_skin, axis=0).astype(int)
            else:
                # Fallback to mean if no pixels after filtering
                avg_skin_tone = np.mean(skin_samples, axis=0).astype(int)
            
            return avg_skin_tone.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating skin tone: {str(e)}")
            return None

    def apply_foundation_to_skin(self, image, foundation_hex):
        """Apply foundation only to skin areas with precise color matching"""
        try:
            # Convert hex to RGB
            foundation_hex = foundation_hex.lstrip('#')
            if len(foundation_hex) != 6:
                return image, "Invalid foundation color"
                
            target_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
            target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
            
            # Convert image to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get precise skin mask
            landmarks = self.get_face_landmarks(image)
            if landmarks:
                skin_mask = self.create_precise_skin_mask(image, landmarks)
            else:
                # Fallback to color-based skin detection
                skin_mask = self.detect_skin_color_ranges(img_rgb)
            
            if np.sum(skin_mask) == 0:
                return image, "No skin area detected for foundation application"
            
            # Normalize mask for smooth blending
            mask_float = skin_mask.astype(float) / 255.0
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Create result image
            result = img_rgb.copy().astype(float)
            
            # Apply foundation with natural color preservation
            for c in range(3):
                original_channel = img_rgb[:, :, c].astype(float)
                target_value = target_rgb[c]
                
                # Smart blending that preserves skin texture
                blend_strength = 0.8  # Strong foundation coverage
                
                # Calculate the difference and apply smoothly
                color_diff = target_value - original_channel
                adjusted_channel = original_channel + (color_diff * mask_float[:, :, c] * blend_strength)
                
                result[:, :, c] = np.clip(adjusted_channel, 0, 255)
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            return result_bgr, "Foundation applied precisely to skin areas"
            
        except Exception as e:
            logger.error(f"Error applying foundation: {str(e)}")
            return image, f"Application error: {str(e)}"

# Global instance
skin_tracker = PrecisionSkinTracker()