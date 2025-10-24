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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe Selfie Segmentation untuk isolasi tubuh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # Define skin regions lebih komprehensif
        self.skin_regions = self._define_skin_regions()

    def _define_skin_regions(self):
        """Define comprehensive skin regions for face and body"""
        return {
            # Wajah - area kulit utama
            "forehead": [10, 67, 69, 104, 108, 151, 337, 338, 297, 332, 284],
            "left_cheek": [117, 118, 119, 100, 47, 126, 209, 49, 131, 134, 51, 4, 5, 50],
            "right_cheek": [346, 347, 348, 329, 277, 355, 429, 279, 360, 363, 281, 4, 5, 280],
            "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98],
            "chin": [200, 201, 194, 204, 202, 212, 216, 206, 92, 165, 167, 164, 393],
            "jaw_left": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397],
            "jaw_right": [397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172],
            "temple_left": [162, 21, 54, 103, 67, 109],
            "temple_right": [389, 251, 284, 332, 297, 338]
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

    def create_comprehensive_skin_mask(self, image, landmarks):
        """Create mask that covers all skin areas including face and visible body"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Collect all skin landmark points
            all_skin_points = []
            
            for region_name, indices in self.skin_regions.items():
                for idx in indices:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        all_skin_points.append([x, y])
            
            if len(all_skin_points) > 2:
                # Create convex hull for entire face
                hull = cv2.convexHull(np.array(all_skin_points))
                
                # Fill the convex hull
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Expand mask to include neck and upper body
                self._expand_mask_for_body(mask, landmarks, h, w)
                
                # Get body segmentation for more natural coverage
                body_mask = self.get_body_segmentation_mask(image)
                
                # Combine face mask with body segmentation
                combined_mask = cv2.bitwise_or(mask, body_mask)
                
                # Refine mask to exclude eyes, lips, and eyebrows
                combined_mask = self._refine_mask_exclusions(combined_mask, landmarks, h, w)
                
                # Smooth the mask for natural blending
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.GaussianBlur(combined_mask, (31, 31), 7)
                
                return combined_mask
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating comprehensive skin mask: {str(e)}")
            return mask

    def _expand_mask_for_body(self, mask, landmarks, h, w):
        """Expand mask to include neck and upper body"""
        try:
            # Get chin points for neck extension
            chin_points = []
            for idx in self.skin_regions["chin"]:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    chin_points.append([x, y])
            
            if chin_points:
                # Extend downward for neck
                chin_bottom = max(point[1] for point in chin_points)
                neck_extension = int(h * 0.3)  # Extend 30% of image height
                
                for x in range(w):
                    if mask[chin_bottom, x] > 0:
                        # Create vertical extension
                        for y in range(chin_bottom, min(chin_bottom + neck_extension, h)):
                            mask[y, x] = 255
                
                # Smooth the extension
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
        except Exception as e:
            logger.error(f"Error expanding mask for body: {str(e)}")

    def _refine_mask_exclusions(self, mask, landmarks, h, w):
        """Refine mask to exclude non-skin areas like eyes and lips"""
        try:
            # Define exclusion areas (eyes, lips, eyebrows)
            exclusion_areas = []
            
            # Left eye
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            # Right eye  
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            # Lips
            lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            exclusion_indices = left_eye_indices + right_eye_indices + lip_indices
            
            exclusion_points = []
            for idx in exclusion_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    exclusion_points.append([x, y])
            
            if len(exclusion_points) > 2:
                # Create exclusion hulls for each feature
                eye_lip_hull = cv2.convexHull(np.array(exclusion_points))
                
                # Subtract from mask
                cv2.fillConvexPoly(mask, eye_lip_hull, 0)
                
        except Exception as e:
            logger.error(f"Error refining mask exclusions: {str(e)}")
        
        return mask

    def get_body_segmentation_mask(self, image):
        """Get body segmentation mask to include visible skin areas"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(rgb_image)
            
            if results.segmentation_mask is not None:
                # Convert mask to binary with better threshold
                body_mask = (results.segmentation_mask * 255).astype(np.uint8)
                
                # Use adaptive threshold for better body detection
                _, body_mask = cv2.threshold(body_mask, 150, 255, cv2.THRESH_BINARY)
                
                # Expand body mask to ensure good coverage
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
                body_mask = cv2.GaussianBlur(body_mask, (25, 25), 5)
                
                return body_mask
            else:
                # Fallback: create mask covering central area
                h, w = image.shape[:2]
                body_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(body_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
                return body_mask
                
        except Exception as e:
            logger.error(f"Error in body segmentation: {str(e)}")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.uint8) * 255

    def analyze_skin_tone_precise(self, image):
        """Analyze skin tone with comprehensive skin detection"""
        try:
            # Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Method 1: Face landmarks-based skin analysis
            landmarks = self.get_face_landmarks(image)
            if landmarks:
                skin_mask = self.create_comprehensive_skin_mask(image, landmarks)
                if np.sum(skin_mask) > 5000:  # Minimum skin area threshold
                    skin_pixels = rgb_image[skin_mask > 128]
                    if len(skin_pixels) > 500:
                        skin_tone = self.calculate_skin_tone(skin_pixels)
                        if skin_tone:
                            return skin_tone, "Skin tone analyzed precisely with comprehensive detection"
            
            # Method 2: Advanced color-based skin detection
            skin_mask = self.detect_skin_color_advanced(rgb_image)
            if np.sum(skin_mask) > 5000:
                skin_pixels = rgb_image[skin_mask > 128]
                if len(skin_pixels) > 500:
                    skin_tone = self.calculate_skin_tone(skin_pixels)
                    if skin_tone:
                        return skin_tone, "Skin tone analyzed (advanced color detection)"
            
            return None, "No sufficient skin area detected. Please ensure face is clearly visible."
            
        except Exception as e:
            logger.error(f"Error in precise skin analysis: {str(e)}")
            return None, f"Analysis error: {str(e)}"

    def detect_skin_color_advanced(self, img_rgb):
        """Advanced skin color detection using multiple color spaces"""
        try:
            # YCrCb range for skin detection (most effective)
            img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
            
            # HSV range for skin detection
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([25, 170, 255], dtype=np.uint8)
            mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            
            # RGB range for skin tones
            lower_rgb = np.array([95, 40, 20], dtype=np.uint8)
            upper_rgb = np.array([255, 210, 180], dtype=np.uint8)
            mask_rgb = cv2.inRange(img_rgb, lower_rgb, upper_rgb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
            skin_mask = cv2.bitwise_and(skin_mask, mask_rgb)
            
            # Clean up and enhance mask
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close)
            skin_mask = cv2.GaussianBlur(skin_mask, (11, 11), 3)
            
            return skin_mask
            
        except Exception as e:
            logger.error(f"Error in advanced skin color detection: {str(e)}")
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    def calculate_skin_tone(self, skin_pixels):
        """Calculate average skin tone from pixel samples with advanced filtering"""
        try:
            if len(skin_pixels) == 0:
                return None
                
            # Convert to numpy array
            skin_samples = skin_pixels.reshape(-1, 3)
            
            # Remove extreme outliers using percentile method
            lower_percentile = np.percentile(skin_samples, 5, axis=0)
            upper_percentile = np.percentile(skin_samples, 95, axis=0)
            
            # Filter samples within reasonable range
            mask = np.all((skin_samples >= lower_percentile) & (skin_samples <= upper_percentile), axis=1)
            filtered_skin = skin_samples[mask]
            
            if len(filtered_skin) > 0:
                # Use weighted average (give more weight to mid-tones)
                weights = np.exp(-0.5 * ((filtered_skin - 128) / 64) ** 2)
                weights = np.prod(weights, axis=1)
                weights = weights / np.sum(weights)
                
                avg_skin_tone = np.average(filtered_skin, axis=0, weights=weights).astype(int)
            else:
                # Fallback to median
                avg_skin_tone = np.median(skin_samples, axis=0).astype(int)
            
            return avg_skin_tone.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating skin tone: {str(e)}")
            return None

    def apply_foundation_to_skin(self, image, foundation_hex):
        """Apply foundation to skin areas with natural, Instagram-like blending"""
        try:
            # Convert hex to RGB
            foundation_hex = foundation_hex.lstrip('#')
            if len(foundation_hex) != 6:
                return image, "Invalid foundation color"
                
            target_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
            target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
            
            # Convert image to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)
            
            # Get comprehensive skin mask
            landmarks = self.get_face_landmarks(image)
            if landmarks:
                skin_mask = self.create_comprehensive_skin_mask(image, landmarks)
            else:
                # Fallback to advanced color-based skin detection
                skin_mask = self.detect_skin_color_advanced(img_rgb.astype(np.uint8))
            
            if np.sum(skin_mask) == 0:
                return image, "No skin area detected for foundation application"
            
            # Create high-quality mask for blending
            mask_float = skin_mask.astype(float) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (21, 21), 5)
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Create foundation layer
            foundation_layer = np.ones_like(img_rgb)
            foundation_layer[:, :, 0] = target_rgb[0]  # R
            foundation_layer[:, :, 1] = target_rgb[1]  # G  
            foundation_layer[:, :, 2] = target_rgb[2]  # B
            
            # Advanced blending techniques
            result = self._advanced_color_blending(img_rgb, foundation_layer, mask_float, target_rgb)
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            return result_bgr, "Foundation applied naturally to all skin areas"
            
        except Exception as e:
            logger.error(f"Error applying foundation: {str(e)}")
            return image, f"Application error: {str(e)}"

    def _advanced_color_blending(self, original, foundation, mask, target_rgb):
        """Advanced color blending for natural foundation application"""
        try:
            result = original.copy()
            
            # Calculate color transformation
            original_avg = np.average(original.reshape(-1, 3), axis=0)
            target_avg = np.array(target_rgb, dtype=float)
            
            # Color correction factors
            color_ratio = target_avg / (original_avg + 1e-6)  # Avoid division by zero
            
            # Apply color correction with smooth masking
            for c in range(3):
                original_channel = original[:, :, c]
                
                # Smart color adjustment that preserves texture
                adjusted_channel = original_channel * color_ratio[c]

                # Blend with original using the mask
                blended_channel = original_channel * (1 - mask[:, :, c]) + adjusted_channel * mask[:, :, c]
                
                result[:, :, c] = np.clip(blended_channel, 0, 255)
            
            # Final smoothing for natural look
            result_uint8 = result.astype(np.uint8)
            
            # Apply slight Gaussian blur only to blended areas for seamless look
            blurred = cv2.GaussianBlur(result_uint8, (3, 3), 0.5)
            
            # Blend the slight blur only in foundation areas
            final_result = result_uint8.copy()
            blur_mask = (mask * 255).astype(np.uint8)[:, :, 0]
            final_result[blur_mask > 128] = blurred[blur_mask > 128]
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in advanced color blending: {str(e)}")
            return original.astype(np.uint8)

# Global instance
skin_tracker = PrecisionSkinTracker()