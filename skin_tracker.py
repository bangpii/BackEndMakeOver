import cv2
import numpy as np
import mediapipe as mp
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedSkinTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh for precise facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Changed to True for better accuracy
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Initialize MediaPipe Face Detection as fallback
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )

    def detect_skin_protected(self, img_rgb):
        """Advanced skin detection with protection for sensitive areas"""
        try:
            # Method 1: YCrCb dengan range yang lebih akurat
            img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Method 2: HSV dengan range yang lebih natural
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
            upper_hsv = np.array([25, 160, 245], dtype=np.uint8)
            mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            
            # Method 3: RGB rules yang lebih selektif
            r, g, b = cv2.split(img_rgb)
            mask_rgb = ((r > 80) & (g > 40) & (b > 20) & 
                        ((cv2.max(r, cv2.max(g, b)) - cv2.min(r, cv2.min(g, b))) > 20) & 
                        (np.abs(r - g) > 10) & (r > g) & (r > b)).astype(np.uint8) * 255
            
            # Combine masks dengan priority
            combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
            combined_mask = cv2.bitwise_or(combined_mask, mask_rgb)
            
            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Error in skin detection: {str(e)}")
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    def refine_mask_advanced(self, mask, img_rgb):
        """Refine mask with advanced edge-aware processing"""
        try:
            # Gaussian blur untuk edges yang sangat halus
            mask = cv2.GaussianBlur(mask, (9, 9), 2)
            
            # Normalize mask untuk soft transition
            mask_float = mask.astype(float) / 255.0
            
            # Create edge-aware mask
            edges = cv2.Canny(img_rgb, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            edges_mask = (edges == 0).astype(float)
            
            # Combine dengan edge protection
            mask_float = mask_float * edges_mask
            
            # Soft threshold untuk natural look
            mask_float = np.clip(mask_float * 1.1, 0, 0.95)
            
            return (mask_float * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error refining mask: {str(e)}")
            return mask

    def get_face_landmarks_precise(self, image):
        """Get precise facial landmarks with enhanced accuracy"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            return results.multi_face_landmarks[0]
        except Exception as e:
            logger.error(f"Error in face landmarks: {str(e)}")
            return None

    def create_smooth_face_mask(self, image, landmarks):
        """Create smooth face mask without triangle artifacts"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Define facial regions dengan points yang lebih natural
            landmark_points = []
            
            # Gunakan points yang lebih natural untuk contour wajah
            # Face oval points (mengelilingi wajah)
            face_oval_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            # Cheek areas untuk coverage yang lebih baik
            cheek_indices = [
                116, 117, 118, 119, 120, 121, 128, 188, 245, 193, 
                346, 347, 348, 349, 350, 357, 377, 465, 422
            ]
            
            # Forehead area
            forehead_indices = [67, 69, 104, 108, 151, 337, 338, 297, 332]
            
            # Combine all indices
            all_indices = list(set(face_oval_indices + cheek_indices + forehead_indices))
            
            # Collect points dengan smoothing
            for idx in all_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append([x, y])
            
            if len(landmark_points) > 2:
                # Create smooth contour dengan convex hull
                hull = cv2.convexHull(np.array(landmark_points))
                
                # Fill the convex hull
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Smooth the mask dengan morphological operations
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                
                # Open untuk remove noise kecil
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                
                # Close untuk fill holes dan smooth edges
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                
                # Advanced Gaussian blur untuk smooth transition
                mask = cv2.GaussianBlur(mask, (25, 25), 8)
                
                # Ensure mask covers hair gaps dengan dilation ringan
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, kernel_dilate, iterations=1)
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating face mask: {str(e)}")
            return mask

    def enhance_hair_gap_coverage(self, mask, image):
        """Enhance coverage for hair gaps and fine details"""
        try:
            # Convert to grayscale untuk edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect fine edges (hair gaps)
            edges_fine = cv2.Canny(gray, 10, 30)
            
            # Dilate fine edges sedikit untuk menutup gap kecil
            kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_dilated = cv2.dilate(edges_fine, kernel_fine, iterations=1)
            
            # Combine dengan original mask
            enhanced_mask = cv2.bitwise_or(mask, edges_dilated)
            
            # Smooth the enhanced mask
            enhanced_mask = cv2.GaussianBlur(enhanced_mask, (7, 7), 2)
            
            return enhanced_mask
            
        except Exception as e:
            logger.error(f"Error enhancing hair gap coverage: {str(e)}")
            return mask

    def analyze_skin_tone_advanced(self, image):
        """Advanced skin tone analysis with multiple methods"""
        try:
            # Convert to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Method 1: Face landmarks-based analysis
            landmarks = self.get_face_landmarks_precise(image)
            if landmarks:
                face_mask = self.create_smooth_face_mask(image, landmarks)
                if np.sum(face_mask) > 0:
                    skin_pixels = rgb_image[face_mask > 128]
                    if len(skin_pixels) > 0:
                        # Advanced color analysis dengan outlier removal
                        skin_samples = skin_pixels.reshape(-1, 3)
                        
                        # Remove extreme outliers
                        lower_percentile = np.percentile(skin_samples, 2, axis=0)
                        upper_percentile = np.percentile(skin_samples, 98, axis=0)
                        
                        mask = np.all((skin_samples >= lower_percentile) & 
                                    (skin_samples <= upper_percentile), axis=1)
                        filtered_skin = skin_samples[mask]
                        
                        if len(filtered_skin) > 0:
                            avg_skin_tone = np.median(filtered_skin, axis=0).astype(int)
                            return avg_skin_tone.tolist(), "Skin tone analyzed precisely"
            
            # Method 2: Advanced skin detection fallback
            skin_mask = self.detect_skin_protected(rgb_image)
            refined_mask = self.refine_mask_advanced(skin_mask, rgb_image)
            
            if np.sum(refined_mask) > 0:
                skin_pixels = rgb_image[refined_mask > 100]
                if len(skin_pixels) > 0:
                    avg_skin_tone = np.median(skin_pixels, axis=0).astype(int)
                    return avg_skin_tone.tolist(), "Skin tone analyzed (advanced detection)"
            
            # Method 3: Face detection fallback
            return self.analyze_skin_tone_fallback(image)
            
        except Exception as e:
            logger.error(f"Error in advanced skin analysis: {str(e)}")
            return self.analyze_skin_tone_fallback(image)

    def analyze_skin_tone_fallback(self, image):
        """Fallback skin tone analysis"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if not results.detections:
                return None, "No face detected"
            
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # Expanded region dengan padding
            padding_w = int(box_w * 0.25)
            padding_h = int(box_h * 0.15)
            
            x_start = max(0, x - padding_w)
            y_start = max(0, y - padding_h)
            x_end = min(w, x + box_w + padding_w)
            y_end = min(h, y + box_h + padding_h)
            
            face_region = image[y_start:y_end, x_start:x_end]
            
            if face_region.size == 0:
                return None, "Invalid face region"
            
            # Advanced sampling dari multiple regions
            region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            region_h, region_w, _ = region_rgb.shape
            
            samples = []
            sampling_points = [
                (region_h//3, region_w//4),    # Left cheek
                (region_h//3, 3*region_w//4),  # Right cheek
                (region_h//6, region_w//2),    # Forehead
                (2*region_h//3, region_w//2),  # Chin
            ]
            
            for y_pos, x_pos in sampling_points:
                # Sample 7x7 area untuk lebih banyak data
                y1 = max(0, y_pos - 3)
                y2 = min(region_h, y_pos + 4)
                x1 = max(0, x_pos - 3)
                x2 = min(region_w, x_pos + 4)
                
                sample_area = region_rgb[y1:y2, x1:x2]
                if sample_area.size > 0:
                    samples.extend(sample_area.reshape(-1, 3))
            
            if not samples:
                return None, "No skin samples found"
            
            samples_array = np.array(samples)
            # Use median untuk menghindari outlier
            avg_skin_tone = np.median(samples_array, axis=0).astype(int)
            
            return avg_skin_tone.tolist(), "Skin tone analyzed (fallback)"
            
        except Exception as e:
            logger.error(f"Error in fallback skin analysis: {str(e)}")
            return None, f"Analysis error: {str(e)}"

    def apply_natural_skin_tone(self, image, foundation_hex, original_image_path=None):
        """Apply foundation with natural blending and enhanced coverage"""
        try:
            # Convert hex to RGB
            foundation_hex = foundation_hex.lstrip('#')
            target_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
            target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
            
            # Convert image to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get advanced skin mask
            landmarks = self.get_face_landmarks_precise(image)
            if landmarks:
                face_mask = self.create_smooth_face_mask(image, landmarks)
                # Enhance coverage for hair gaps
                face_mask = self.enhance_hair_gap_coverage(face_mask, image)
            else:
                # Fallback to skin detection
                face_mask = self.detect_skin_protected(img_rgb)
                face_mask = self.refine_mask_advanced(face_mask, img_rgb)
            
            # Normalize mask untuk alpha blending yang sangat halus
            mask_float = face_mask.astype(float) / 255.0
            
            # Preserve original image texture and details
            result = img_rgb.copy().astype(float)
            
            # Apply natural color blending
            for c in range(3):
                original_channel = img_rgb[:, :, c].astype(float)
                target_value = target_rgb[c]
                
                # Natural blending dengan texture preservation
                blend_strength = 0.7  # Optimal untuk natural look
                blended = original_channel * (1 - mask_float * blend_strength) + \
                         target_value * mask_float * blend_strength
                
                # Maintain original texture details
                original_detail = original_channel - cv2.GaussianBlur(original_channel, (0, 0), 1)
                blended = blended + original_detail * 0.3
                
                result[:, :, c] = np.clip(blended, 0, 255)
            
            # Final smoothing untuk natural look
            result = cv2.GaussianBlur(result, (0, 0), 0.5)  # Reduced blur
            result_uint8 = result.astype(np.uint8)
            
            # Convert back to BGR
            result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
            
            return result_bgr, "Foundation applied naturally with enhanced coverage"
            
        except Exception as e:
            logger.error(f"Error applying natural skin tone: {str(e)}")
            return image, f"Application error: {str(e)}"

# Global instance
skin_tracker = AdvancedSkinTracker()