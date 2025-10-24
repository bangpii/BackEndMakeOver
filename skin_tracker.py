import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class SkinTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh for precise facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Face Detection as fallback
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # More accurate model
            min_detection_confidence=0.7
        )

    def get_face_landmarks(self, image):
        """Get precise facial landmarks using MediaPipe Face Mesh"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            return results.multi_face_landmarks[0]
        except Exception as e:
            logger.error(f"Error in face landmarks: {str(e)}")
            return None

    def create_precise_face_mask(self, image, landmarks):
        """Create precise face mask using facial landmarks"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Define face regions using landmarks
            landmark_points = []
            
            # Cheek areas (more precise)
            left_cheek_indices = [116, 117, 118, 119, 120, 121, 128, 188, 245, 193, 55, 8]
            right_cheek_indices = [345, 346, 347, 348, 349, 350, 357, 377, 465, 422, 285, 8]
            
            # Forehead area
            forehead_indices = [10, 67, 69, 104, 108, 151, 337, 338, 297, 332, 284, 251]
            
            # Jawline and face contour
            face_contour_indices = list(range(10, 30)) + list(range(50, 70)) + list(range(100, 120))
            
            # Combine all indices
            all_indices = left_cheek_indices + right_cheek_indices + forehead_indices + face_contour_indices
            
            for idx in all_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append([x, y])
            
            if len(landmark_points) > 2:
                # Create convex hull for face area
                hull = cv2.convexHull(np.array(landmark_points))
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Apply morphological operations to smooth the mask
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (25, 25), 0)
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating face mask: {str(e)}")
            return mask

    def analyze_skin_tone_precise(self, image):
        """Analyze skin tone with precise facial region detection"""
        try:
            landmarks = self.get_face_landmarks(image)
            
            if landmarks is None:
                # Fallback to face detection
                return self.analyze_skin_tone_fallback(image)
            
            # Create precise face mask
            face_mask = self.create_precise_face_mask(image, landmarks)
            
            if np.sum(face_mask) == 0:
                return self.analyze_skin_tone_fallback(image)
            
            # Extract skin samples from masked area
            skin_pixels = image[face_mask > 128]
            
            if len(skin_pixels) == 0:
                return self.analyze_skin_tone_fallback(image)
            
            # Convert to RGB for analysis
            skin_rgb = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
            
            # Remove outliers using percentiles
            lower_percentile = np.percentile(skin_rgb, 5, axis=0)
            upper_percentile = np.percentile(skin_rgb, 95, axis=0)
            
            # Filter out extreme values
            mask = np.all((skin_rgb >= lower_percentile) & (skin_rgb <= upper_percentile), axis=1)
            filtered_skin = skin_rgb[mask]
            
            if len(filtered_skin) == 0:
                avg_skin_tone = np.mean(skin_rgb, axis=0).astype(int)
            else:
                avg_skin_tone = np.mean(filtered_skin, axis=0).astype(int)
            
            return avg_skin_tone.tolist(), "Skin tone analyzed precisely"
            
        except Exception as e:
            logger.error(f"Error in precise skin analysis: {str(e)}")
            return self.analyze_skin_tone_fallback(image)

    def analyze_skin_tone_fallback(self, image):
        """Fallback skin tone analysis using face detection"""
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
            
            # Expand region for better sampling
            padding_w = int(box_w * 0.2)
            padding_h = int(box_h * 0.1)
            
            x_start = max(0, x - padding_w)
            y_start = max(0, y - padding_h)
            x_end = min(w, x + box_w + padding_w)
            y_end = min(h, y + box_h + padding_h)
            
            face_region = image[y_start:y_end, x_start:x_end]
            
            if face_region.size == 0:
                return None, "Invalid face region"
            
            # Sample from specific facial areas
            region_h, region_w, _ = face_region.shape
            
            # Cheek areas
            left_cheek = face_region[region_h//3:2*region_h//3, region_w//8:region_w//3]
            right_cheek = face_region[region_h//3:2*region_h//3, 2*region_w//3:7*region_w//8]
            
            # Forehead
            forehead = face_region[region_h//10:region_h//4, region_w//3:2*region_w//3]
            
            samples = []
            for region in [left_cheek, right_cheek, forehead]:
                if region.size > 0:
                    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    # Sample center area
                    h_r, w_r, _ = region_rgb.shape
                    center_y, center_x = h_r//2, w_r//2
                    
                    # Sample 5x5 area
                    sample_size = 5
                    y1 = max(0, center_y - sample_size//2)
                    y2 = min(h_r, center_y + sample_size//2 + 1)
                    x1 = max(0, center_x - sample_size//2)
                    x2 = min(w_r, center_x + sample_size//2 + 1)
                    
                    sample_area = region_rgb[y1:y2, x1:x2]
                    if sample_area.size > 0:
                        samples.extend(sample_area.reshape(-1, 3))
            
            if not samples:
                return None, "No skin samples found"
            
            samples_array = np.array(samples)
            avg_skin_tone = np.mean(samples_array, axis=0).astype(int)
            
            return avg_skin_tone.tolist(), "Skin tone analyzed (fallback)"
            
        except Exception as e:
            logger.error(f"Error in fallback skin analysis: {str(e)}")
            return None, f"Analysis error: {str(e)}"

    def apply_foundation_precise(self, image, foundation_hex):
        """Apply foundation with precise facial mapping"""
        try:
            landmarks = self.get_face_landmarks(image)
            
            if landmarks is None:
                return self.apply_foundation_fallback(image, foundation_hex), "Applied with fallback"
            
            # Convert hex to BGR
            foundation_hex = foundation_hex.lstrip('#')
            foundation_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
            foundation_bgr = (foundation_rgb[2], foundation_rgb[1], foundation_rgb[0])
            
            # Create precise face mask
            face_mask = self.create_precise_face_mask(image, landmarks)
            
            if np.sum(face_mask) == 0:
                return self.apply_foundation_fallback(image, foundation_hex), "Applied with fallback"
            
            # Create foundation layer
            foundation_layer = np.zeros_like(image)
            foundation_layer[:] = foundation_bgr
            
            # Convert mask to float for blending
            mask_float = face_mask.astype(float) / 255.0
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Enhanced blending with different strengths for different areas
            blend_strength = 0.75  # 75% foundation
            
            # Create blended result
            blended = (foundation_layer * blend_strength + image * (1 - blend_strength))
            
            # Apply only to face area with smooth transition
            result_image = (blended * mask_float + image * (1 - mask_float)).astype(np.uint8)
            
            return result_image, "Foundation applied precisely"
            
        except Exception as e:
            logger.error(f"Error in precise foundation application: {str(e)}")
            return self.apply_foundation_fallback(image, foundation_hex), f"Applied with fallback: {str(e)}"

    def apply_foundation_fallback(self, image, foundation_hex):
        """Fallback foundation application"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if not results.detections:
                return image
            
            # Convert hex to BGR
            foundation_hex = foundation_hex.lstrip('#')
            foundation_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
            foundation_bgr = (foundation_rgb[2], foundation_rgb[1], foundation_rgb[0])
            
            result_image = image.copy()
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Expand region
                padding = int(box_h * 0.2)
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(w, x + box_w + padding)
                y_end = min(h, y + box_h + padding)
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
                
                # Smooth mask
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                
                # Apply foundation
                mask_float = mask.astype(float) / 255.0
                mask_float = np.stack([mask_float] * 3, axis=-1)
                
                foundation_layer = np.zeros_like(image)
                foundation_layer[:] = foundation_bgr
                
                blend_strength = 0.7
                blended = (foundation_layer * blend_strength + image * (1 - blend_strength))
                
                result_image = (blended * mask_float + result_image * (1 - mask_float)).astype(np.uint8)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error in fallback foundation application: {str(e)}")
            return image