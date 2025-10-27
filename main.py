from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import logging
import os
import uuid
from datetime import datetime
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# LIVE FACE TRACKER CLASS
# =============================

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
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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

# =============================
# SKIN TRACKER CLASS 
# =============================

class SkinTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def analyze_skin_tone_precise(self, image):
        """Analyze skin tone dengan deteksi wajah yang presisi"""
        try:
            # Convert image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process dengan MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None, "No face detected"
            
            # Get the first face landmarks
            landmarks = results.multi_face_landmarks[0]
            
            h, w = image.shape[:2]
            
            # Define precise skin regions (pipi, dahi, dagu)
            skin_regions = []
            
            # Cheek regions
            cheek_indices = [117, 118, 119, 100, 47, 126, 209, 49, 346, 347, 348, 329, 277, 355, 429, 279]
            for idx in cheek_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Take small region around each point
                    skin_regions.append(image[max(0, y-5):min(h, y+5), max(0, x-5):min(w, x+5)])
            
            # Forehead region
            forehead_indices = [10, 67, 69, 104, 108, 151, 337, 338]
            for idx in forehead_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    skin_regions.append(image[max(0, y-3):min(h, y+3), max(0, x-3):min(w, x+3)])
            
            # Filter out empty regions
            skin_regions = [region for region in skin_regions if region.size > 0]
            
            if not skin_regions:
                return None, "No skin regions detected"
            
            # Combine all skin regions
            combined_skin = np.vstack([region.reshape(-1, 3) for region in skin_regions])
            
            # Calculate average skin tone (BGR format)
            avg_skin_tone = np.mean(combined_skin, axis=0)
            
            # Convert to RGB format for consistency
            avg_skin_tone_rgb = (int(avg_skin_tone[2]), int(avg_skin_tone[1]), int(avg_skin_tone[0]))
            
            return avg_skin_tone_rgb, "Skin tone analyzed successfully"
            
        except Exception as e:
            logger.error(f"Error in skin tone analysis: {str(e)}")
            return None, f"Analysis error: {str(e)}"
    
    def apply_foundation_to_skin(self, image, foundation_hex):
        """Apply foundation color ke area kulit wajah"""
        try:
            # Convert hex to BGR (OpenCV format)
            foundation_hex = foundation_hex.lstrip('#')
            if len(foundation_hex) != 6:
                return image, "Invalid foundation color"
                
            # Convert HEX to BGR
            r = int(foundation_hex[0:2], 16)
            g = int(foundation_hex[2:4], 16)
            b = int(foundation_hex[4:6], 16)
            foundation_bgr = (b, g, r)
            
            # Convert image to RGB for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return image, "No face detected for foundation application"
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Create mask for face skin areas
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Define face contour points for mask
            face_contour_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
            points = []
            for idx in face_contour_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])
            
            if len(points) > 2:
                # Create convex hull for face
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
                
                # Exclude eyes, mouth, and nostrils
                self._exclude_facial_features(mask, landmarks, h, w)
            
            # Apply foundation with smooth blending
            result = image.copy()
            mask_float = mask.astype(float) / 255.0
            mask_float = cv2.GaussianBlur(mask_float, (51, 51), 15)
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Blend foundation color
            foundation_layer = np.ones_like(image, dtype=float)
            foundation_layer[:, :, 0] = foundation_bgr[0]
            foundation_layer[:, :, 1] = foundation_bgr[1]
            foundation_layer[:, :, 2] = foundation_bgr[2]
            
            # Soft blending
            blended = image.astype(float) * (1 - mask_float) + foundation_layer * mask_float * 0.7 + image.astype(float) * mask_float * 0.3
            
            result = np.clip(blended, 0, 255).astype(np.uint8)
            
            return result, "Foundation applied successfully"
            
        except Exception as e:
            logger.error(f"Error applying foundation: {str(e)}")
            return image, f"Application error: {str(e)}"
    
    def _exclude_facial_features(self, mask, landmarks, h, w):
        """Exclude eyes, mouth, and nostrils from foundation application"""
        try:
            # Eye regions
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Mouth region
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
            
            exclusion_indices = left_eye_indices + right_eye_indices + mouth_indices
            
            exclusion_points = []
            for idx in exclusion_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    exclusion_points.append([x, y])
            
            if len(exclusion_points) > 2:
                exclusion_hull = cv2.convexHull(np.array(exclusion_points))
                cv2.fillConvexPoly(mask, exclusion_hull, 0)
                
        except Exception as e:
            logger.error(f"Error excluding facial features: {str(e)}")

# =============================
# FASTAPI APP
# =============================

# Global instances
live_face_tracker = LiveFaceTracker()
skin_tracker = SkinTracker()

app = FastAPI(title="MakeOver Backend")

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if not exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Store user sessions
user_sessions = {}

# Skin tone categories
SKIN_TONE_CATEGORIES = {
    "COOL": [
        {"name": "Fair", "hex": "#F9E6E6", "rgb": (249, 230, 230)},
        {"name": "Warm Vanilla", "hex": "#FBE8D0", "rgb": (251, 232, 208)},
        {"name": "Peach", "hex": "#FFD5B8", "rgb": (255, 213, 184)},
        {"name": "Almond", "hex": "#E6B89C", "rgb": (230, 184, 156)},
        {"name": "Walnut", "hex": "#C9A17D", "rgb": (201, 161, 125)},
        {"name": "Chest-nut", "hex": "#B07B50", "rgb": (176, 123, 80)},
    ],
    "NEUTRAL": [
        {"name": "Chantilly", "hex": "#FCEFE8", "rgb": (252, 239, 232)},
        {"name": "Shell", "hex": "#F2E3D5", "rgb": (242, 227, 213)},
        {"name": "Sand", "hex": "#E7CBA9", "rgb": (231, 203, 169)},
        {"name": "Wheat", "hex": "#F5D7A5", "rgb": (245, 215, 165)},
        {"name": "Cappuccino", "hex": "#C9A97E", "rgb": (201, 169, 126)},
        {"name": "Cashew", "hex": "#D8B68A", "rgb": (216, 182, 138)},
    ],
    "WARM": [
        {"name": "Porce-lain", "hex": "#FFF3E8", "rgb": (255, 243, 232)},
        {"name": "Nude", "hex": "#F9DBC4", "rgb": (249, 219, 196)},
        {"name": "Honey", "hex": "#EAB676", "rgb": (234, 182, 118)},
        {"name": "Butter-Scotch", "hex": "#FFD18C", "rgb": (255, 209, 140)},
        {"name": "Golden", "hex": "#EFCB68", "rgb": (239, 203, 104)},
        {"name": "Caramel", "hex": "#D9A25F", "rgb": (217, 162, 95)},
    ]
}

def save_uploaded_file(file_contents, filename):
    """Save uploaded file to uploads directory"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_contents)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

def image_to_base64(image):
    """Convert OpenCV image to base64"""
    try:
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Encode with good quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

def find_best_matching_foundation(skin_tone_rgb):
    """Find the best matching foundation colors based on skin tone"""
    if not skin_tone_rgb:
        return {"category": "NEUTRAL", "matches": SKIN_TONE_CATEGORIES["NEUTRAL"][:3]}
    
    r, g, b = skin_tone_rgb
    skin_tone_array = np.array([r, g, b])
    
    best_matches = []
    
    # Calculate color distance for all foundation shades
    for category, shades in SKIN_TONE_CATEGORIES.items():
        for shade in shades:
            shade_rgb = np.array(shade["rgb"])
            # Calculate Euclidean distance in RGB space
            distance = np.linalg.norm(skin_tone_array - shade_rgb)
            best_matches.append({
                "category": category,
                "shade": shade,
                "distance": distance
            })
    
    # Sort by closest match
    best_matches.sort(key=lambda x: x["distance"])
    
    # Get top matches
    top_matches = best_matches[:6]
    
    # Group by category
    category_matches = {}
    for match in top_matches:
        category = match["category"]
        if category not in category_matches:
            category_matches[category] = []
        category_matches[category].append(match["shade"])
    
    # Get best from each category
    recommended_matches = []
    for category, matches in category_matches.items():
        recommended_matches.extend(matches[:2])
    
    return {
        "primary_category": top_matches[0]["category"],
        "recommended_matches": recommended_matches[:4]
    }

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI backend!"}

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Read file contents
        contents = await file.read()
        
        # Check file size (prevent 414 error)
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        unique_filename = f"{session_id}_original{file_extension}"
        
        # Save original file
        original_path = save_uploaded_file(contents, unique_filename)
        
        if not original_path:
            raise HTTPException(status_code=500, detail="Failed to save file")
        
        # Decode image for processing
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store user session with proper image data
        original_base64 = image_to_base64(image)
        if not original_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        user_sessions[session_id] = {
            "original_path": original_path,
            "original_image_base64": original_base64,
            "original_image": image.copy(),  # Store the actual image array
            "upload_time": datetime.now().isoformat()
        }
        
        # Analyze skin tone
        skin_tone, message = skin_tracker.analyze_skin_tone_precise(image)
        
        foundation_matches = None
        if skin_tone:
            foundation_matches = find_best_matching_foundation(skin_tone)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Photo uploaded successfully",
            "skin_tone_rgb": skin_tone,
            "skin_tone_hex": f"#{skin_tone[0]:02x}{skin_tone[1]:02x}{skin_tone[2]:02x}" if skin_tone else None,
            "foundation_recommendations": foundation_matches,
            "processed_image": f"data:image/jpeg;base64,{original_base64}"
        }
        
    except Exception as e:
        logger.error(f"Error in upload-photo endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/apply-foundation")
async def apply_foundation(
    file: UploadFile = File(...),
    foundation_hex: str = Form("#F9E6E6"),
    session_id: str = Form(None)
):
    try:
        # Read image file
        contents = await file.read()
        
        # Check file size
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply foundation with precise skin tracking
        result_image, message = skin_tracker.apply_foundation_to_skin(image, foundation_hex)
        
        result_base64 = image_to_base64(result_image)
        if not result_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        return {
            "success": True,
            "message": message,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "applied_foundation": foundation_hex
        }
        
    except Exception as e:
        logger.error(f"Error in apply-foundation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/reset-to-original")
async def reset_to_original(session_id: str = Form(...)):
    try:
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = user_sessions[session_id]
        
        # Get the original image base64 from session
        original_base64 = session_data["original_image_base64"]
        
        if not original_base64:
            raise HTTPException(status_code=500, detail="Original image not available")
        
        return {
            "success": True,
            "message": "Reset to original photo",
            "processed_image": f"data:image/jpeg;base64,{original_base64}"
        }
        
    except Exception as e:
        logger.error(f"Error in reset-to-original endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze skin tone with precise tracking
        skin_tone, message = skin_tracker.analyze_skin_tone_precise(image)
        
        if skin_tone is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": message}
            )
        
        # Find matching foundation
        foundation_matches = find_best_matching_foundation(skin_tone)
        
        image_base64 = image_to_base64(image)
        if not image_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        return {
            "success": True,
            "message": message,
            "skin_tone_rgb": skin_tone,
            "skin_tone_hex": f"#{skin_tone[0]:02x}{skin_tone[1]:02x}{skin_tone[2]:02x}",
            "foundation_recommendations": foundation_matches,
            "processed_image": f"data:image/jpeg;base64,{image_base64}"
        }
        
    except Exception as e:
        logger.error(f"Error in analyze-skin endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/sessions")
async def get_sessions():
    """Debug endpoint to check active sessions"""
    return {
        "active_sessions": len(user_sessions),
        "sessions": list(user_sessions.keys())
    }

@app.post("/api/process-live-frame")
async def process_live_frame(
    image_data: str = Form(...),
    cheek_color: str = Form(None),
    lipstick_color: str = Form(None)
):
    try:
        # Process frame dengan live face tracker
        result_base64, message = live_face_tracker.process_live_frame(
            image_data, cheek_color, lipstick_color
        )
        
        if result_base64 is None:
            return JSONResponse(
                status_code=500,
                content={"error": message}
            )
        
        return {
            "success": True,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Error in process-live-frame endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)