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
from skin_tracker import skin_tracker

# Import MediaPipe untuk LiveFaceTracker
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("MediaPipe successfully imported")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logger.error(f"MediaPipe import failed: {str(e)}")

app = FastAPI(title="MakeOver Backend")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ========== OPTIMIZED LIVE FACE TRACKER ==========

class OptimizedLiveFaceTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available - LiveFaceTracker disabled")
            return
            
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

    def apply_fast_effects(self, image, cheek_color=None, lipstick_color=None):
        """Fast effects application untuk real-time"""
        try:
            result = image.copy()
            h, w = image.shape[:2]
            
            if cheek_color:
                cheek_hex = cheek_color.lstrip('#')
                cheek_rgb = tuple(int(cheek_hex[i:i+2], 16) for i in (0, 2, 4))
                cheek_bgr = (cheek_rgb[2], cheek_rgb[1], cheek_rgb[0])
                
                # Optimized cheek overlay dengan bentuk oval
                cheek_overlay = np.zeros_like(image, dtype=np.uint8)
                
                # Cheek positions
                left_cheek_center = (int(w*0.25), int(h*0.5))
                right_cheek_center = (int(w*0.75), int(h*0.5))
                cheek_radius_x = int(w * 0.15)
                cheek_radius_y = int(h * 0.1)
                
                # Draw ellipses untuk cheeks
                cv2.ellipse(cheek_overlay, left_cheek_center, (cheek_radius_x, cheek_radius_y), 
                           0, 0, 360, cheek_bgr, -1)
                cv2.ellipse(cheek_overlay, right_cheek_center, (cheek_radius_x, cheek_radius_y), 
                           0, 0, 360, cheek_bgr, -1)
                
                # Create smooth mask
                cheek_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(cheek_mask, left_cheek_center, (cheek_radius_x, cheek_radius_y), 
                           0, 0, 360, 255, -1)
                cv2.ellipse(cheek_mask, right_cheek_center, (cheek_radius_x, cheek_radius_y), 
                           0, 0, 360, 255, -1)
                
                # Blur mask untuk smooth edges
                cheek_mask = cv2.GaussianBlur(cheek_mask, (51, 51), 0)
                cheek_mask_float = cheek_mask.astype(float) / 255.0
                cheek_mask_float = np.stack([cheek_mask_float] * 3, axis=-1)
                
                # Blend dengan opacity rendah
                result = result.astype(float) * (1 - cheek_mask_float * 0.3) + \
                        cheek_overlay.astype(float) * cheek_mask_float * 0.3
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            if lipstick_color:
                lip_hex = lipstick_color.lstrip('#')
                lip_rgb = tuple(int(lip_hex[i:i+2], 16) for i in (0, 2, 4))
                lip_bgr = (lip_rgb[2], lip_rgb[1], lip_rgb[0])
                
                # Optimized lip overlay
                lip_overlay = np.zeros_like(image, dtype=np.uint8)
                lip_center = (w//2, int(h*0.65))
                lip_size = (int(w*0.2), int(h*0.08))
                
                cv2.ellipse(lip_overlay, lip_center, lip_size, 0, 0, 360, lip_bgr, -1)
                
                # Create lip mask
                lip_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(lip_mask, lip_center, lip_size, 0, 0, 360, 255, -1)
                
                # Blur mask
                lip_mask = cv2.GaussianBlur(lip_mask, (31, 31), 0)
                lip_mask_float = lip_mask.astype(float) / 255.0
                lip_mask_float = np.stack([lip_mask_float] * 3, axis=-1)
                
                # Blend dengan opacity lebih tinggi
                result = result.astype(float) * (1 - lip_mask_float * 0.6) + \
                        lip_overlay.astype(float) * lip_mask_float * 0.6
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result, "Effects applied successfully"
            
        except Exception as e:
            logger.error(f"Error applying fast effects: {str(e)}")
            return image, f"Error: {str(e)}"

# Initialize optimized tracker
optimized_tracker = OptimizedLiveFaceTracker()

# ========== FUNGSI UTILITAS ==========

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
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Sedikit lebih tinggi untuk kualitas
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
    
    for category, shades in SKIN_TONE_CATEGORIES.items():
        for shade in shades:
            shade_rgb = np.array(shade["rgb"])
            distance = np.linalg.norm(skin_tone_array - shade_rgb)
            best_matches.append({
                "category": category,
                "shade": shade,
                "distance": distance
            })
    
    best_matches.sort(key=lambda x: x["distance"])
    top_matches = best_matches[:6]
    
    category_matches = {}
    for match in top_matches:
        category = match["category"]
        if category not in category_matches:
            category_matches[category] = []
        category_matches[category].append(match["shade"])
    
    recommended_matches = []
    for category, matches in category_matches.items():
        recommended_matches.extend(matches[:2])
    
    return {
        "primary_category": top_matches[0]["category"],
        "recommended_matches": recommended_matches[:4]
    }

# ========== ENDPOINTS ==========

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "service": "MakeOver Backend",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        unique_filename = f"{session_id}_original{file_extension}"
        
        original_path = save_uploaded_file(contents, unique_filename)
        
        if not original_path:
            raise HTTPException(status_code=500, detail="Failed to save file")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        original_base64 = image_to_base64(image)
        if not original_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        user_sessions[session_id] = {
            "original_path": original_path,
            "original_image_base64": original_base64,
            "original_image": image.copy(),
            "upload_time": datetime.now().isoformat()
        }
        
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

# ========== OPTIMIZED LIVE PROCESSING ENDPOINT ==========

@app.post("/api/process-live-frame")
async def process_live_frame(data: dict):
    """Optimized endpoint untuk processing frame live camera"""
    try:
        if not data or 'image' not in data:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No image data provided"}
            )
        
        frame_data = data.get('image')
        cheek_color = data.get('cheek_color')
        lipstick_color = data.get('lipstick_color')
        
        # Validasi: jika tidak ada warna yang dipilih, return frame asli
        if not cheek_color and not lipstick_color:
            return {
                "success": True,
                "processed_image": frame_data,  # Return frame asli
                "message": "No colors selected",
                "effects_applied": False
            }
        
        # Validate colors
        if cheek_color:
            cheek_color = cheek_color.lstrip('#')
            if len(cheek_color) != 6:
                cheek_color = None
        
        if lipstick_color:
            lipstick_color = lipstick_color.lstrip('#')
            if len(lipstick_color) != 6:
                lipstick_color = None
        
        # Decode base64 image
        try:
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Invalid image data"}
                )
                
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid image data: {str(e)}"}
            )
        
        # Resize untuk performance (lebih kecil untuk performa lebih baik)
        h, w = image.shape[:2]
        if w > 480:  # Lebih kecil untuk performa real-time
            scale = 480 / w
            new_w = 480
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply effects dengan optimized tracker
        try:
            result_image, message = optimized_tracker.apply_fast_effects(
                image, 
                f"#{cheek_color}" if cheek_color else None,
                f"#{lipstick_color}" if lipstick_color else None
            )
        except Exception as e:
            logger.error(f"Error applying effects: {str(e)}")
            # Fallback: return original frame jika error
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "processed_image": f"data:image/jpeg;base64,{result_base64}",
                "message": f"Error applying effects: {str(e)}",
                "effects_applied": False
            }
        
        # Encode result
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', result_image, encode_param)
            result_base64 = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encoding result: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Failed to encode image: {str(e)}"}
            )
        
        return {
            "success": True,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "message": message,
            "effects_applied": True,
            "debug_info": {
                "original_size": f"{w}x{h}",
                "processed_size": f"{result_image.shape[1]}x{result_image.shape[0]}",
                "cheek_color": "applied" if cheek_color else "none",
                "lipstick_color": "applied" if lipstick_color else "none"
            }
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in process-live-frame: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Internal server error: {str(e)}"}
        )

# Endpoint lainnya tetap sama...
@app.post("/api/apply-foundation")
async def apply_foundation(
    file: UploadFile = File(...),
    foundation_hex: str = Form("#F9E6E6"),
    session_id: str = Form(None)
):
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
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

@app.get("/api/sessions")
async def get_sessions():
    """Debug endpoint to check active sessions"""
    return {
        "active_sessions": len(user_sessions),
        "sessions": list(user_sessions.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)