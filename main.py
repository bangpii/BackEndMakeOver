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

# Import LiveFaceTracker dari file terpisah
try:
    from face_live_tracker import LiveFaceTracker
    live_face_tracker = LiveFaceTracker()
    MEDIAPIPE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LiveFaceTracker successfully imported")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logger.error(f"❌ LiveFaceTracker import failed: {str(e)}")
    live_face_tracker = None
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    logger.error(f"❌ Error initializing LiveFaceTracker: {str(e)}")
    live_face_tracker = None

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

# ========== FUNGSI UTILITAS ==========

def save_uploaded_file(file_contents, filename):
    """Save uploaded file to uploads directory"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_contents)
        return file_path
    except Exception as e:
        logger.error(f"❌ Error saving file: {str(e)}")
        return None

def image_to_base64(image):
    """Convert OpenCV image to base64"""
    try:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"❌ Error converting image to base64: {str(e)}")
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

# ========== SIMPLE FALLBACK UNTUK LIVE PROCESSING ==========

def apply_simple_color_overlay(image, cheek_color=None, lipstick_color=None):
    """Apply simple color overlay tanpa face detection"""
    try:
        result = image.copy()
        
        if cheek_color:
            # Convert hex to BGR
            cheek_hex = cheek_color.lstrip('#')
            cheek_rgb = tuple(int(cheek_hex[i:i+2], 16) for i in (0, 2, 4))
            cheek_bgr = (cheek_rgb[2], cheek_rgb[1], cheek_rgb[0])  # RGB to BGR
            
            # Create cheek overlay (simple rectangular areas)
            h, w = image.shape[:2]
            cheek_overlay = np.zeros_like(image)
            
            # Define cheek areas (simple rectangles)
            left_cheek = (int(w*0.1), int(h*0.4), int(w*0.4), int(h*0.7))
            right_cheek = (int(w*0.6), int(h*0.4), int(w*0.9), int(h*0.7))
            
            # Apply color to cheek areas
            cheek_overlay[left_cheek[1]:left_cheek[3], left_cheek[0]:left_cheek[2]] = cheek_bgr
            cheek_overlay[right_cheek[1]:right_cheek[3], right_cheek[0]:right_cheek[2]] = cheek_bgr
            
            # Blend with original image
            alpha = 0.3  # Transparency
            result = cv2.addWeighted(result, 1, cheek_overlay, alpha, 0)
        
        if lipstick_color:
            # Convert hex to BGR
            lip_hex = lipstick_color.lstrip('#')
            lip_rgb = tuple(int(lip_hex[i:i+2], 16) for i in (0, 2, 4))
            lip_bgr = (lip_rgb[2], lip_rgb[1], lip_rgb[0])  # RGB to BGR
            
            # Create lip overlay
            h, w = image.shape[:2]
            lip_overlay = np.zeros_like(image)
            
            # Define lip area (simple rectangle)
            lip_area = (int(w*0.3), int(h*0.6), int(w*0.7), int(h*0.75))
            
            # Apply color to lip area
            lip_overlay[lip_area[1]:lip_area[3], lip_area[0]:lip_area[2]] = lip_bgr
            
            # Blend with original image
            alpha = 0.5  # More opaque for lips
            result = cv2.addWeighted(result, 1, lip_overlay, alpha, 0)
        
        return result, "Color effects applied successfully"
        
    except Exception as e:
        logger.error(f"❌ Error in simple color overlay: {str(e)}")
        return image, f"Error applying color: {str(e)}"

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
        "live_tracker_available": live_face_tracker is not None,
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
        logger.error(f"❌ Error in upload-photo endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
        logger.error(f"❌ Error in apply-foundation endpoint: {str(e)}")
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
        logger.error(f"❌ Error in reset-to-original endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        skin_tone, message = skin_tracker.analyze_skin_tone_precise(image)
        
        if skin_tone is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": message}
            )
        
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
        logger.error(f"❌ Error in analyze-skin endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ========== ENDPOINT LIVE PROCESSING YANG DIPERBAIKI ==========

@app.post("/api/process-live-frame")
async def process_live_frame(data: dict):
    """Endpoint untuk processing frame live camera dengan efek real-time"""
    try:
        logger.info("📸 Received live frame processing request")
        
        if not data or 'image' not in data:
            logger.error("❌ No image data in request")
            raise HTTPException(status_code=400, detail="No image data provided")
        
        frame_data = data.get('image')
        cheek_color = data.get('cheek_color')
        lipstick_color = data.get('lipstick_color')
        
        logger.info(f"🎨 Processing frame - Cheek: {cheek_color}, Lipstick: {lipstick_color}")
        
        # Validate colors
        if cheek_color:
            cheek_color = cheek_color.lstrip('#')
            if len(cheek_color) != 6:
                cheek_color = None
                logger.warning("⚠️ Invalid cheek color format")
        
        if lipstick_color:
            lipstick_color = lipstick_color.lstrip('#')
            if len(lipstick_color) != 6:
                lipstick_color = None
                logger.warning("⚠️ Invalid lipstick color format")
        
        # Decode base64 image
        try:
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("❌ Failed to decode image")
                raise HTTPException(status_code=400, detail="Invalid image data")
                
            logger.info(f"✅ Image decoded successfully: {image.shape}")
            
        except Exception as e:
            logger.error(f"❌ Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Resize untuk performance
        h, w = image.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w = 800
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"🔄 Image resized to: {image.shape}")
        
        # Apply effects
        result_image = image.copy()
        messages = []
        
        if cheek_color or lipstick_color:
            try:
                # Gunakan LiveFaceTracker jika tersedia, jika tidak gunakan fallback
                if MEDIAPIPE_AVAILABLE and live_face_tracker is not None:
                    logger.info("🎯 Using LiveFaceTracker for precise face detection")
                    
                    # Apply cheek color jika ada
                    if cheek_color:
                        cheek_result, cheek_msg = live_face_tracker.apply_cheek_color(
                            result_image, f"#{cheek_color}" if cheek_color else None
                        )
                        if "applied" in cheek_msg.lower():
                            result_image = cheek_result
                            messages.append("✅ Cheek color applied")
                        else:
                            messages.append(f"❌ Cheek: {cheek_msg}")
                    
                    # Apply lipstick jika ada
                    if lipstick_color:
                        lip_result, lip_msg = live_face_tracker.apply_lipstick(
                            result_image, f"#{lipstick_color}" if lipstick_color else None
                        )
                        if "applied" in lip_msg.lower():
                            result_image = lip_result
                            messages.append("✅ Lipstick applied")
                        else:
                            messages.append(f"❌ Lipstick: {lip_msg}")
                else:
                    # Gunakan simple fallback
                    logger.info("🔄 Using simple color overlay (MediaPipe not available)")
                    result_image, message = apply_simple_color_overlay(
                        result_image, 
                        f"#{cheek_color}" if cheek_color else None,
                        f"#{lipstick_color}" if lipstick_color else None
                    )
                    messages.append(message)
                
                logger.info("✅ Color effects applied successfully")
            except Exception as e:
                logger.error(f"❌ Error applying color effects: {str(e)}")
                messages.append(f"❌ Error applying effects: {str(e)}")
        else:
            messages.append("ℹ️ No colors selected")
        
        # Encode result
        try:
            result_base64 = image_to_base64(result_image)
            if not result_base64:
                logger.error("❌ Failed to encode result image")
                raise HTTPException(status_code=500, detail="Failed to process image")
                
            logger.info("✅ Result image encoded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error encoding result: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")
        
        return {
            "success": True,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "message": " | ".join(messages),
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "live_tracker_available": live_face_tracker is not None,
            "debug_info": {
                "original_size": f"{w}x{h}",
                "processed_size": f"{result_image.shape[1]}x{result_image.shape[0]}",
                "cheek_color": cheek_color,
                "lipstick_color": lipstick_color
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in process-live-frame: {str(e)}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/apply-cheek-color")
async def apply_cheek_color(
    file: UploadFile = File(...),
    cheek_hex: str = Form(...),
    session_id: str = Form(None)
):
    """Endpoint khusus untuk apply cheek color"""
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        if not MEDIAPIPE_AVAILABLE or live_face_tracker is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "MediaPipe not available for precise face detection"
                }
            )
        
        result_image, message = live_face_tracker.apply_cheek_color(image, cheek_hex)
        
        result_base64 = image_to_base64(result_image)
        if not result_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        return {
            "success": True,
            "message": message,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "applied_cheek_color": cheek_hex
        }
        
    except Exception as e:
        logger.error(f"❌ Error in apply-cheek-color endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/apply-lipstick")
async def apply_lipstick(
    file: UploadFile = File(...),
    lipstick_hex: str = Form(...),
    session_id: str = Form(None)
):
    """Endpoint khusus untuk apply lipstick"""
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
            
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        if not MEDIAPIPE_AVAILABLE or live_face_tracker is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "MediaPipe not available for precise face detection"
                }
            )
        
        result_image, message = live_face_tracker.apply_lipstick(image, lipstick_hex)
        
        result_base64 = image_to_base64(result_image)
        if not result_base64:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        return {
            "success": True,
            "message": message,
            "processed_image": f"data:image/jpeg;base64,{result_base64}",
            "applied_lipstick": lipstick_hex
        }
        
    except Exception as e:
        logger.error(f"❌ Error in apply-lipstick endpoint: {str(e)}")
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
    logger.info("🚀 Starting MakeOver Backend Server...")
    logger.info(f"📊 MediaPipe Available: {MEDIAPIPE_AVAILABLE}")
    logger.info(f"🎯 Live Tracker Available: {live_face_tracker is not None}")
    uvicorn.run(app, host="0.0.0.0", port=5000)