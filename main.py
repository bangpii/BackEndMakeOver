from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import logging
import os
import uuid
from datetime import datetime
from skin_tracker import skin_tracker

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
UPLOADS_DIR = "user_uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

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
    """Save uploaded file to user_uploads directory"""
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_contents)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

def image_to_base64(image):
    """Convert OpenCV image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
        
        # Store user session
        user_sessions[session_id] = {
            "original_path": original_path,
            "original_image_base64": image_to_base64(image),
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
            "processed_image": f"data:image/jpeg;base64,{image_to_base64(image)}"
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
        
        return {
            "success": True,
            "message": "Reset to original photo",
            "processed_image": session_data["original_image_base64"]
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
                content={"error": message}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)