from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
import base64
import logging

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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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

def analyze_skin_tone(image_array):
    """Analyze skin tone from face detection"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_image)
        
        if not results.detections:
            return None, "No face detected"
        
        # Get the first face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = image_array.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)
        
        # Expand the face region slightly for better sampling
        padding = int(box_h * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        box_w = min(w - x, box_w + 2 * padding)
        box_h = min(h - y, box_h + 2 * padding)
        
        # Define face region for skin sampling
        face_roi = image_array[y:y+box_h, x:x+box_w]
        
        if face_roi.size == 0:
            return None, "Invalid face region"
        
        # Convert to RGB for skin tone analysis
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Sample multiple points on cheeks and forehead
        samples = []
        height, width, _ = face_rgb.shape
        
        # Cheek areas (avoid edges)
        cheek_left = face_rgb[height//3:2*height//3, width//8:width//4]
        cheek_right = face_rgb[height//3:2*height//3, 3*width//4:7*width//8]
        
        # Forehead area
        forehead = face_rgb[height//8:height//4, width//3:2*width//3]
        
        # Sample colors from different regions
        for region in [cheek_left, cheek_right, forehead]:
            if region.size > 0:
                # Take center pixel and surrounding area
                region_h, region_w, _ = region.shape
                center_y, center_x = region_h//2, region_w//2
                
                # Sample 3x3 area around center
                y_start = max(0, center_y-1)
                y_end = min(region_h, center_y+2)
                x_start = max(0, center_x-1)
                x_end = min(region_w, center_x+2)
                
                sample_area = region[y_start:y_end, x_start:x_end]
                if sample_area.size > 0:
                    # Flatten and take average
                    samples.extend(sample_area.reshape(-1, 3))
        
        if not samples:
            return None, "No skin samples found"
        
        # Convert to numpy array and calculate average skin tone
        samples_array = np.array(samples)
        avg_skin_tone = np.mean(samples_array, axis=0).astype(int)
        
        return avg_skin_tone.tolist(), "Skin tone analyzed successfully"
        
    except Exception as e:
        logger.error(f"Error in skin tone analysis: {str(e)}")
        return None, f"Analysis error: {str(e)}"

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
    
    # Get top 6 matches
    top_matches = best_matches[:6]
    
    # Group by category and get the best from each
    category_matches = {}
    for match in top_matches:
        category = match["category"]
        if category not in category_matches:
            category_matches[category] = match["shade"]
        elif len(category_matches[category]) < 2:  # Max 2 per category
            if isinstance(category_matches[category], dict):
                category_matches[category] = [category_matches[category], match["shade"]]
            else:
                category_matches[category].append(match["shade"])
    
    # Flatten the matches
    recommended_matches = []
    for category_matches_list in category_matches.values():
        if isinstance(category_matches_list, list):
            recommended_matches.extend(category_matches_list)
        else:
            recommended_matches.append(category_matches_list)
    
    return {
        "primary_category": top_matches[0]["category"],
        "recommended_matches": recommended_matches[:3]  # Return top 3
    }

def apply_foundation_to_face(image_array, foundation_hex):
    """Apply foundation color to detected face area"""
    try:
        # Convert hex to BGR
        foundation_hex = foundation_hex.lstrip('#')
        foundation_rgb = tuple(int(foundation_hex[i:i+2], 16) for i in (0, 2, 4))
        foundation_bgr = (foundation_rgb[2], foundation_rgb[1], foundation_rgb[0])
        
        # Convert to RGB for face detection
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if not results.detections:
            return image_array, "No face detected for foundation application"
        
        # Create a copy of the original image
        result_image = image_array.copy()
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image_array.shape
            
            # Calculate face coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # Expand the face region slightly
            padding = int(box_h * 0.15)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(w, x + box_w + padding)
            y_end = min(h, y + box_h + padding)
            
            # Create a smooth mask for the face area
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
            
            # Apply Gaussian blur to mask edges for smooth transition
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            # Create foundation layer
            foundation_layer = np.zeros_like(image_array)
            foundation_layer[:] = foundation_bgr
            
            # Blend foundation with original image using the mask
            mask_float = mask.astype(float) / 255.0
            mask_float = np.stack([mask_float] * 3, axis=-1)
            
            # Adjust blend strength (0.7 = 70% foundation, 30% original)
            blend_strength = 0.7
            blended = (foundation_layer * blend_strength + image_array * (1 - blend_strength))
            
            # Apply the blended result only to the masked area
            result_image = (blended * mask_float + image_array * (1 - mask_float)).astype(np.uint8)
        
        return result_image, "Foundation applied successfully"
        
    except Exception as e:
        logger.error(f"Error applying foundation: {str(e)}")
        return image_array, f"Foundation application error: {str(e)}"

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI backend!"}

@app.post("/api/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze skin tone
        skin_tone, message = analyze_skin_tone(image)
        
        if skin_tone is None:
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
        
        # Find matching foundation
        foundation_matches = find_best_matching_foundation(skin_tone)
        
        # Convert original image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
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

@app.post("/api/apply-foundation")
async def apply_foundation(file: UploadFile = File(...), foundation_hex: str = "#F9E6E6"):
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply foundation to face
        result_image, message = apply_foundation_to_face(image, foundation_hex)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "message": message,
            "processed_image": f"data:image/jpeg;base64,{image_base64}",
            "applied_foundation": foundation_hex
        }
        
    except Exception as e:
        logger.error(f"Error in apply-foundation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)