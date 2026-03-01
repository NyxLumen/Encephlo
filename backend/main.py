from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
# import numpy as np
# from core.fusion_engine import extract_features, run_svm  <-- Sid uncomments these later

app = FastAPI(title="Encephlo 3.0 Core API")

# 1. THE SHIELD: CORS Middleware (Allows your React frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to localhost:5173 in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE CONTRACT: Strict Output Schema
# If Sid's code doesn't return EXACTLY this format, the API throws an error.
# This makes it impossible for him to silently break your frontend.
class DiagnosticResponse(BaseModel):
    status: str
    diagnosis: str
    confidence: float
    inference_time_ms: float
    heatmap_url: str | None = None  # We will send the heatmap as a base64 string or static file url later

# --- 🛑 SID'S WORKSPACE STARTS HERE 🛑 ---

def process_neural_fusion(image_bytes: bytes) -> dict:
    """
    SID: Do NOT touch the API routes. Only write code inside this function.
    1. Decode the image_bytes into a cv2/numpy array.
    2. Run your Otsu crop.
    3. Extract the DenseNet and EfficientNet vectors.
    4. Concat and pass to SVM.
    5. Return a dictionary exactly matching the keys below.
    """
    import time
    start = time.time()
    
    # TODO: Replace this dummy logic with the real Feature Fusion Engine
    time.sleep(1.5) # Simulating network/GPU think time
    fake_diagnosis = "Glioma"
    fake_confidence = 94.2
    
    end = time.time()
    
    return {
        "diagnosis": fake_diagnosis,
        "confidence": fake_confidence,
        "inference_time_ms": round((end - start) * 1000, 2)
    }

# --- 🛑 SID'S WORKSPACE ENDS HERE 🛑 ---


# 3. THE ENDPOINT: What your React app will actually hit
@app.post("/api/v1/analyze", response_model=DiagnosticResponse)
async def analyze_scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read the file from the React frontend
        image_bytes = await file.read()
        
        # Pass to Sid's isolated logic box
        results = process_neural_fusion(image_bytes)
        
        # Enforce the contract
        return DiagnosticResponse(
            status="success",
            diagnosis=results["diagnosis"],
            confidence=results["confidence"],
            inference_time_ms=results["inference_time_ms"],
            heatmap_url=None # We will wire up the ScoreCAM 3D texture here later
        )
        
    except Exception as e:
        # If Sid's code crashes, it gets caught here and returns a clean error to React
        raise HTTPException(status_code=500, detail=f"Neural Engine Failure: {str(e)}")

if __name__ == "__main__":
    # Sid runs this file by typing: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)