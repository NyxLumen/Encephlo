from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import os
import uuid
import shutil

# Import the actual Fusion Engine we just built
# (Note: If you put fusion_engine.py inside a 'core' folder, change this to: from core.fusion_engine import FusionEngine)
from fusion_engine import FusionEngine 

app = FastAPI(title="Encephlo 3.0 Core API")

# 1. THE SHIELD: CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. THE CONTRACT: Strict Output Schema
class DiagnosticResponse(BaseModel):
    status: str
    diagnosis: str
    confidence: float
    inference_time_ms: float
    heatmap_url: str | None = None  

# 3. INITIALIZE THE BRAIN
print("🧠 Booting Neural Engine...")
# This loads ViT, DenseNet, and the SVM into RAM/VRAM
engine = FusionEngine(models_dir="models")
print("✅ Engine Online.")

# Setup a temporary folder to hold the image while the models look at it
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 4. THE ENDPOINT
@app.post("/api/v1/analyze", response_model=DiagnosticResponse)
async def analyze_scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Create a safe, unique filename
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        start_time = time.time()
        
        # Catch all 3 outputs from the engine
        diagnosis, confidence, heatmap_b64 = engine.predict(file_path)
        
        end_time = time.time()
        inference_time = round((end_time - start_time) * 1000, 2)
        
        return DiagnosticResponse(
            status="success",
            diagnosis=diagnosis,
            confidence=round(confidence * 100, 2), 
            inference_time_ms=inference_time,
            heatmap_url=heatmap_b64 # <-- Passing the Base64 image directly!
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural Engine Failure: {str(e)}")
        
    finally:
        # Cleanup: Delete the image so your hard drive doesn't fill up
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)