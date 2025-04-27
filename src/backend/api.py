import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure the backend modules are in the path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.backend.analysis.audio_analysis import AudioDeepfakeAnalyzer
from src.backend.analysis.deep_image_analyzer import DeepImageAnalyzer
from src.backend.analysis.video_analysis import VideoAnalyzer
from src.backend.media_analyzer import MediaAnalyzerService

app = FastAPI(
    title="Vikasya Deepfake Detection API",
    description="API for detecting deepfake in images, videos, and audio files",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize media analyzer service
analyzer_service = MediaAnalyzerService()

# Mapping of file extensions to media types
MEDIA_TYPE_MAPPING = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".avi": "video",
    ".mov": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
}


@app.get("/")
async def root():
    return {
        "message": "Welcome to Vikasya Deepfake Detection API. Visit /docs for API documentation."
    }


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...), force_type: Optional[str] = None):
    """
    Analyze media file for deepfake detection.

    Parameters:
    - file: The media file to analyze
    - force_type: Optional parameter to force analysis as a specific media type (image, video, or audio)

    Returns:
    - Analysis results
    """
    # Determine media type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if force_type:
        if force_type not in ["image", "video", "audio"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid media type. Must be image, video, or audio.",
            )
        media_type = force_type
    else:
        media_type = MEDIA_TYPE_MAPPING.get(file_ext)
        if not media_type:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_ext}"
            )

    # Create temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        # Get the path of the temporary file
        temp_file_path = temp_file.name
        # Copy content of uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)

    try:
        # Analyze based on media type
        if media_type == "audio":
            analyzer = AudioDeepfakeAnalyzer()
            result = analyzer.analyze(temp_file_path)
        elif media_type == "image":
            analyzer = DeepImageAnalyzer()
            result = analyzer.analyze(temp_file_path)
        elif media_type == "video":
            analyzer = VideoAnalyzer()
            result = analyzer.analyze(temp_file_path)

        # Add media type to the result
        result["media_type"] = media_type

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@app.get("/status")
async def check_status():
    """Check if the API is running and services are available"""
    models_status = {}

    # Check audio model
    try:
        audio_analyzer = AudioDeepfakeAnalyzer()
        models_status["audio_model"] = os.path.exists(audio_analyzer.tflite_model_path)
    except:
        models_status["audio_model"] = False

    # Add checks for other models as needed

    return {"status": "online", "models": models_status}


# API Frontend class for integration with main.py
class ApiFrontend:
    def __init__(self, analyzer_service):
        self.analyzer_service = analyzer_service

    def run(self, host="0.0.0.0", port=8000):
        """Run the FastAPI application"""
        uvicorn.run("backend.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
