import re
import shutil

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, FastAPI
from fastapi.responses import JSONResponse
import os
import uuid
from Test_Cheating_Detection.detection import FRAMES_DIR, process_candidate_exam
from Test_Cheating_Detection.generate_pics_from_videos import extract_frames
from fastapi.middleware.cors import CORSMiddleware

router = APIRouter()

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

@router.post("/check-cheating/")
async def upload_video(file: UploadFile = File(...), candidate_name: str = Form(...)):
    # Validate extension
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[-1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    # candidate_name is used in file paths, so drop anything that could escape the folder (e.g. "../")
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", candidate_name)
    save_path = os.path.join(VIDEO_DIR, safe_name + ext)
    candidate_directory = os.path.join(FRAMES_DIR, safe_name)
    with open(save_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Sample one frame every 2 seconds, then run YOLO + MediaPipe on each frame
        extract_frames(save_path, candidate_directory, 2)
        res = process_candidate_exam(safe_name)
    finally:
        shutil.rmtree(candidate_directory, ignore_errors=True)
        os.remove(save_path)
    return JSONResponse(content=res)

app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
