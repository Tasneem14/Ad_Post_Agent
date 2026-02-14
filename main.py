from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import base64
import os
import uvicorn
import shutil
from pathlib import Path

# Import the LangGraph workflow from the existing script
try:
    from content_orchestrationfal import app as workflow_app
except ImportError as e:
    print(f"Error importing content_orchestrationfal: {e}")
    # Fallback for testing without the actual script if needed, 
    # but in production this should fail or be handled.
    workflow_app = None

app = FastAPI(title="Content Orchestration Agent")

# CORS (allow all for simplicity in this demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

def file_to_data_uri(file: UploadFile) -> str:
    """Helper to convert uploaded file to data URI for Fal.ai"""
    if not file:
        return None
    try:
        contents = file.file.read()
        encoded = base64.b64encode(contents).decode("utf-8")
        media_type = file.content_type or "application/octet-stream"
        return f"data:{media_type};base64,{encoded}"
    except Exception as e:
        print(f"Error processing file {file.filename}: {e}")
        return None

@app.post("/generate")
async def generate_content(
    platform: str = Form(...),
    intent: str = Form(...),
    content_idea: str = Form(...),
    description: str = Form(...),
    reference_text: Optional[str] = Form(None),
    user_media_choice: Optional[str] = Form(None),
    reference_image: Optional[UploadFile] = File(None),
    video_init_image: Optional[UploadFile] = File(None)
):
    if not workflow_app:
        raise HTTPException(status_code=500, detail="Workflow app not initialized. Check imports.")

    # Prepare inputs for the agent
    user_inputs = {
        "content_idea": content_idea,
        "description": description,
        "reference_text": reference_text or ""
    }

    # Process files
    uploaded_files = {
        "reference_image": None,
        "video_init_image": None
    }
    
    if reference_image:
        uploaded_files["reference_image"] = file_to_data_uri(reference_image)
    
    if video_init_image:
        uploaded_files["video_init_image"] = file_to_data_uri(video_init_image)

    # Initial state for LangGraph
    initial_state = {
        "platform": platform,
        "intent": intent,
        "user_inputs": user_inputs,
        "uploaded_files": uploaded_files,
        "user_media_choice": user_media_choice,
        "errors": []
    }

    try:
        # Invoke the LangGraph workflow
        final_state = workflow_app.invoke(initial_state)
        
        return {
            "status": "success",
            "generated_text": final_state.get("generated_text"),
            "generated_media_url": final_state.get("generated_media_url"),
            "errors": final_state.get("errors", [])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
