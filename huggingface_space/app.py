"""
HuggingFace Space entry point for HDR AI Proposal Verification Assistant.
This file is used by HuggingFace Spaces to run the application.
"""
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # HuggingFace Spaces expects the app to run on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
