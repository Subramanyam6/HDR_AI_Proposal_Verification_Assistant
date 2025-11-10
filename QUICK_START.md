# Quick Start Guide

## Testing Docker Build Locally

```bash
# Build the image
docker build -t hdr-proposal-verification .

# Run locally (port 8000)
docker run -p 8000:7860 \
  -e OPENAI_API_KEY=your-key-here \
  -e ENVIRONMENT=production \
  hdr-proposal-verification

# Visit http://localhost:8000
```

## HuggingFace Space Setup (Quick Steps)

1. **Create Space on HuggingFace**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `HDR_AI_Proposal_Verification_Assistant`
   - SDK: **Docker**
   - Hardware: CPU basic (or upgrade)
   - License: Apache 2.0

2. **Clone Your Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/HDR_AI_Proposal_Verification_Assistant
   cd HDR_AI_Proposal_Verification_Assistant
   ```

3. **Copy Files to HF Space**
   ```bash
   # From your main project directory
   cp Dockerfile /path/to/hf-space/
   cp huggingface_space/app.py /path/to/hf-space/
   cp huggingface_space/README.md /path/to/hf-space/
   cp -r backend /path/to/hf-space/
   cp -r frontend /path/to/hf-space/
   ```

4. **Add OpenAI API Key Secret**
   - Go to Space Settings → Variables and secrets
   - Add secret: `OPENAI_API_KEY` = your actual key

5. **Commit and Push**
   ```bash
   cd /path/to/hf-space
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. **Wait for Build** (5-10 minutes)
   - Check Logs tab in your Space
   - Once built, app is live!

## Important Notes

- **Port**: HuggingFace uses port 7860 (already configured)
- **Models**: Must be in `backend/app/models/` directory
- **Frontend**: Will be built automatically in Docker
- **API Key**: Set as HuggingFace Secret, not in code

## File Checklist for HF Space

- [x] `Dockerfile` (root)
- [x] `app.py` (root - HF entry point)
- [x] `README.md` (root - Space description)
- [x] `backend/` (entire directory)
- [x] `frontend/` (entire directory)
- [x] `backend/app/models/` (all model files)
- [x] `backend/app/samples/` (sample text files)

## Troubleshooting

**Build fails?**
- Check Logs tab for errors
- Verify all model files are present
- Ensure Dockerfile syntax is correct

**Models not loading?**
- Verify `backend/app/models/` exists
- Check file permissions
- Review startup logs

**Frontend not showing?**
- Check if `frontend/dist/` was built
- Verify FastAPI is serving static files
- Check browser console for errors
