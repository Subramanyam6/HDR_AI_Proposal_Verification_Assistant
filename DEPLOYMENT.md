# Deployment Guide

This guide covers deploying the HDR AI Proposal Verification Assistant to GitHub and HuggingFace Spaces.

## Prerequisites

- Docker installed and running
- GitHub account
- HuggingFace account
- OpenAI API key (for AI suggestions)

## Step 1: Prepare for GitHub

### 1.1 Update .gitignore

Ensure `.gitignore` excludes sensitive files but includes necessary model files:

```bash
# Models should be included (they're needed for deployment)
# But exclude large training datasets
```

### 1.2 Test Docker Build Locally

```bash
# Build the Docker image
docker build -t hdr-proposal-verification .

# Test locally
docker run -p 8000:7860 \
  -e OPENAI_API_KEY=your-key-here \
  -e ENVIRONMENT=production \
  hdr-proposal-verification

# Visit http://localhost:8000
```

### 1.3 Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files (models will be included)
git add .

# Commit
git commit -m "Initial commit: HDR Proposal Verification Assistant"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant.git

# Push to GitHub
git push -u origin main
```

## Step 2: Create HuggingFace Space

### 2.1 Create New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `HDR_AI_Proposal_Verification_Assistant` (or your preferred name)
   - **SDK**: Select **"Docker"**
   - **Visibility**: Public or Private (your choice)
   - **Hardware**: Select **"CPU basic"** (or upgrade if needed for faster inference)
   - **License**: Apache 2.0

### 2.2 Clone the Space Repository

```bash
# Clone your new HuggingFace Space
git clone https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Verification_Assistant_V2
cd HDR_AI_Proposal_Verification_Assistant
```

### 2.3 Copy Required Files to HF Space

From your main project directory:

```bash
# Copy Dockerfile
cp Dockerfile /path/to/hf-space/

# Copy app.py (HuggingFace entry point)
cp huggingface_space/app.py /path/to/hf-space/

# Copy README.md
cp huggingface_space/README.md /path/to/hf-space/

# Copy backend code
cp -r backend /path/to/hf-space/

# Copy frontend code
cp -r frontend /path/to/hf-space/

# Copy model files (IMPORTANT!)
cp -r backend/app/models /path/to/hf-space/backend/app/

# Copy sample files
cp -r backend/app/samples /path/to/hf-space/backend/app/

# Copy requirements.txt
cp backend/requirements.txt /path/to/hf-space/backend/
```

### 2.4 Create .env File (Optional)

Create `.env` file in HF Space root for secrets:

```bash
# In HF Space directory
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Note**: For production, use HuggingFace Secrets instead:
1. Go to your Space settings
2. Navigate to "Variables and secrets"
3. Add `OPENAI_API_KEY` as a secret

### 2.5 Update Backend Config for HF Spaces

The backend should automatically detect the environment. Ensure `backend/app/config.py` handles production correctly.

### 2.6 Commit and Push to HuggingFace

```bash
cd /path/to/hf-space

# Add all files
git add .

# Commit
git commit -m "Initial deployment: HDR Proposal Verification Assistant"

# Push to HuggingFace
git push
```

### 2.7 Monitor Deployment

1. Go to your Space page on HuggingFace
2. Click on the "Logs" tab to see build progress
3. Wait for the build to complete (usually 5-10 minutes)
4. Once built, your app will be live at: `https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Verification_Assistant_V2`

## Step 3: Environment Variables (HuggingFace Secrets)

### Required Secrets:

1. **OPENAI_API_KEY**: Your OpenAI API key for AI suggestions
   - Go to Space Settings → Variables and secrets
   - Add new secret: `OPENAI_API_KEY`
   - Value: Your actual API key

### Optional Environment Variables:

- `ENVIRONMENT`: Set to `production` (default)
- `CORS_ORIGINS`: Set to `*` for public access

## Step 4: Verify Deployment

### 4.1 Test the Application

1. Visit your HuggingFace Space URL
2. Upload a sample PDF or paste text
3. Click "Run Verification"
4. Verify all checks work correctly
5. Test AI suggestions (requires OpenAI API key)

### 4.2 Check Logs

If something doesn't work:
1. Go to Space → Logs tab
2. Check for errors during startup
3. Verify models are loading correctly
4. Check API endpoints are accessible

## Troubleshooting

### Models Not Loading

- Ensure `backend/app/models/` directory exists with all model files
- Check file permissions
- Verify model paths in `backend/app/config.py`

### Frontend Not Loading

- Verify `frontend/dist/` was built correctly
- Check Dockerfile copied frontend build correctly
- Verify FastAPI is serving static files from correct path

### API Errors

- Check CORS settings in `backend/app/main.py`
- Verify port is set to 7860 (HuggingFace requirement)
- Check environment variables are set correctly

### OpenAI API Errors

- Verify `OPENAI_API_KEY` secret is set in HuggingFace
- Check API key is valid and has credits
- Review error messages in logs

## File Structure for HF Space

```
HDR_AI_Proposal_Verification_Assistant/
├── Dockerfile              # Main Dockerfile
├── app.py                  # HuggingFace entry point
├── README.md               # Space description
├── backend/
│   ├── app/
│   │   ├── models/        # ML model files (IMPORTANT!)
│   │   ├── samples/       # Sample proposal files
│   │   └── ...
│   └── requirements.txt
└── frontend/
    └── ...                # Frontend source (will be built in Docker)
```

## Notes

- **Model Files**: Make sure all model files (`*.pkl`, `*.safetensors`, etc.) are included in the repository
- **Port**: HuggingFace Spaces uses port 7860 by default
- **Build Time**: First build may take 10-15 minutes due to model downloads
- **Memory**: Ensure your HF Space has enough memory for DistilBERT model (~2GB)

## Updating the Space

To update your Space:

```bash
cd /path/to/hf-space

# Pull latest changes from main repo (if syncing)
# Or make changes directly

git add .
git commit -m "Update: description of changes"
git push
```

HuggingFace will automatically rebuild your Space.

## Support

For issues or questions:
- GitHub Issues: [Your repo URL]/issues
- Email: subramanyam.duggirala@outlook.com
