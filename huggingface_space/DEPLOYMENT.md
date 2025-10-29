# 🚀 Deployment Guide for Hugging Face Spaces

## Directory Structure

```
huggingface_space/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── README.md                 # Space description (displays on HF)
├── .gitignore               # Git ignore rules
├── model/
│   ├── vectorizer.pkl       # TF-IDF vectorizer
│   ├── classifier.pkl       # Logistic Regression model
│   └── config.json          # Model configuration
└── DEPLOYMENT.md            # This file
```

## Quick Start - Test Locally

```bash
cd huggingface_space
python app.py
```

Visit http://localhost:7860 to test the interface.

## Deploy to Hugging Face Spaces

### Method 1: Web UI (Easiest)

1. **Create a new Space:**
   - Go to https://huggingface.co/new-space
   - Name: `hdr-proposal-verifier`
   - SDK: **Gradio**
   - Make it **Public**

2. **Upload files via web interface:**
   - Click "Files" tab
   - Upload all files maintaining the directory structure
   - Ensure `model/` directory and all pickle files are uploaded

3. **Space will auto-build and deploy!**
   - Check the "Logs" tab if there are issues
   - Once built, your Space is live at: `https://huggingface.co/spaces/Subramanyam6/hdr-proposal-verifier`

### Method 2: Git CLI (Recommended for updates)

1. **Clone your Space repository:**
   ```bash
   git clone https://huggingface.co/spaces/Subramanyam6/hdr-proposal-verifier
   cd hdr-proposal-verifier
   ```

2. **Copy all files:**
   ```bash
   cp -r ../huggingface_space/* .
   ```

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "Initial deployment: TF-IDF model"
   git push
   ```

4. **Space will auto-rebuild!**

### Method 3: GitHub Sync (Best for continuous updates)

1. **Push `huggingface_space/` to GitHub:**
   ```bash
   cd /path/to/HDR_AI_Proposal_Verification_Assistant
   git add huggingface_space/
   git commit -m "Add HF Space"
   git push origin main
   ```

2. **On Hugging Face Space settings:**
   - Go to your Space settings
   - Enable "GitHub Sync"
   - Connect to: `Subramanyam6/HDR_AI_Proposal_Verification_Assistant`
   - Set directory: `huggingface_space/`

3. **Future updates:** Just push to GitHub, Space auto-syncs!

## Authentication

For git push to Hugging Face:
- **Username:** Subramanyam6
- **Password:** Use your Hugging Face **Access Token** (not password)
  - Create token at: https://huggingface.co/settings/tokens
  - Scope: **write**

## Troubleshooting

### Space won't build
- Check "Logs" tab on HF Space
- Verify all files uploaded correctly
- Ensure `model/` directory exists with all 3 files

### Model errors
- Verify pickle files aren't corrupted
- Check scikit-learn version matches (1.4.2)
- Re-export models if needed: `python ../synthetic_proposals/scripts/10_export_model.py`

### UI not displaying correctly
- Clear browser cache
- Check CSS in app.py
- Verify Gradio version (4.44.0)

## Testing Checklist

- [ ] Local test works (`python app.py`)
- [ ] Model loads without errors
- [ ] PDF upload works
- [ ] Predictions return correctly
- [ ] UI displays properly
- [ ] All dependencies in requirements.txt

## Next Steps

1. **Test with real PDFs:**
   - Upload sample HDR proposals
   - Verify predictions make sense
   - Collect user feedback

2. **Monitor performance:**
   - Check HF Space usage metrics
   - Monitor for errors in logs

3. **Iterate:**
   - Add BERT/DistilBERT model option
   - Improve UI based on feedback
   - Add more verification rules

## Support

Issues? Contact: subramanyam.duggirala@gmail.com  
GitHub: https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant

