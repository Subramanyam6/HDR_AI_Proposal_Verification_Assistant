---
title: HDR Proposal Verification Assistant
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# HDR AI Proposal Verification Assistant

A production-ready ML-powered system for automated verification of HDR proposal compliance, built with React + FastAPI.

## Tech Stack

- **Frontend**: React 18 + Vite + TypeScript + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + PyTorch + Transformers + scikit-learn
- **ML Models**: DistilBERT, TF-IDF + Logistic Regression, Naive Bayes
- **Deployment**: Docker (single container) + GitHub Actions CI/CD

## Features

The system detects four types of compliance issues:

### ML-Based Detection
- **Crosswalk errors**: Wrong requirement IDs (R1-R5) cited in wrong sections
- **Banned phrases**: Prohibited language like "unconditional guarantee", "guaranteed savings"
- **Name inconsistency**: PM name in different formats (e.g., "John Smith" vs "John S.")

### Rule-Based Detection
- **Date inconsistency**: Compares anticipated submission date vs signed/sealed date

## Project Structure

```
HDR_AI_Proposal_Verification_Assistant/
├── frontend/                # React + Vite application
│   ├── src/
│   │   ├── components/      # UI components (shadcn/ui)
│   │   ├── store/           # Zustand state management
│   │   ├── App.tsx          # Main application
│   │   └── types.ts         # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── main.py          # FastAPI app entry
│   │   ├── config.py        # Settings
│   │   ├── routers/         # API endpoints
│   │   │   └── verify.py    # /api/verify, /api/samples
│   │   ├── services/        # Business logic
│   │   │   ├── pdf_extractor.py
│   │   │   ├── tfidf_model.py
│   │   │   ├── distilbert_model.py
│   │   │   ├── nb_model.py
│   │   │   ├── rule_engine.py
│   │   │   └── rag_service.py
│   │   ├── models/          # ML model files (download manually)
│   │   └── samples/         # Sample text files
│   └── requirements.txt
│
├── synthetic_proposals/     # Data generation & training (unchanged)
│   ├── config/              # Configuration (knobs.yaml)
│   ├── scripts/             # Pipeline and ML training scripts
│   └── dataset/             # Generated data (gitignored)
│
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # Local development
├── .env.example             # Environment template
└── README.md
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/Subramanyam6/HDR_AI_Proposal_Verification_Assistant.git
cd HDR_AI_Proposal_Verification_Assistant

# 2. Download model files (see setup section below)

# 3. Run with Docker Compose
docker-compose up

# 4. Open browser
open http://localhost:8000
```

### Option 2: Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Setup

### 1. Download Model Files

⚠️ **IMPORTANT**: Model files (~260MB) are NOT in Git. Download from HuggingFace Spaces:

Visit: https://huggingface.co/spaces/Subramanyam6/HDR_AI_Proposal_Verification_Assistant_V2/tree/main/huggingface_space/model

Download and place in `backend/app/models/`:
- `vectorizer.pkl` (405 KB)
- `classifier.pkl` (314 KB)
- `config.json` (373 bytes)
- `distilbert/model.safetensors` (255 MB)
- `distilbert/config.json`, `distilbert/tokenizer.json`, etc.

See `backend/app/models/README.md` for complete file list.

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env.development

# Add your OpenAI API key (optional, for GPT-4o suggestions)
OPENAI_API_KEY=sk-your-key-here
```

### 3. Build and Run

```bash
# Build Docker image
docker build -t hdr-verification .

# Run container
docker-compose up

# Or run directly
docker run -p 8000:8000 --env-file .env.development hdr-verification
```

## Model Performance

Based on 7,200 training samples, 1,400 dev samples:

| Model | Micro-F1 | Macro-F1 | Inference Speed |
|-------|----------|----------|-----------------|
| **TF-IDF + LogReg** | **0.410** | **0.414** | Very Fast |
| Naive Bayes | 0.482 | 0.464 | Very Fast |
| DistilBERT | 0.324 | 0.225 | Moderate |

## Training New Models

To generate synthetic data and train models, see [TRAINING.md](TRAINING.md).

```bash
cd synthetic_proposals
make build                    # Generate 10,000 proposals
python scripts/08_tfidf_logreg.py    # Train TF-IDF model
python scripts/09_transformer_mps_verified.py --max-epochs 20  # Train DistilBERT
cp -r dataset/model/* ../backend/app/models/  # Export to backend
```

## API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Frontend**: http://localhost:8000/

### Key Endpoints

**POST /api/verify**
- Upload PDF or paste text
- Returns ML predictions + rule-based checks + AI suggestions

**GET /api/samples**
- Returns all sample proposal texts

## Deployment

### HuggingFace Spaces

Auto-deploys from `main` branch via GitHub Actions:

```bash
git push origin main
```

Requires secrets:
- `HF_TOKEN` - HuggingFace access token
- `HF_USERNAME` - HuggingFace username

### Azure Container Instances

```bash
az container create \
  --resource-group hdr-rg \
  --name hdr-verification \
  --image ghcr.io/subramanyam6/hdr_ai_proposal_verification_assistant:latest \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
```

## Documentation

- `TRAINING.md` - Complete model training guide
- `CLAUDE.md` - Claude Code project instructions (archived)
- `backend/app/models/README.md` - Model file download instructions
- `synthetic_proposals/README.md` - Dataset pipeline details

## Architecture

```
React Frontend (Port 8000)
    ↓ HTTP
FastAPI Backend (/api/*)
    ↓
┌─────────────────────────┐
│ PDF Extractor (PyPDF2)  │
│ TF-IDF Model            │
│ DistilBERT Model        │
│ Naive Bayes Model       │
│ Rule Engine (Regex)     │
│ RAG Service (GPT-4o)    │
└─────────────────────────┘
```

## License

MIT License

## Author

Subramanyam
