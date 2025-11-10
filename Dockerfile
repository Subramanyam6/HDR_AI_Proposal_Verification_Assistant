# Multi-stage Dockerfile for React + FastAPI app (Single Container)
# Optimized for production deployment

# =============================================================================
# Stage 1: Build React frontend
# =============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --prefer-offline --no-audit

# Copy source code
COPY frontend/ ./

# Build frontend
RUN npm run build

# =============================================================================
# Stage 2: Python backend with models (Final image)
# =============================================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy model files
COPY backend/app/models/ ./backend/app/models/

# Copy sample files
COPY backend/app/samples/ ./backend/app/samples/

# Copy PDF files (generated from samples)
COPY backend/app/pdfs/ ./backend/app/pdfs/

# Copy frontend build from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI with uvicorn (HuggingFace Spaces uses port 7860)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]