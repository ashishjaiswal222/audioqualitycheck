# Start from a slim python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables to strictly disable C-level threading (prevents CPU thrashing)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install system dependencies (ffmpeg and libsndfile1 for soundfile/pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy uv dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application
COPY . .

# Run model download script to cache weights in the image
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from app.verification.models_loader import init_worker_models; init_worker_models()"

# Expose port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
