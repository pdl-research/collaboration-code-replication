# Reproducible environment for Lynch et al. analysis of the Anthropic Economic Index
# This Dockerfile pins the OS, Python version, and all system-level dependencies
# to ensure identical results across machines.

FROM python:3.11.9-slim-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies required by numpy/scipy compiled extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command: open a shell
CMD ["bash"]
