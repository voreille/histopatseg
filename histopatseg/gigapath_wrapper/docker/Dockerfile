# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget git build-essential curl ca-certificates \
    python3 python3-pip python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH "$CONDA_DIR/bin:$PATH"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda init bash

# Clone and install gigapath using their recommended method
ARG HF_TOKEN
RUN git clone https://github.com/prov-gigapath/prov-gigapath.git && \
    cd prov-gigapath && \
    /opt/conda/bin/conda env create -f environment.yaml && \
    /opt/conda/bin/conda run -n gigapath pip install -e .

# Pre-download HF model with proper authentication
RUN mkdir -p /root/.cache/huggingface && \
    HF_TOKEN=$HF_TOKEN /opt/conda/bin/conda run -n gigapath python -c "import os, sys; \
    from huggingface_hub import login; \
    token = os.environ.get('HF_TOKEN'); \
    if token: \
    login(token); \
    import timm; \
    try: \
    timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True); \
    print('Model downloaded successfully') \
    except Exception as e: \
    print(f'Error loading model: {e}', file=sys.stderr); \
    if 'timm' in str(e): \
    print('timm package not found, installing...', file=sys.stderr); \
    import subprocess; \
    subprocess.check_call(['pip', 'install', 'timm']); \
    timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True); \
    print('Model downloaded successfully after installing timm') \
    else: \
    print('No HF_TOKEN provided, skipping model download', file=sys.stderr)"

# Copy your project code - structure it properly for imports
COPY histopatseg /app/histopatseg

# Make the CLI script executable
RUN chmod +x /app/histopatseg/gigapath/cli/extract_embeddings.py

# Set entrypoint to the CLI script
ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "gigapath", "python", "/app/histopatseg/gigapath/cli/extract_embeddings.py"]