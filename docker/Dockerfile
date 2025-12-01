# Use your original Anaconda base image
FROM continuumio/anaconda3:2024.10-1
LABEL maintainer="vibechris@gmail.com"

# Create non-root user with UID 1000 and give sudo
RUN apt-get update && apt-get install -y sudo && \
    useradd -m -u 1000 -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-vscode && \
    chmod 0440 /etc/sudoers.d/90-vscode

# Install extra packages
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    tree \
    neovim \
    graphviz \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY ./src/environment.yaml /tmp/docker/src/environment.yaml

# Switch to vscode user
USER vscode

# Create conda environment as vscode user
RUN conda env create -f /tmp/docker/src/environment.yaml --yes

WORKDIR /code
ENV PYTHONPATH="/code"

# Make conda auto-activate for vscode
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate boolean_reservoir' >> ~/.bashrc

# Initialize conda
RUN conda init bash

# Default command
CMD [ "bash" ]
