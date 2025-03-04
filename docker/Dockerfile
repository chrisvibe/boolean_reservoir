# Use an Anaconda base image
FROM continuumio/anaconda3:2024.10-1

# Maintainer info (optional)
LABEL maintainer="vibechris@gmail.com"

# Set the working directory
WORKDIR /code
ENV PYTHONPATH="/code"

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    bzip2 \
    tree \
    neovim \
    graphviz \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the environment file and create the conda environment
COPY ./src/environment.yaml /tmp/docker/src/environment.yaml
RUN conda env create -f /tmp/docker/src/environment.yaml --yes

# Automatically activate the conda environment when opening a new shell
RUN echo "conda activate boolean_reservoir" >> ~/.bashrc

# Set the conda environment to be activated by default
SHELL ["/bin/bash", "--login", "-c"]

# Default command
CMD ["bash"]

