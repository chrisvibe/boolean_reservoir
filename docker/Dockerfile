FROM continuumio/anaconda3:2024.10-1

# Set the working directory
WORKDIR /code

# Copy the source code
COPY ./src /tmp/docker/src

# Set the PYTHONPATH
ENV PYTHONPATH=/code

# Create the conda environment
RUN conda env create --file /tmp/docker/src/environment.yaml

# Install additional packages
RUN apt-get update && apt-get install -y \
    tree \
    neovim \
    graphviz

# Create environment activation script
RUN echo "source activate boolean_reservoir" > /etc/profile.d/activate_boolean_reservoir.sh

# Ensure the script is executable
RUN chmod +x /etc/profile.d/activate_boolean_reservoir.sh

# Use bash as the default shell and source the environment activation script
SHELL ["bash", "--login", "-c"]

# Default command
CMD ["bash"]

