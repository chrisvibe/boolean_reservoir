FROM continuumio/anaconda3
WORKDIR /code
COPY ./src/* /tmp
ENV PYTHONPATH=/code

RUN apt-get update && apt-get install -y \
    tree \
    neovim \
    graphviz

RUN conda env create -f /tmp/environment.yml
# RUN /opt/conda/bin/conda init bash && \
#     echo "source /opt/conda/bin/activate boolean_reservoir" >> ~/.bashrc && \
#     /bin/bash -c "source ~/.bashrc && source /tmp/install.sh"
