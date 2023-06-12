# import pre-built image nvidia/cuda that comes with pre-installed CUDA packages
FROM nvidia/cuda:11.4.0-base-ubuntu20.04

# Install Python 3.7 and pip
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get -y install python3.7
RUN apt-get -y install python3-pip
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN python3.7 -m pip install --upgrade --force pip

# unset PYTHONPATH to prevent problems later
ENV PYTHONPATH=

# Install wget
RUN apt-get update \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

# Installing Miniconda
ENV MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
ENV MINICONDA_PREFIX=/usr/local
RUN wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
RUN chmod +x $MINICONDA_INSTALLER_SCRIPT
RUN ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# Updating Conda with python=3.7 (Required for Xtreme compatibility)
RUN conda install --channel defaults conda python=3.7 --yes
RUN conda update --channel defaults --all --yes

# Install Git
RUN apt-get update && apt-get install -y git

# Xtreme codebase setup
WORKDIR /root
RUN git clone https://github.com/google-research/xtreme.git /root/mount/xtreme

# Add install_tools.sh from folder to replace the one in the xtreme repo
COPY install_tools.sh /root/mount/xtreme
# Giving permissions to run scripts from xtreme/
RUN chmod -R +x /root/mount/xtreme
RUN /root/mount/xtreme/install_tools.sh

# Replace train.sh, train_qa.sh, predict_qa.sh in xtreme/
COPY train.sh /root/mount/xtreme/scripts
COPY train_qa.sh /root/mount/xtreme/scripts
COPY predict_qa.sh /root/mount/xtreme/scripts

# Add the main scripts (run.sh, predict.sh, pre_evaluate.sh, evaluate.sh) to root folder to execute from the image
COPY run.sh /root
COPY predict.sh /root
COPY pre_evaluate.sh /root
COPY evaluate.sh /root

# Install Jupyter and colab requirements
RUN pip --use-deprecated=legacy-resolver install jupyterlab jupyter_http_over_ws ipywidgets google-colab  \
    && jupyter serverextension enable --py jupyter_http_over_ws   \
    && jupyter nbextension enable --py widgetsnbextension

# Expose port for notebook and run jupyter
ARG COLAB_PORT=8081
EXPOSE 8081
ENV COLAB_PORT=8081
CMD [ "/bin/sh", "-c",    "jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --allow-root --port $COLAB_PORT --NotebookApp.port_retries=0 --ip 0.0.0.0"]