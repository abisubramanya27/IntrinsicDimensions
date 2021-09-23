# Latest nvidia ngc pytorch image
FROM  nvcr.io/nvidia/pytorch:21.08-py3

# Setting the work directory
WORKDIR /root

# Install Jupyter and colab requirements
RUN pip --use-deprecated=legacy-resolver install jupyterlab jupyter_http_over_ws ipywidgets  \
    && jupyter serverextension enable --py jupyter_http_over_ws   \
    && jupyter nbextension enable --py widgetsnbextension

# Expose port for notebook and run jupyter
ARG COLAB_PORT=1234
EXPOSE 1234
ENV COLAB_PORT=1234
