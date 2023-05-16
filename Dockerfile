FROM python:3.9.0
RUN pip install --upgrade pip

# Run application as non-root in the Docker container.
# Every COPY instruction needs the --chown=<user>:<group> flag to signal to
# change the file or directory owner to the worker user (it’s root by default).
RUN adduser --disabled-login worker
USER worker
WORKDIR /home/worker

# Copy all content from the current directory.
COPY --chown=worker:worker . /home/worker

# Install pyenv.
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="/home/worker/.pyenv"
ENV PATH="/home/worker/.pyenv/bin:${PATH}"

# Set mlflow tracking uri.
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Install deps.
RUN pip install --user --no-warn-script-location -r /home/worker/requirements.txt
ENV PATH="/home/worker/.local/bin:${PATH}"
ENV JUPYTER_ENABLE_LAB=yes

CMD cd DataScienceMLTest/ && mlflow server --host=0.0.0.0
