FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git ca-certificates && apt-get clean
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install --upgrade google-cloud-storage
RUN pip install jupyterlab


CMD ["sh","-c", "jupyter lab --no-browser --ip=0.0.0.0 --allow-root --notebook-dir='/code' --port=8888 --LabApp.token='' --LabApp.allow_origin='*' --LabApp.base_url=$OCTOPUS_JPY_BASE_URL"]
# COPY VocCode VocCode
# COPY CityCode CityCode
