FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu118-py38-torch200

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip list