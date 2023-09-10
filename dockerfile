# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

EXPOSE 7001

COPY . app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


WORKDIR /app
# Install pip requirements
#COPY requirements.txt .
#COPY flask_server.py .
#COPY yolo_v8 yolo_v8
RUN echo "deb http://archive.debian.org/debian stretch main" > /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx libxext6 libglib2.0-0 -y

RUN cd /app
RUN pip install -r requirements.txt

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT [ "python", "./flask_server.py" ] 
CMD [ "python", "./flask_server.py" ] 