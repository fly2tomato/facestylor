FROM 10.1.32.6:28080/ai_open_platform/openmmlab-mmgen-facestylor-base:latest
WORKDIR /mmgen-facestylor
RUN pip install cmake && pip install dlib && pip install wget
RUN pip install gradio
RUN pip install ipdb
#RUN mkdir -p src/data && mkdir -p src/work_dirs/experiments && mkdir src/work_dirs/pre-trained && mkdir -p src/user/input && mkdir -p src/user/output
RUN export PYTHONUNBUFFERED=1
ADD src ./src
WORKDIR src
#ADD torch/checkpoints /root/.cache/torch/hub/checkpoints
ENTRYPOINT ["python", "app.py"]

