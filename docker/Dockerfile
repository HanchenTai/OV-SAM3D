ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM aikedaer/cuda113_cudnn_torch1.12:latest
ENV TORCH_CUDA_ARCH_LIST="6.1+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

COPY sources.list /home/sources.list
RUN mv /home/sources.list /etc/apt/sources.list  
RUN apt-get update && apt-get install -y --no-install-recommends vim wget tmux htop openssh-server x11-apps && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd
RUN echo 'root:q' |chpasswd
RUN sed -ri 's/^#*PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/^#*Port\s+.*/Port 3333/g' /etc/ssh/sshd_config
RUN echo 'X11UseLocalHost yes' >> /etc/ssh/sshd_config
RUN echo 'AllowTcpForwarding yes' >> /etc/ssh/sshd_config

ADD requirements.txt /home/requirements.txt
RUN cd /home && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN apt-get update && apt-get install -y git
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

RUN apt-get update && apt-get install -y build-essential python3-dev libopenblas-dev
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torchmetrics==0.11.0
RUN pip install setuptools==68.0.0

RUN pip install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 --no-deps
RUN pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588 --no-deps

RUN ln -s /usr/bin/python3 /usr/bin/python
ENV MAX_JOBS=4

CMD ["/usr/sbin/sshd", "-D"]
