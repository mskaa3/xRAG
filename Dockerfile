# FROM  nvidia/cuda:12.2.2-devel-ubuntu20.04
# ENV PATH /opt/conda/bin:$PATH
# WORKDIR /opt/app

FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"


RUN apt-get update --fix-missing && \
    apt-get install -y wget git&& \
    apt-get clean
RUN apt-get install -y libaio-dev

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
# RUN /bin/bash ~/miniconda.sh -b -p /opt/conda 

# RUN echo "source activate base" > ~/.bashrc
# RUN conda install -y python=3.9
# RUN conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# RUN pip install transformers==4.38.0 accelerate==0.27.2 datasets==2.17.1 deepspeed==0.13.2 sentencepiece wandb
# RUN pip install flash-attn --no-build-isolation --no-cache-dir --upgrade --force-reinstall
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt
RUN pip uninstall -y transformer-engine

RUN MAX_JOBS=1 pip install flash-attn==2.5.8 --no-build-isolation
# CMD ["bash"]
RUN sed -i 's/force=True/force=False/g' /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py