FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY . .

RUN pip install -e .[cu121_ampere]
RUN pip install transformers accelerate sentencepiece peft trl
RUN pip uninstall transformer-engine -y
RUN pip install torch torchvision torchaudio
RUN pip install bitsandbytes
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121