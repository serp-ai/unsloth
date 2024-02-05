# Build
`docker build -t unsloth-env .`

## Run
`docker run --gpus all -it --rm --ipc=host -v $(pwd)/unsloth:/workspace -v C:/Users/labou/.cache/huggingface:/root/.cache/huggingface unsloth-env`

### Train (inside container)
`python train.py`