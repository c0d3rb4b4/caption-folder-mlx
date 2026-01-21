# caption-folder-mlx

## Install
```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U mlx-vlm pillow
```

## Run
```
python caption_folder_mlx.py /path/to/images --recursive
```

## Another example

```
python caption_folder_mlx.py /path/to/images --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```