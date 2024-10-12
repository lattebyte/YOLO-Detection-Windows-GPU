## Getting Started


### Setup
We recommend setting up a virtual environment. Using e.g. Anaconda run as administration, the `YOLO` package can be installed via:

```bash
cd /target/folder
git clone https://github.com/lattebyte/YOLO-Detection-Windows-GPU.git
conda create -n yolo_env -y python=3.9
conda activate yolo_env
```

Install PyTorch with CUDA for GPU based computing. The example below is for CUDA 11.8:
CUDA Installation: https://developer.nvidia.com/cuda-11-8-0-download-archive
PyTorch Installation: https://developer.nvidia.com/cuda-11-8-0-download-archive

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics
pip install opencv-python
```

Use local video
or
Connect USB camera/ webcam

### Running from python

AI model will be automatically downloaded:
```bash
python yolo_det_pos.py
```
