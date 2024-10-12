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
Connect Lattebyte CM16A series camera/ webcam

### Running from python using any YOLO model

AI model will be automatically downloaded:  
To use YOLO11n:  
```bash
python yolo_det_pos.py yolo11n
```
To use YOLOv8m:  
```bash
python yolo_det_pos.py yolov8m
```

To use YOLOv5nu:  
```bash
python yolo_det_pos.py yolov5nu
```