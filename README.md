
# RCNNPose PyTorch + Multiple Object Tracker
**Mask R-CNN and Keypoint R-CNN wrapper + BYOL and SimSiam wrapper for Multiple Object Tracking with PyTorch.**

## Installation
#### Prerequisites
* Install [PyTorch](https://pytorch.org/get-started/locally/) `>= 1.2` with [torchvision](https://pytorch.org/get-started/locally/) `>= 0.4`
* [CKPT](https://drive.google.com/file/d/1tYw3Ikdm24kJT9SJwnbbDCYsCMxgvC9l/view?usp=sharing) unzip at ./simsiam/ckpt/
* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) `>= 10.0` , [Cudnn](https://developer.nvidia.com/rdp/cudnn-download) `>= 7.6.5` (test env)


```
git clone https://github.com/jimilee/rcnnpose-pytorch-tracker.git
```

```
pip install -r requirements.txt
```


## Running examples
```
cd examples
python image_demo.py
python video_demo.py
python tracker_demo.py
```

## License
* [RCNNPose PyTorch License](https://github.com/prasunroy/rcnnpose-pytorch/blob/master/LICENSE)
* [Barlowtwins License](https://github.com/facebookresearch/barlowtwins/blob/main/LICENSE)
<br />
<br />
