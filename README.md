
# RCNNPose PyTorch + Multiple Object Tracker
**Mask R-CNN and Keypoint R-CNN wrapper + BYOL and SimSiam wrapper for Multiple Object Tracking with PyTorch.**

## Directory

```
rcnnpose-pytorch-tracker/
    ㄴ .idea/
    ㄴ examples/
        ㄴ media/
        ㄴ KalmanFilter.py   -- additional method
        ㄴ eval_MOT17.py     -- evaluation entire MOT Challenge datasets
        ㄴ tracker.py        -- tracker main
        ㄴ tracker_utils.py  -- tracker utils.
        ㄴ video_demo.py     -- demo(video)
        ㄴ image_demo.py     -- demo(image)
    ㄴ rcnnpose
        ㄴ __init__.py
        ㄴ estimator.py
        ㄴ utils.py 
    ㄴ simsiam
        ㄴ models/
        ㄴ simsiam_standalone.py 
    ㄴ trackeval

```

## Installation
#### Prerequisites
* Install [PyTorch](https://pytorch.org/get-started/locally/) `>= 1.2` with [torchvision](https://pytorch.org/get-started/locally/) `>= 0.4`
* Download [ckpt](https://drive.google.com/file/d/1tYw3Ikdm24kJT9SJwnbbDCYsCMxgvC9l/view?usp=sharing) and unzip at ./simsiam/ckpt/
* [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) `>= 10.0` , [Cudnn](https://developer.nvidia.com/rdp/cudnn-download) `>= 7.6.5` (test env)
* [TrackEval](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md) -> officail python evaluation code & data

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
