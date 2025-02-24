# Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimation

[Tianyang Xu](https://xu-tianyang.github.io/), [Jiyong Rao](https://scholar.google.com/citations?user=bGxBmaQAAAAJ&hl=en&oi=ao), [Xiaoning Song](https://scholar.google.co.uk/citations?user=qIGhZCcAAAAJ&hl=en), [Zhenhua Feng](https://scholar.google.co.uk/citations?user=Y6KtijIAAAAJ&hl=en), and [Xiao-Jun Wu](https://scholar.google.co.uk/citations?user=5IST34sAAAAJ&hl=en), "Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimation", IJCV, 2025

[[paper](https://link.springer.com/article/10.1007/s11263-025-02355-0)] [[pretrained models](https://1drv.ms/f/c/516ca5af9c3a92b7/EqSxjqfURJRLnohjuJNnpuIBi8LAVUyJQ-cw7d39AbE4Mw?e=aMVTDo)]

#### 🔥🔥🔥 News

- **2025-02-24:** Code and pre-trained models are released. 🎊🎊🎊
- **2025-02-23:** This repo is released.

---

> **Abstract:**
This is an official pytorch implementation of [*Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimatio*](https://link.springer.com/article/10.1007/s11263-025-02355-0). 
In this work, to achieve general mammal pose estimation, we developed a novel keypoints-interactive model, namely KITPose, to pursue structure-supporting dependencies among keypoints and body parts. In particular, we explicitly disentangle the keypoint-specific clues from the backbone features without any spatial splitting. An effective design named Generalised Heatmap Regression Loss is proposed to enable the adaptive adjustment of intermediate features to optimise keypoint representations.
Simultaneously, to preserve the semantic information in the image, a new concept, referred to as body part prompts, is introduced to provide discriminative context, organising the information interactions. Furthermore, to automatically balance the importance between each keypoints, a novel adaptive weight strategy is introduced to common MSE loss. The designed architecture reflects its superiority and generalisation for general mammal pose estimation, which has been evaluated through extensive  experiments on the AP10K, AnimalKingdom, and COCO datasets.

![Illustrating the architecture of the proposed KITPose](/figures/kitpose.png)
## Main Results
### Results on AP10k val
## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not h[README.md](..%2FREADME.md)ave permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from our model zoo ([OneDrive](https://1drv.ms/f/c/516ca5af9c3a92b7/EqSxjqfURJRLnohjuJNnpuIBi8LAVUyJQ-cw7d39AbE4Mw?e=aqgYHV))
   ```
   ${POSE_ROOT}
    `-- models
         |-- ap10k
         |   |-- KITPose_E2C4_w32_256x256.pth
         |   |-- KITPose_E2C4_w48_256x256.pth
         |   |-- KITPose_E2C4_w32_384x384.pth
         |   |-- KITPose_E2C4_w48_384x384.pth
         |-- ak
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   |-- KITPose_E2C6_w48_256x256.pth
         `-- animalpose
         |   |-- KITPose_E2C5_w32_256x256.pth
         |   |-- KITPose_E2C5_w_256x256.pth
         `-- coco
         |   |-- KITPose_E2C4_w32_256x256.pth
         |   |-- KITPose_E2C4_w48_256x256.pth
         |   |-- KITPose_E2C4_w32_384x384.pth
         |   |-- KITPose_E2C4_w48_384x384.pth

   ```
   
### Data preparation

### Training and Testing

### Citation
If you use our code or models in your research, please cite with:
```
@article{xu2025learning,
  title={Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimation},
  author={Xu, Tianyang and Rao, Jiyong and Song, Xiaoning and Feng, Zhenhua and Wu, Xiao-Jun},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2025},
  publisher={Springer}
}
```