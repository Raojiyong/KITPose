# Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimation (IJCV 2025)

## Introduction
This is an official pytorch implementation of [*Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimatio*](https://link.springer.com/article/10.1007/s11263-025-02355-0). 
In this work, to achieve general mammal pose estimation, we developed a novel keypoints-interactive model, namely KITPose, to pursue structure-supporting dependencies among keypoints and body parts. In particular, we explicitly disentangle the keypoint-specific clues from the backbone features without any spatial splitting. An effective design named Generalised Heatmap Regression Loss is proposed to enable the adaptive adjustment of intermediate features to optimise keypoint representations.
Simultaneously, to preserve the semantic information in the image, a new concept, referred to as body part prompts, is introduced to provide discriminative context, organising the information interactions. Furthermore, to automatically balance the importance between each keypoints, a novel adaptive weight strategy is introduced to common MSE loss. The designed architecture reflects its superiority and generalisation for general mammal pose estimation, which has been evaluated through extensive  experiments on the AP10K, AnimalKingdom, and COCO datasets.

![Illustrating the architecture of the proposed KITPose](/figures/kitpose.png)
## Main Results
### Results on AP10k val
| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| pose_resnet_50     | 96.4 |     95.3 |  89.0 |  83.2 | 88.4 | 84.0 |  79.6 | 88.5 |     34.0 |
| pose_resnet_101    | 96.9 |     95.9 |  89.5 |  84.4 | 88.4 | 84.5 |  80.7 | 89.1 |     34.0 |
| pose_resnet_152    | 97.0 |     95.9 |  90.0 |  85.0 | 89.2 | 85.3 |  81.3 | 89.6 |     35.0 |
| **pose_hrnet_w32** | 97.1 |     95.9 |  90.3 |  86.4 | 89.1 | 87.1 |  83.3 | 90.3 |     37.7 |

### Note:
- Flip test is used.
- Input size is 256x256
- pose_resnet_[50,101,152] is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| pose_resnet_50     |    256x192 | 34.0M   |    8.9 | 0.704 | 0.886 |  0.783 |  0.671 |  0.772 | 0.763 | 0.929 |  0.834 |  0.721 |  0.824 |
| pose_resnet_50     |    384x288 | 34.0M   |   20.0 | 0.722 | 0.893 |  0.789 |  0.681 |  0.797 | 0.776 | 0.932 |  0.838 |  0.728 |  0.846 |
| pose_resnet_101    |    256x192 | 53.0M   |   12.4 | 0.714 | 0.893 |  0.793 |  0.681 |  0.781 | 0.771 | 0.934 |  0.840 |  0.730 |  0.832 |
| pose_resnet_101    |    384x288 | 53.0M   |   27.9 | 0.736 | 0.896 |  0.803 |  0.699 |  0.811 | 0.791 | 0.936 |  0.851 |  0.745 |  0.858 |
| pose_resnet_152    |    256x192 | 68.6M   |   15.7 | 0.720 | 0.893 |  0.798 |  0.687 |  0.789 | 0.778 | 0.934 |  0.846 |  0.736 |  0.839 |
| pose_resnet_152    |    384x288 | 68.6M   |   35.3 | 0.743 | 0.896 |  0.811 |  0.705 |  0.816 | 0.797 | 0.937 |  0.858 |  0.751 |  0.863 |
| **pose_hrnet_w32** |    256x192 | 28.5M   |    7.1 | 0.744 | 0.905 |  0.819 |  0.708 |  0.810 | 0.798 | 0.942 |  0.865 |  0.757 |  0.858 |
| **pose_hrnet_w32** |    384x288 | 28.5M   |   16.0 | 0.758 | 0.906 |  0.825 |  0.720 |  0.827 | 0.809 | 0.943 |  0.869 |  0.767 |  0.871 |
| **pose_hrnet_w48** |    256x192 | 63.6M   |   14.6 | 0.751 | 0.906 |  0.822 |  0.715 |  0.818 | 0.804 | 0.943 |  0.867 |  0.762 |  0.864 |
| **pose_hrnet_w48** |    384x288 | 63.6M   |   32.9 | 0.763 | 0.908 |  0.829 |  0.723 |  0.834 | 0.812 | 0.942 |  0.871 |  0.767 |  0.876 |

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- pose_resnet_[50,101,152] is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).
- GFLOPs is for convolution and linear layers only.


### Results on COCO test-dev2017 with detector having human AP of 60.9 on COCO test-dev2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| pose_resnet_152    |    384x288 | 68.6M   |   35.3 | 0.737 | 0.919 |  0.828 |  0.713 |  0.800 | 0.790 | 0.952 |  0.856 |  0.748 |  0.849 |
| **pose_hrnet_w48** |    384x288 | 63.6M   |   32.9 | 0.755 | 0.925 |  0.833 |  0.719 |  0.815 | 0.805 | 0.957 |  0.874 |  0.763 |  0.863 |
| **pose_hrnet_w48\*** |    384x288 | 63.6M   |   32.9 | 0.770 | 0.927 |  0.845 |  0.734 |  0.831 | 0.820 | 0.960 |  0.886 |  0.778 |  0.877 |

### Note:
- Flip test is used.
- Person detector has person AP of 60.9 on COCO test-dev2017 dataset.
- pose_resnet_152 is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).
- GFLOPs is for convolution and linear layers only.
- pose_hrnet_w48\* means using additional data from [AI challenger](https://challenger.ai/dataset/keypoint) for training.

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
   # Alternatively, if you do not have permissions or prefer
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
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   `-- KITPose_E2C6_w32_256x256.pth
         |-- ak
         |   |-- KITPose_E2C6_w32_256x256.pth
         |   |-- KITPose_E2C6_w48_256x256.pth
         `-- animalpose
         |   |-- pose_hrnet_w32_256x256.pth
         |   |-- pose_hrnet_w48_256x256.pth
         |   |-- pose_resnet_101_256x256.pth
         |   |-- pose_resnet_152_256x256.pth
         `-- coco
         |   |-- pose_hrnet_w32_256x256.pth
         |   |-- pose_hrnet_w48_256x256.pth
         |   |-- pose_resnet_101_256x256.pth
         |   |-- pose_resnet_152_256x256.pth

   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
```

### Visualization

#### Visualizing predictions on COCO val

```
python visualization/plot_coco.py \
    --prediction output/coco/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json \
    --save-path visualization/results

```


<img src="figures\visualization\coco\score_610_id_2685_000000002685.png" height="215"><img src="figures\visualization\coco\score_710_id_153229_000000153229.png" height="215"><img src="figures\visualization\coco\score_755_id_343561_000000343561.png" height="215">

<img src="figures\visualization\coco\score_755_id_559842_000000559842.png" height="209"><img src="figures\visualization\coco\score_770_id_6954_000000006954.png" height="209"><img src="figures\visualization\coco\score_919_id_53626_000000053626.png" height="209">



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