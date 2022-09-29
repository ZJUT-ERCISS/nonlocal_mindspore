# Contents
- [Contents](#contents)
  - [Description](#description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
  - [Training Process](#training-process)
  - [Evaluation Process](#evaluation-process)
  - [Citation](#citation)



## [Description](#contents)

This code is a re-implementation of the video classification experiments in the paper [Non-local Neural Networks](https://arxiv.org/abs/1711.07971). The code is developed based on the [Mindspore]((https://www.mindspore.cn/install/en)) framework.

## [Model Architecture](#contents)
<div align=center>
<img src=./src/pic/nonlocal_block.png> 

Figure 1 nonlocal_block </div>
A non-local operation is a flexible building block and can be easily used together with convolutional/recurrent layers. It can be added into the earlier part of deep neural networks, unlike fc layers that are often used in the end. This allows us to build a richer hierarchy that combines both non-local and local information.
<div align=center>
<img src=./src/pic/baseline_ResNet50_C2D.png> 

Table 1 baseline_ResNet50_C2D</div>
Table 1 shows our C2D baseline under a ResNet-50 backbone.In this repositories, we use the Inflated 3D ConvNet(I3D) under a ResNet-50 backbone. One can turn the C2D model in Table 1into a 3D convolutional counterpart by “inflating” the kernels. For example, a 2D k×k kernel can be inflated as a 3D t×k×k kernel that spans t frames. And we add 5 blocks (3 to res4 and 2 to res3, to every other residual block). For more information, please read the [paper](./src/example/1711.07971v1.pdf).
## [Dataset](#contents)

Dataset used: [Kinetics400](https://www.deepmind.com/open-source/kinetics)

- Description: Kinetics-400 is a commonly used dataset for benchmarks in the video field. For details, please refer to its official website [Kinetics](https://www.deepmind.com/open-source/kinetics). For the download method, please refer to the official address [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics), and use the download script provided by it to download the dataset.

- Dataset size：
    |category | Number of data  | 
    | :------: | :----------: | 
    |Training set | 238797 |  
    |Validation set | 19877 | 
Because of the expirations of some YouTube links, the sizes of kinetics dataset copies may be different.
```text
The directory structure of Kinetic-400 dataset looks like:

    .
    |-kinetic-400
        |-- train
        |   |-- ___qijXy2f0_000011_000021.mp4       // video file
        |   |-- ___dTOdxzXY_000022_000032.mp4       // video file
        |    ...
        |-- test
        |   |-- __Zh0xijkrw_000042_000052.mp4       // video file
        |   |-- __zVSUyXzd8_000070_000080.mp4       // video file
        |-- val
        |   |-- __wsytoYy3Q_000055_000065.mp4       // video file
        |   |-- __vzEs2wzdQ_000026_000036.mp4       // video file
        |    ...
        |-- kinetics-400_train.csv                  // training dataset label file.
        |-- kinetics-400_test.csv                   // testing dataset label file.
        |-- kinetics-400_val.csv                    // validation dataset label file.

        ...
```
## [Environment Requirements](#contents)
To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - python 3.7.5
    - Cython 0.29.30
    - cython-bbox 0.1.3
    - decord 0.6.0
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - matplotlib 3.4.1
    - motmetrics 1.2.5
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - scikit-learn 1.0.2
    - scipy 1.7.3
    - pycocotools 2.0
- Hardware
    - Prepare hardware environment with GPU(Nvidia).
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

### [Requirements Installation](#contents)

Some packages in `requirements.txt` need Cython package to be installed first. For this reason, you should use the following commands to install dependencies:

```shell
pip install Cython && pip install -r requirements.txt
```

### [Dataset Preparation](#contents)

Nonlocal model uses kinetics400 dataset to train and validate in this repository. 

### [Model Checkpoints](#contents)

Our non-local model which migrated from the pretrain model for pytorch [i3d_nl_dot_product_r50](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb_20200814-7c30d5bb.pth) is finetuned on the Kinetics400 dataset for 1 epochs.
It can be downloaded here: [[nonlocal_kinetics400_mindspore.ckpt]](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/Ec-B_Hr00QRAs49Vd7Qg4PkBslya1SjAola4hg64tpI6Vg?e=YNm0Ig)

### [Running](#contents)

To train or finetune the model, you can run the following script:

```shell

cd scripts/

# run training example
bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]


```
To validate the model, you can run the following script:
```shell
cd scripts/

# run evaluation example
bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
│  infer.py                                     // infer script
│  README.md                                    // descriptions about Nonlocal
│  train.py                                     // training scrip
└─src
    ├─config
    │      nonlocal.yaml                        // Nonlocal parameter configuration
    ├─data
    │  │  builder.py                            // build data
    │  │  download.py                           // download dataset
    │  │  generator.py                          // generate video dataset
    │  │  images.py                             // process image
    │  │  kinetics400.py                        // kinetics400 dataset
    │  │  meta.py                               // public API for dataset
    │  │  path.py                               // IO path
    │  │  video_dataset.py                      // video dataset
    │  │
    │  └─transforms
    │          builder.py                       // build transforms
    │          video_center_crop.py             // center crop
    │          video_normalize.py               // normalize
    │          video_random_crop.py             // random crop
    │          video_random_horizontal_flip.py  // random horizontal flip
    │          video_reorder.py                 // reorder
    │          video_rescale.py                 // rescale
    │          video_short_edge_resize.py       // short edge resize
    │
    ├─example
    │      nonlocal_kinetics400_eval.py         // eval nonlocal model
    │      nonlocal_kinetics400_train.py        // train nonlocal model
    │
    ├─loss
    │      builder.py                           // build loss
    │
    ├─models
    │  │  builder.py                            // build model
    │  │  nonlocal3d.py                                // nonlocal model
    │  │
    │  └─layers
    │          adaptiveavgpool3d.py             // adaptive average pooling 3D.
    │          dropout_dense.py                 // dense head
    │          inflate_conv3d.py                // inflate conv3d block
    |          maxpool3d.py                     // 3D max pooling
    |          maxpool3dwithpad.py              // 3D max pooling with padding operation
    │          resnet3d.py                      // resnet backbone
    │          unit3d.py                        // unit3d module
    │
    ├─optim
    │      builder.py                           // build optimizer
    │
    ├─schedule
    │      builder.py                           // build learning rate shcedule
    │      lr_schedule.py                       // learning rate shcedule
    │
    └─utils
            callbacks.py                        // eval loss monitor
            check_param.py                      // check parameters
            class_factory.py                    // class register
            config.py                           // parameter configuration
            six_padding.py                      // convert padding list into tuple

```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in nonlocal.yaml

- config for Nonlocal, Kinetics400 dataset

```text
# ==============================================================================
# model architecture
model_name: "nonlocal"

# The dataset sink mode.
dataset_sink_mode: False

# Context settings.
context:
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"

# model settings of every parts
model:
    type: nonlocal3d
    in_d: 32
    in_h: 224
    in_w: 224
    num_classes: 400
    keep_prob: 0.5

# learning rate for training process
learning_rate:
    lr_scheduler: "cosine_annealing"
    lr: 0.0003
    lr_epochs: [2, 4]
    lr_gamma: 0.1
    eta_min: 0.0
    t_max: 100
    max_epoch: 5
    warmup_epochs: 1

# optimizer for training process
optimizer:
    type: 'SGD'
    momentum: 0.9
    weight_decay: 0.0001

loss:
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

train:
    pre_trained: True
    pretrained_model: "./ms_nonlocal_dot_kinetics400_finetune.ckpt"
    ckpt_path: "./output/"
    epochs: 100
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 10
    run_distribute: True

eval:
    pretrained_model: "./nonlocal-1_4975.ckpt"

infer:
    pretrained_model: "./nonlocal-1_4975.ckpt"
    batch_size: 1
    image_path: ""
    normalize: True
    output_dir: "./infer_output"

# Kinetic400 dataset config
data_loader:
    train:
        dataset:
              type: Kinetic400
              path: "/data/kinetics-dataset"
              split: 'train'
              seq: 32
              seq_mode: 'interval'
              num_parallel_workers: 1
              shuffle: True
              batch_size: 6
              frame_interval: 6

        map:
            operations:
                - type: VideoShortEdgeResize
                  size: 256
                  interpolation: 'bicubic'
                - type: VideoRandomCrop
                  size: [224, 224]
                - type: VideoRandomHorizontalFlip
                  prob: 0.5
                - type: VideoRescale
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
                - type: VideoNormalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.255]
            input_columns: ["video"]

    eval:
        dataset:
            type: Kinetic400
            path: "/data/kinetics-dataset"
            split: 'val'
            seq: 32
            seq_mode: 'interval'
            num_parallel_workers: 1
            shuffle: False
            batch_size: 1
            frame_interval: 6
        map:
            operations:
                - type: VideoShortEdgeResize
                  size: 256
                  interpolation: 'bicubic'
                - type: VideoCenterCrop
                  size: [256, 256]
                - type: VideoRescale
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
                - type: VideoNormalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.255]               
            input_columns: ["video"]
    group_size: 1
# ==============================================================================

```

## [Training Process](#contents)

- train_distributed.log for Kinetics400

```shell
epoch: 1 step: 4975, loss is 0.44932037591934204
epoch: 1 step: 4975, loss is 0.3773573338985443
epoch: 1 step: 4975, loss is 0.19342052936553955
epoch: 1 step: 4975, loss is 0.5734817385673523
epoch: 1 step: 4975, loss is 0.09291025996208191
epoch: 1 step: 4975, loss is 0.5412027835845947
epoch: 1 step: 4975, loss is 0.08211661130189896
epoch: 1 step: 4975, loss is 0.9573349356651306
epoch time: 18000 s, per step time: 2064 ms
epoch time: 18000 s, per step time: 2063 ms
epoch time: 18000 s, per step time: 2064 ms
epoch time: 18000 s, per step time: 2064 ms
epoch time: 18001 s, per step time: 2065 ms
epoch time: 18001 s, per step time: 2065 ms
epoch time: 18001 s, per step time: 2065 ms
epoch time: 18002 s, per step time: 2066 ms
...
```

## [Evaluation Process](#contents)

- eval.log for Kinetics400

```text
[Start eval `nonlocal`]
eval: 1/19877
eval: 2/19877
eval: 3/19877
eval: 4/19877
eval: 5/19877
eval: 6/19877
eval: 7/19877
eval: 8/19877
eval: 9/19877
eval: 10/19877
...
eval: 19874/19877
eval: 19875/19877
eval: 19876/19877
eval: 19877/19877
{'Top_1_Accuracy': 0.7248, 'Top_5_Accuracy': 0.9072}
```
## [Citation](#contents)
```BibTeX
@article{NonLocal2018,
    author = {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
    title = {Non-local Neural Networks},
    year = {2018},
    journal = {CVPR},
    doi = {10.1109/CVPR.2018.00813},
}
```
```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}


