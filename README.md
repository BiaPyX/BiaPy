# EM image segmentation

This repository contains a complete workflow to make semantic segmentation for EM images. The code is based on Keras and TensorFlow as backend. For further implementation details and project usage please visit our [documentation](https://em-image-segmentation.readthedocs.io/en/latest/).

![.](https://github.com/danifranco/EM_Image_Segmentation/blob/master/docs/source/img/seg.gif)

## Getting started 
These instructions will get you a copy of the project up and running on your machine.

Run a fast-version of 2D U-Net template in Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/danifranco/EM_Image_Segmentation/tree/master/templates/U-Net_2D_workflow.ipynb) 

### Prerequisites
To set-up a development environment with all necessary dependencies, you can use the  file located in [env/DL_EM_base_env.yml](env/DL_EM_base_env.yml) to create it as follows:

```
conda env create -f env/DL_EM_base_env.yml
```

### Choose a template
In [templates](templates/) directory are located different templates that reproduce the results presented in our paper. 

[U-Net_2D_template.py](templates/U-Net_2D_template.py) and [U-Net_3D_template.py](templates/U-Net_3D_template.py) are the main templates and the rest differ in the network used (together with other minor changes in cases where the network needs them). As an exception, two different templates on this folder need an special explanation:

- [big_data_template.py](templates/big_data_template.py): use this template as a baseline to make segmantic segmentation with an 2D U-Net. The difference with [U-Net_2D_template.py](templates/U-Net_2D_template.py) is that this template should be used with large datasets, as it uses `flow_from_directory()` instead of `flow()` function of Keras. Notice that the dataset directory structure changes.
- [general_template.py](templates/general_template.py): in this template we gather all implemented options that this project covers. Should be used to extract different code blocks you are interested in. 

To run state-of-the-art works implemented in this project you can use the template prepared on each case: [xiao_template_V1.py](sota_implementations/xiao_2018/xiao_template_V1.py), [cheng_2D_template_V1.py](sota_implementations/cheng_2017/cheng_2D_template_V1.py), [cheng_3D_template_V1.py](sota_implementations/cheng_2017/cheng_3D_template_V1.py), [oztel_template_V1.py](sota_implementations/oztel_2017/oztel_template_V1.py) or [casser_template_V1.py](sota_implementations/casser_2018/casser_template_V1.py). 

### Data structure

This project follows the same directory structure as [ImageDataGenerator](https://keras.io/preprocessing/image/) class of Keras. The data directory tree should be this:

<details> <summary>Details</summary>

```
dataset/
├── test
│   ├── x
│   │   ├── testing-0001.tif
│   │   ├── testing-0002.tif
│   │   ├── . . .
│   └── y
│       ├── testing_groundtruth-0001.tif
│       ├── testing_groundtruth-0002.tif
│       ├── . . .
└── train
    ├── x
    │   ├── training-0001.tif
    │   ├── training-0002.tif
    │   ├── . . .
    └── y
        ├── training_groundtruth-0001.tif
        ├── training_groundtruth-0002.tif
        ├── . . .
```

</details>

However, as you should be familiarized with, when big datasets are used, which should be using a code based on 3D_template.py, the directory tree changes a little bit. This is because the usage of `flow_from_directory()`, which needs the data to be structured as follows:

<details> <summary>Details</summary>

```
dataset/
├── test
│   ├── x
│   │   └── x
│   │       ├── im0500.png
│   │       ├── im0501.png
│   │       ├── . . .
│   └── y
│       └── y
│   │       ├── im0500.png
│   │       ├── im0501.png
│   │       ├── . . .
└── train
    ├── x
    │   └── x
    │       ├── im0500.png
    │       ├── im0501.png
    │       ├── . . .
    └── y
        └── y
            ├── mask_0097.tif
            ├── mask_0098.tif
            ├── mask_0097.tif
            ├── . . .
```
</details>

EM datasets used on this work:
- [Lucchi](https://www.epfl.ch/labs/cvlab/data/data-em/ "EPFL")
- [Lucchi++](https://sites.google.com/view/connectomics/ "Lucchi++")
- [Kasthuri++](https://sites.google.com/view/connectomics/ "Kasthuri++")

### Run the code 
An example to run it in bash shell could be this:
```Bash
# Load the environment created first
conda activate DL_EM_base_env     

code_dir="/home/user/EM_Image_Segmentation"  # Absolute path to this repo code 
data_dir="/home/user/dataset"                # Absolute path to the dataset
job_dir="/home/user/out_dir"                 # Absolute path where the output data will be generated
job_id="400"                                 # Just a string to identify the job 
job_counter=0                                # Number that should be increased when one need to run the same job multiple times
gpu_number="0"                               # Number of the GPU to run the job in (according to 'nvidia-smi' command)

python -u template.py $code_dir $data_dir $job_dir --id $jobID --rid $jobCounter --gpu $gpu_number 
```

### Other state-of-the-art implementations

The following state-of-the-art approaches for EM semantic segmentation have been implemented:

- 3D U-Net + post-processing. Everything contained in [xiao_2018](sota_implementations/xiao_2018). 
```
Chi Xiao, Xi Chen, Weifu Li, Linlin Li, Lu Wang, Qiwei Xie, and Hua Han, "Automatic mitochondria 
segmentation for em data using a 3d supervised convolutional network", Frontiers in Neuroanatomy 
12 (2018), 92.
```

- 2D and 3D U-Net with proposed stochastic downsampling layer. Everything contained in [cheng_2017](sota_implementations/cheng_2017).
```
H. Cheng and A. Varshney, "Volume segmentation using convolutional neural networks with limited 
training data," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, 
pp. 590-594.
```

- CNN + post-processing. Everything contained in [oztel_2017](sota_implementations/oztel_2017).
```
Ismail Oztel, Gozde Yolcu, Ilker Ersoy, Tommi White, and Filiz Bunyak, "Mitochondria segmentation 
in electron microscopy volumes using deep convolutional neural network", 2017 IEEE International 
Conference on Bioinformatics and Biomedicine (BIBM), IEEE, 2017, pp. 1195-1200.
``` 

- 2D U-Net + post-processing. Everything contained in [casser_2018](sota_implementations/casser_2018).
```
Vincent Casser, Kai Kang, Hanspeter Pfister, and Daniel Haehn, "Fast mitochondria segmentation 
for connectomics", arXiv preprint arXiv:1812.06024 (2018).
```

