# Deep Learning for EM images

This repository contains the code to make semantic segmentation using U-Net based architecture for EM images. This code is based on Keras and TensorFlow as backend. 

Furthemore, some of state-of-the-art approaches have been reproduced and implemented to compare our method in a more robust way, with the goal of supplement the lack of information in some of those works. 

## Getting started 
These instructions will get you a copy of the project up and running on your machine.

### Prerequisites
To set-up a development environment with all necessary dependencies, you can use the  file located in [env/DL_EM_base_env.yml](env/DL_EM_base_env.yml) to create it as follows:

```
conda env create -f DL_EM_base_env.yml
```

### Choose a template
In [templates](templates/) directory are located a few different templates that could be used to start your project. Each one is suited to different settings:

- [template.py](templates/template.py): use this template as a baseline to make segmantic segmentation with an 2D U-Net on small datasets.
- [big_data_template.py](templates/big_data_template.py): same as the first, but should be used with large datasets, as it makes use of `flow_from_directory()` instead of `flow()` method. Notice that the dataset directory structure changes.
- [3D_template.py](templates/3D_template.py): use this template as a baseline to make segmantic segmentation with a 3D U-Net on small datasets.

In case you are interested in reproducing one of the state-of-the-art works implemented in this project, you can use the template prepared on each case: [xiao_template.py](xiao_2018/xiao_template.py), [cheng_template.py](cheng_2017/cheng_template.py), [oztel_template.py](oztel_2017/oztel_template.py) or [casser__template.py](casser_2018/casser_template.py). 

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
	│   	├── im0500.png
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

For instance, one of EM datasets used on this work can be downloaded [here](https://www.epfl.ch/labs/cvlab/data/data-em/ "EPFL").

### Run the code 
An example to run it in bash shell could be this:
```Bash
# Load the environment created first
conda activate DL_EM_base_env     

code_dir="/home/user/DeepLearning_EM"  # Path to this repo code 
data_dir="/home/user/dataset"          # Path to the dataset
job_dir="/home/user/out_dir"           # Path where the output data will be generated
job_id="400"                           # Just a string to identify the job 
job_counter=0                          # Number that should be increased when one need to run the same job multiple times
gpu_number="0"                         # Number of the GPU to run the job in (according to 'nvidia-smi' command)

python -u template.py ${code_dir} "${data_dir}" "${job_dir}" --id "${jobID}" --rid "${jobCounter}" --gpu ${gpu_number} 
```

### Other state-of-the-art implementations

The following state-of-the-art approaches for EM semantic segmentation have been implemented:

- 3D U-Net + post-processing. Everything contained in [xiao_2018](xiao_2018). 
```
Chi Xiao, Xi Chen, Weifu Li, Linlin Li, Lu Wang, Qiwei Xie, and Hua Han, "Automatic mitochondria 
segmentation for em data using a 3d supervised convolutional network", Frontiers in Neuroanatomy 
12 (2018), 92.
```

- 2D and 3D U-Net with proposed stochastic downsampling layer. Everything contained in [cheng_2017](cheng_2017).
```
H. Cheng and A. Varshney, "Volume segmentation using convolutional neural networks with limited 
training data," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, 
pp. 590-594.
```

- CNN + post-processing. Everything contained in [oztel_2017](oztel_2017).
```
Ismail Oztel, Gozde Yolcu, Ilker Ersoy, Tommi White, and Filiz Bunyak, "Mitochondria segmentation 
in electron microscopy volumes using deep convolutional neural network", 2017 IEEE International 
Conference on Bioinformatics and Biomedicine (BIBM), IEEE, 2017, pp. 1195-1200.
``` 

- 2D U-Net + post-processing. Everything contained in [casser_2018](casser_2018).
```
Vincent Casser, Kai Kang, Hanspeter Pfister, and Daniel Haehn, "Fast mitochondria segmentation 
for connectomics", arXiv preprint arXiv:1812.06024 (2018).
```
