<img src="img/biapy_logo.svg" width="450"></a>

# BiaPy: Bioimage analysis pipelines in Python

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-blue.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.12-orange.svg" /></a>
    <a href= "https://github.com/danifranco/BiaPy/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href= "https://biapy.readthedocs.io/en/latest/">
      <img src="https://img.shields.io/badge/Doc-Latest-2BAF2B.svg" /></a>
    <a href= "https://ieeexplore.ieee.org/abstract/document/10230593">
      <img src="https://img.shields.io/badge/Paper-2BAF2B.svg" /></a>
</p>

[BiaPy](https://github.com/danifranco/BiaPy) is an open source Python library for building bioimage analysis pipelines. This repository is actively under development by the Biomedical Computer Vision group at the [University of the Basque Country](https://www.ehu.eus/en/en-home) and the [Donostia International Physics Center](http://dipc.ehu.es/). 

The library provides an easy way to create image processing pipelines that are typically used in the analysis of biology microscopy images in 2D and 3D. Namely, BiaPy contains ready-to-use solutions for the tasks of [semantic segmentation](https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html), [instance segmentation](https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html), [object detection](https://biapy.readthedocs.io/en/latest/workflows/detection.html), [image denoising](https://biapy.readthedocs.io/en/latest/workflows/denoising.html), [single image super-resolution](https://biapy.readthedocs.io/en/latest/workflows/super_resolution.html), [self-supervised learning](https://biapy.readthedocs.io/en/latest/workflows/self_supervision.html) and [image classification](https://biapy.readthedocs.io/en/latest/workflows/classification.html). The source code is based on Keras/TensorFlow as backend. Given BiaPy’s deep learning based core, a machine with a graphics processing unit (GPU) is recommended for fast training and execution.                                                                                                                                            
![BiaPy workflows](./img/BiaPy-workflow-readme.svg)                                                                                                                                   
     
## Citation                                                                                                             
                                                                                                                                                                                                                           
```bibtex
@inproceedings{franco2023biapy,
  title={BiaPy: a ready-to-use library for Bioimage Analysis Pipelines},
  author={Franco-Barranco, Daniel and Andr{\'e}s-San Rom{\'a}n, Jes{\'u}s A and G{\'o}mez-G{\'a}lvez, Pedro and Escudero, Luis M and Mu{\~n}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
``` 
