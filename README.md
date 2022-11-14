# BiaPy: Bioimage analysis pipelines in Python

[BiaPy](https://github.com/danifranco/BiaPy) is an open source Python library for building bioimage analysis pipelines. This repository is actively under development by the Biomedical Computer Vision group at the [University of the Basque Country](https://www.ehu.eus/en/en-home) and the [Donostia International Physics Center](http://dipc.ehu.es/). 

The library provides an easy way to create image processing pipelines that are typically used in the analysis of biology microscopy images in 2D and 3D. Namely, BiaPy contains ready-to-use solutions for the tasks of [semantic segmentation](https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html), [instance segmentation](https://biapy.readthedocs.io/en/latest/workflows/instance_segmentation.html), [object detection](https://biapy.readthedocs.io/en/latest/workflows/detection.html), [image denoising](https://biapy.readthedocs.io/en/latest/workflows/denoising.html), [single image super-resolution](https://biapy.readthedocs.io/en/latest/workflows/super_resolution.html), [self-supervised learning](https://biapy.readthedocs.io/en/latest/workflows/self_supervision.html) and [image classification](https://biapy.readthedocs.io/en/latest/workflows/classification.html). The source code is based on Keras/TensorFlow as backend. Given BiaPy’s deep learning based core, a machine with a graphics processing unit (GPU) is recommended for fast training and execution.                                                                                                                                            
![BiaPy workflows](./img/BiaPy-workflow-readme.svg)                                                                                                                                   
     
## Citation                                                                                                             
                                                                                                                        
This repository is the base of the following work:                                                                      
                                                                                                                        
```bibtex
@Article{Franco-Barranco2021,
    author={Franco-Barranco, Daniel and Mu\~{n}oz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
    title={Stable Deep Neural Network Architectures for Mitochondria Segmentation on Electron Microscopy Volumes},
    journal={Neuroinformatics},
    year={2021},
    month={Dec},
    day={02},
    issn={1559-0089},
    doi={10.1007/s12021-021-09556-1},
    url={https://doi.org/10.1007/s12021-021-09556-1}
}
``` 
