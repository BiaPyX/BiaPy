# EM image segmentation                                                                                                 
                                                                                                                        
This repository contains a complete workflow to perform semantic and instance segmentation of electron microscopy (EM) images. The code is based on Keras and TensorFlow as backend. 
                                                                                                                        
To use the code please visit our [documentation site](https://em-image-segmentation.readthedocs.io/en/latest/).         
                                                                                                                        
### Semantic segmentation                                                                                               
                                                                                                                        
![.](https://github.com/danifranco/EM_Image_Segmentation/blob/master/docs/source/img/seg.gif)                           
                                                                                                                        
### Instance segmentation                                                                                               

<table>
  <tr>
    <td>EM image</td>
     <td>Ground Truth</td>
     <td>Prediction</td>
  </tr>
  <tr>
    <td><img src="https://github.com/danifranco/EM_Image_Segmentation/blob/master/docs/source/video/nucmm_z_volume.gif" width=280></td>
    <td><img src="https://github.com/danifranco/EM_Image_Segmentation/blob/master/docs/source/video/nucmm_z_volume_mask.gif" width=280 ></td>
    <td><img src="https://github.com/danifranco/EM_Image_Segmentation/blob/master/docs/source/video/nucmm_z_volume_mask_pred.gif" width=280 ></td>
  </tr>
 </table>
 
## Citation                                                                                                             
                                                                                                                        
This repository is the base of the following work:                                                                      
                                                                                                                        
```bibtex
@Article{Franco-Barranco2021,
    author={Franco-Barranco, Daniel and Muñoz-Barrutia, Arrate and Arganda-Carreras, Ignacio},
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
