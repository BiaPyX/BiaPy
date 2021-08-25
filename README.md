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
                                                                                                                        
```                                                                                                                     
@misc{francobarranco2021stable,                                                                                         
      title={Stable deep neural network architectures for mitochondria segmentation on electron microscopy volumes},    
      author={Daniel Franco-Barranco and Arrate Muñoz-Barrutia and Ignacio Arganda-Carreras},                           
      year={2021},                                                                                                      
      eprint={2104.03577},                                                                                              
      archivePrefix={arXiv},                                                                                            
      primaryClass={eess.IV}                                                                                            
}                                                                                                                       
``` 
