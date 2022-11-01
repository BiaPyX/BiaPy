Select workflow
---------------

Depending on the workflow the input and the ground truth varies as is described below:

* `Semantic segmentation <../workflows/semantic_segmentation.html>`_, the input is an image where the area/object in hand is present whereas the ground truth is another image, of the same shape as the input, with a label assigned to each pixel.  
* `Instance segmentation <../workflows/instance_segmentation.html>`_, same as semantic segmentation but the ground truth covers not only the label of each pixel but an unique identifier of that object.  
* `Detection <../workflows/detection.html>`_, when you do not need to have a pixel-level accuracy output, but still recognize objects in images, this workflow is your choice. The objects in this workflow are denoted by points in their center because the important thing here is to detect them, know where their are and how many of them. The input here is an image and the ground truth a CSV file with the coordinates of each object center point.   
* `Denoising <../workflows/denoising.html>`_, the purpose of this workflow is to remove noise from a given input. As a clean version of the same image sometimes is not factible, or even not possible as in biomedical imaging field, we incorporate a workflow based in Noise2Void which only requires a noisy input image to be trained. 
* `Super-resolution <../workflows/super_resolution.html>`_, aims at reconstructing high-resolution (HR) images from low-resolution (LR) ones. Normally, the HR image shape is ×2, ×3 or ×4 larger than the LR image. The input pair here is an LR image with and it HR version as ground truth. 
* `Self-supervision <../workflows/self_supervision.html>`_, refers to those techniques based on training a DL model without labels. For this, the model is intended to solve a so-called pretext task so it can learn image features. Afterwards, the model can be retrained, or finetunned in the DL jargon, in a labeled workflow. Thus, since the model has been trained before, and does not start from scratch, the learning process converges faster. The input in this workflow is just an image as there is no need to provide the ground truth. 
* `Classification <../workflows/classification.html>`_, aims to match a given input image to its corresponding class (ground truth).

[image_workflow under construction]
