import os
import sys
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from scipy import ndimage

code_dir = "/home/user/BiaPy" # git clone git@github.com:BiaPyX/BiaPy.git
input_dir = "/home/user/datasets/lightmycells/src" # where all "Studies" folders of lightmycells challenge reside
organelle = "Actin" # Possible options: ['Nucleus', 'Mitochondria', 'Actin' , 'Tubulin']
out_dir = "/home/user/datasets/lightmycells/BiaPy_data_structure/"+organelle


sys.path.insert(0, code_dir)
from biapy.utils.util import save_tif

folders = sorted(next(os.walk(input_dir))[1])
for i, fol in tqdm(enumerate(folders), total=len(folders)):
    f = os.path.join(input_dir,fol)
    print(f"Analising {f} ...")

    images = sorted(next(os.walk(f))[2])

    new_images = []
    new_type_images = []
    for j, im in enumerate(images):
        if organelle in im:
            new_images.append(im)
        elif im and 'Tubulin' not in im and 'Nucleus' not in im and 'Mitochondria' not in im and 'Actin' not in im:
            new_type_images.append(im)

    if len(new_type_images) > 0:      
        for n, id_ in tqdm(enumerate(new_images), total=len(new_images)):
            img_folder = os.path.join(input_dir, fol)

            img = imread(os.path.join(img_folder, id_))
            img = np.squeeze(img)
            if len(img.shape) == 2: img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            
            pattern = "_".join(id_.split('_')[:-1])
            images_related = [x for x in new_type_images if pattern in x and 'Tubulin' not in x and 'Nucleus' not in x and 'Mitochondria' not in x and 'Actin' not in x]
            image_modality = images_related[0].split('_')[2]

            save_tif(img, os.path.join(out_dir,"y", f"{fol}_"+image_modality+"_"+id_), [f"{fol}_"+image_modality+"_"+id_], 
                verbose=False)

            for j, id_2 in enumerate(images_related):
                img = imread(os.path.join(img_folder, id_2))
                img = np.squeeze(img)
                if len(img.shape) == 2: img = np.expand_dims(img, axis=-1)
                img = np.expand_dims(img, axis=0)
                save_tif(img, os.path.join(out_dir,"x", f"{fol}_"+image_modality+"_"+id_), [f"{fol}_"+image_modality+"_"+id_2], verbose=False)

print("Finished!")

