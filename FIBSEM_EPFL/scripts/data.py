from tqdm import tqdm
import numpy as np
import sys
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator
import cv2


def load_data(train_path, train_mask_path, test_path, test_mask_path,
              image_shape, create_val=True, val_split=0.1,
              seedValue=42):
    """Load train, validation and test data from the given paths.

       Args:
            train_path (str): path to the training data.
            train_mask_path (str): path to the training data masks. 
            test_path (str): path to the test data.
            test_mask_path (str): path to the test data masks.
            image_shape (array of 3 int): dimensions of the images.
            create_val (bool, optional): if true validation data is
            created.
            val_split (float, optional): % of the train data used as 
            validation (value between o and 1).
            seedValue (int, optional): seed value.

       Returns:
            In case create_val == True:
                X_train (numpy array): train images.
                Y_train (numpy array): train images' mask. 
                X_val (numpy array): validation images.
                Y_val (numpy array): validation images' mask. 
                X_test (numpy array): test images.
                Y_test (numpy array): test images' mask. 
            If not:
                X_train (numpy array): train images.
                Y_train (numpy array): train images' mask.
                X_test (numpy array): test images.
                Y_test (numpy array): test images' mask.
    """

    train_ids = sorted(next(os.walk(train_path))[2])
    train_mask_ids = sorted(next(os.walk(train_mask_path))[2])
    
    test_ids = sorted(next(os.walk(test_path))[2])
    test_mask_ids = sorted(next(os.walk(test_mask_path))[2])
    
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), image_shape[0], image_shape[1], 
                        image_shape[2]), dtype=np.uint8)
    Y_train = np.zeros((len(train_mask_ids), image_shape[0], 
                        image_shape[1],image_shape[2]), dtype=np.uint8)
    
    print('Loading train images . . .')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(os.path.join(train_path, id_))
        img = np.expand_dims(img, axis=-1)
        X_train[n] = img
    
    print('Loading train masks . . .')
    for n, id_ in tqdm(enumerate(train_mask_ids),
                       total=len(train_mask_ids)):
        mask = imread(os.path.join(train_mask_path, id_))
        mask = np.expand_dims(mask, axis=-1)
        Y_train[n] = mask
    
    Y_train = Y_train/255
    
    # Get and resize test images and masks
    X_test = np.zeros((len(test_ids), image_shape[0], image_shape[1], 
                       image_shape[2]), dtype=np.uint8)
    Y_test = np.zeros((len(test_mask_ids), image_shape[0],
                       image_shape[1], image_shape[2]), dtype=np.uint8)
    
    print('Loading test images . . .')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        img = imread(os.path.join(test_path, id_))
        img = np.expand_dims(img, axis=-1)
        X_test[n] = img
    
    print('Loading test masks . . .')
    for n, id_ in tqdm(enumerate(test_mask_ids), 
                       total=len(test_mask_ids)):
        mask = imread(os.path.join(test_mask_path, id_))
        mask = np.expand_dims(mask, axis=-1)
        Y_test[n] = mask
    
    Y_test = Y_test/255
    
    if (create_val == True):    
        X_train, X_val, \
        Y_train, Y_val = train_test_split(X_train,
                                          Y_train,
                                          train_size=1-val_split,
                                          test_size=val_split,
                                          random_state=seedValue)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test 
    else:
        return X_train, Y_train, X_test, Y_test


# Custom function to rotate the images with a fixed degree
def fixed_degree_rotation(image):
    """Rotate given image with a fixed degree

       Args:
            image (img): image to be rotated.

       Returns:
            out_image (numpy array): image rotated.
    """
    img = np.array(image)

    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    out_image = cv2.warpAffine(img, M, (w, h))
    out_image = np.expand_dims(out_image, axis=-1)
    return out_image


def da_generator(X_train, Y_train, X_val, Y_val, batch_size_value, 
                 job_id, save_examples=True, aug_base_dir='aug', 
                 hflip=True, vflip=True, seedValue=42, 
                 fill_mode='reflect', preproc_function=True):
    """Makes data augmentation of the given input data.

       Args:
            X_train (numpy array): train data.
            Y_train (numpy array): train mask data.
            X_val (numpy array): validation data.
            Y_val (numpy array): validation mask data.
            batch_size_value (int): batch size.
            job_id (str): job identifier. 
            save_examples (bool, optional): if true 5 examples of DA 
            are stored.
            aug_base_dir (str): data augmentation base directory to
            store generated files.
            hflip (bool, optional): if true horizontal flips are made.
            vflip (bool, optional): if true vertical flip are made.
            seedValue (int, optional): seed value.
            fill_mode (str, optional): ImageDataGenerator of Keras fill
            mode values.
            preproc_function (bool, optional): if true preprocess 
            function to make random 180 degrees rotations are performed. 

       Returns:
            train_generator (Keras iterable of flow_from_directory): 
            train data iterator.
            val_generator (Keras iterable of flow_from_directory):
            validation data iterator.
    """
    
    if (preproc_function == True):
        data_gen_args = dict(horizontal_flip=hflip,
                             vertical_flip=vflip,
                             fill_mode=fill_mode,
                             preprocessing_function=fixed_degree_rotation)
    else:
        data_gen_args = dict(horizontal_flip=hflip,
                             vertical_flip=vflip,
                             fill_mode=fill_mode)
                             
    
    # Train data, provide the same seed and keyword arguments to 
    # the fit and flow methods
    X_datagen_train = ImageDataGenerator(**data_gen_args)
    Y_datagen_train = ImageDataGenerator(**data_gen_args)
    X_datagen_train.fit(X_train, augment=True, seed=seedValue)
    Y_datagen_train.fit(Y_train, augment=True, seed=seedValue)
    
    # Validation data, no data augmentation, but we create a generator 
    # anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_val, augment=False, seed=seedValue)
    Y_datagen_val.fit(Y_val, augment=False, seed=seedValue)
    
    # Check a few of generated images
    if (save_examples == True):
        aug_x = os.path.join('aug', job_id, 'x')
        aug_y = os.path.join('aug', job_id, 'y')
        if not os.path.exists(aug_x):          
            os.makedirs(aug_x)                 
        if not os.path.exists(aug_y):          
            os.makedirs(aug_y)
     
        i=0
        for batch in X_datagen_train.flow(X_train, save_to_dir=aug_x,\
                                          batch_size=batch_size_value,
                                          shuffle=True, seed=seedValue,
                                          save_prefix='x',
                                          save_format='jpeg'):
            i += 1
            if i > 5:
                break
        i=0
        for batch in Y_datagen_train.flow(Y_train, save_to_dir=aug_y,\
                                          batch_size=batch_size_value,
                                          shuffle=True, seed=seedValue,
                                          save_prefix='y',
                                          save_format='jpeg'):
            i += 1
            if i > 5:
                break
    
    X_train_augmented = X_datagen_train.flow(X_train, 
                                             batch_size=batch_size_value,
                                             shuffle=False,
                                             seed=seedValue)
    Y_train_augmented = Y_datagen_train.flow(Y_train, 
                                             batch_size=batch_size_value,
                                             shuffle=False,
                                             seed=seedValue)
    X_val_flow = X_datagen_val.flow(X_val, batch_size=batch_size_value,
                                    shuffle=False, seed=seedValue)
    Y_val_flow = Y_datagen_val.flow(Y_val, batch_size=batch_size_value,
                                    shuffle=False, seed=seedValue)
    
    # Combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    val_generator = zip(X_val_flow, Y_val_flow)
    
    return train_generator, val_generator


def crop_data(data, data_mask, width, height):
    """Crop data into smaller pieces

       Args:
            data (4D numpy array): data to crop.
            data_mask (4D numpy array): data masks to crop.
            width (str): output image width.
            height (str): output image height.

       Returns:
            cropped_data (4D numpy array): cropped data images. 
            cropped_data_mask (4D numpy array): cropped data masks. 
    """

    print("Cropping [" + str(data.shape[1]) + ', ' + str(data.shape[2]) 
          + "] images into [" + str(width) + ', ' + str(height) 
          + "] . . .")  

    # Calculate the number of images to be generated
    h_num = int(data.shape[1] / width) + (data.shape[1] % width > 0)
    v_num = int(data.shape[2] / height) + (data.shape[2] % height > 0)
    total_cropped = data.shape[0]*h_num*v_num

    # Crop data
    cropped_data = np.zeros((total_cropped, width, height, 
                            data.shape[3]), dtype=np.uint8)
    cont=0
    for img_num in range(0, data.shape[0]):
        for i in range(0, h_num):
            for j in range(0, v_num):
                cropped_data[cont]= data[img_num,
                                         (i*width):((i+1)*height),
                                         (j*width):((j+1)*height)]
                cont=cont+1

    # Crop mask data
    cropped_data_mask = np.zeros((total_cropped, width, height, 
                            data.shape[3]), dtype=np.uint8)
    cont=0
    for img_num in range(0, data.shape[0]):
        for i in range(0, h_num):
            for j in range(0, v_num):
                cropped_data_mask[cont]= data_mask[img_num,
                                                  (i*width):((i+1)*height),
                                                  (j*width):((j+1)*height)]
                cont=cont+1

    return cropped_data, cropped_data_mask
            
