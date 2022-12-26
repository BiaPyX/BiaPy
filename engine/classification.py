import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from engine.base_workflow import Base_Workflow

class Classification(Base_Workflow):
    def __init__(self, cfg, model, class_names=None, post_processing=False):
        super().__init__(cfg, model, post_processing)
        
        self.stats['test_accuracy'] = 0
        self.stats['cm'] = None
        self.all_pred = [] 
        if self.cfg.DATA.TEST.LOAD_GT: 
            self.all_gt = []
        self.test_filenames = []
        self.class_names = class_names

    def process_sample(self, X, Y, filenames, norm):   
        self.test_filenames.append(filenames)   

        self.stats['patch_counter'] += X.shape[0]

        # Predict each patch
        l = int(math.ceil(X.shape[0]/self.cfg.TRAIN.BATCH_SIZE))
        for k in tqdm(range(l), leave=False):
            top = (k+1)*self.cfg.TRAIN.BATCH_SIZE if (k+1)*self.cfg.TRAIN.BATCH_SIZE < X.shape[0] else X.shape[0]
            p = self.model.predict(X[k*self.cfg.TRAIN.BATCH_SIZE:top], verbose=0)
            p = np.argmax(p, axis=1)
            self.all_pred.append(p)

        if self.cfg.DATA.TEST.LOAD_GT: 
            self.all_gt.append(Y)

    def after_all_images(self):
        # Save predictions in a csv file
        df = pd.DataFrame(self.test_filenames, columns=['filename'])
        df['class'] = np.array(self.all_pred)
        f= os.path.join(self.cfg.PATHS.RESULT_DIR.PATH, "predictions.csv")
        os.makedirs(self.cfg.PATHS.RESULT_DIR.PATH, exist_ok=True)
        df.to_csv(f, index=False, header=True)

        if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
            self.stats['test_accuracy'] = accuracy_score(self.all_gt, self.all_pred)
            self.stats['cm'] = confusion_matrix(self.all_gt, self.all_pred)
    
    def print_stats(self, image_counter):
        if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
            print('Test Accuracy: ', round((self.stats['test_accuracy'] * 100), 2), "%")
            print("Confusion matrix: ")
            print(self.stats['cm'])
            if self.class_names is not None:
                display_labels = ["Category {} ({})".format(i, self.class_names[i]) for i in range(self.cfg.MODEL.N_CLASSES)]
            else:
                display_labels = ["Category {}".format(i) for i in range(self.cfg.MODEL.N_CLASSES)]
            print(classification_report(self.all_gt, self.all_pred, target_names=display_labels))

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass
        
    def normalize_stats(self, image_counter):
        pass