import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from engine.base_workflow import Base_Workflow

class Classification(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        
        self.stats['test_accuracy'] = 0
        self.stats['cm'] = None

    def after_all_images(self, Y):
        self.Y = Y
        self.all_pred = np.concatenate(self.all_pred)

        # Save predictions in a csv file
        df = pd.DataFrame(self.test_filenames, columns=['Nuclei file'])
        df['pred_class'] = np.array(self.all_pred).squeeze()
        f= os.path.join(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, "..", "predictions.csv")
        os.makedirs(self.cfg.PATHS.RESULT_DIR.PER_IMAGE, exist_ok=True)
        df.to_csv(f, index=False, header=True)

        if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
            self.stats['test_accuracy'] = accuracy_score(np.argmax(Y, axis=-1), self.all_pred)
            self.stats['cm'] = confusion_matrix(np.argmax(Y, axis=-1), self.all_pred)
    
    def print_stats(self, image_counter):
        if self.cfg.DATA.TEST.LOAD_GT and self.cfg.TEST.EVALUATE:
            print('Test Accuracy: ', round((self.stats['test_accuracy'] * 100), 2), "%")
            print("Confusion matrix: ")
            print(self.stats['cm'])
            display_labels = ["Category {}".format(i) for i in range(self.cfg.MODEL.N_CLASSES)]
            print(classification_report(np.argmax(self.Y, axis=-1), self.all_pred, target_names=display_labels))

    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass
        
    def normalize_stats(self, image_counter):
        pass