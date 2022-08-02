from engine.base_workflow import Base_Workflow

class Semantic_Segmentation(Base_Workflow):
    def __init__(self, cfg, model, post_processing=False):
        super().__init__(cfg, model, post_processing)
        
    def after_merge_patches(self, pred, Y, filenames):
        pass

    def after_full_image(self, pred, Y, filenames):
        pass

    def after_all_images(self, Y):
        super().after_all_images(None)

    def normalize_stats(self, image_counter):
        super().normalize_stats(image_counter)

    def print_stats(self, image_counter):
        super().print_stats(image_counter)
        super().print_post_processing_stats()


        