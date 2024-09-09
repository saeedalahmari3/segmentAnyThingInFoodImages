import os 
import sys 
import cv2 
import numpy as np


class overlay():
    def __init__(self,path2Images, path2Masks, SaveDir='-overlay',alpha=0.4):
        self.alpha = alpha
        self.path2Images = path2Images
        self.path2Masks = path2Masks
        self.path2SaveTo = self.path2Masks + '-overlay'
        self.image_names = [item for item in os.listdir(path2Images) if not item.startswith('.')]
        
    def make_overlays(self):
        # adopted from https://stackoverflow.com/questions/63177320/apply-lightly-transparent-mask-on-top-of-image-with-known-co-ordinates
        # read image
        for image_name in self.image_names:
            mask_name = image_name.replace('.jpg','.png')
            mask_name = mask_name.replace('image','label')
            img = cv2.imread(os.path.join(self.path2Images,image_name),-1)
            # create cyan image
            cyan = np.full_like(img,(255,255,0)) # Cyan (255,255,0), Pink (248,177,149)
            # add cyan to img and save as new image
            img_cyan = cv2.addWeighted(img, self.alpha, cyan, 1-self.alpha, 0)
            # create black image for mask base
            mask_1c = cv2.imread(os.path.join(self.path2Masks,mask_name),-1)
            mask = self.duplicate_mask(mask_1c)
            result = np.where(mask==255, img_cyan, img)
            self.save_overlay(result,image_name)

    def duplicate_mask(self,mask):
        if len(mask.shape) < 3:
            mask_3c = np.zeros((mask.shape[0],mask.shape[1],3))
            mask_3c[:,:,0] = mask 
            mask_3c[:,:,1] = mask
            mask_3c[:,:,2] = mask
            return mask_3c
        else:
            return mask
    def save_overlay(self,results,image_name):
        if not os.path.exists(self.path2SaveTo):
            os.makedirs(self.path2SaveTo)
        cv2.imwrite(os.path.join(self.path2SaveTo,image_name),results)


path2Images = '/Volumes/WD Element/Collaboration/FoodSegmentationComparison/ComboDatasetResults/data/test_images_labels/images'
path2Masks = '/Volumes/WD Element/Collaboration/FoodSegmentationComparison/ComboDatasetResults/data/test_images_labels/labels'

overlay_obj = overlay(path2Images, path2Masks)
overlay_obj.make_overlays()