#Python script for getting segmentation metrics 

import os 
import sys 
import cv2
import numpy as np 
import pandas as pd
path2Masks = '/Volumes/WD Element/Collaboration/FoodSegmentation_LORA_SAM/results_food_20epochs/'
path2Labels = '/Volumes/WD Element/Collaboration/FoodSegmentationComparison/ComboDatasetResults/Labels/Cropped'

def get_iou(mask,label):
    mask = np.matrix.flatten(mask)
    label = np.matrix.flatten(label)
    #iou = np.sum(mask * label)/ float(np.sum(mask + label))
    intersection = mask * label
    summation = (mask + label) > 0
    iou = intersection.sum() / (float(summation.sum()) + 0.0001)
    return iou

def get_accuracy(mask, label):
    mask = np.matrix.flatten(mask)
    label = np.matrix.flatten(label)
    accuracy = np.sum(mask == label)/ float(mask.size)
    return accuracy

def get_dice(mask,label):
    intersection = np.sum(mask * label)
    summation = np.sum(mask) + np.sum(label)
    return 2 * intersection / (float(summation) + 0.0001)

list_of_accuracy = []
list_of_iou = []
list_of_dice = []
list_of_names = []
average_iou = 0
average_accuracy = 0
average_dice = 0
for image_name in os.listdir(path2Masks):
    if image_name.startswith('.') or not image_name.startswith('image'):
        continue
    groundtruth_name = image_name.replace('imagecropped_','labelcropped_')
    groundtruth_name = groundtruth_name.replace('.jpg','.png')
    pred_mask = cv2.imread(os.path.join(path2Masks, image_name),0)
    gt_mask = cv2.imread(os.path.join(path2Labels, groundtruth_name),0)
    pred_mask = (pred_mask > 0) * 1
    gt_mask = (gt_mask > 0) * 1
    #gt_mask = pred_mask
    # computing IoU matrix for masks 
    iou = get_iou(pred_mask,gt_mask)
    accuracy = get_accuracy(pred_mask,gt_mask)
    dice = get_dice(pred_mask,gt_mask)
    average_iou = average_iou + iou 
    average_accuracy = average_accuracy + accuracy
    average_dice = average_dice + dice
    list_of_accuracy.append(accuracy)
    list_of_iou.append(iou)
    list_of_dice.append(dice)
    list_of_names.append(image_name)
average_accuracy = average_accuracy / float(len(list_of_accuracy))
average_iou = average_iou / float(len(list_of_iou))
average_dice = average_dice / float(len(list_of_dice))
print(average_accuracy, average_iou, average_dice)
df = pd.DataFrame()
df['Name'] = list_of_names
df['Accuracy'] = list_of_accuracy
df['IoU'] = list_of_iou
df['Dice'] = list_of_dice 

df.to_csv(os.path.basename(path2Masks)+'.csv',index=False)