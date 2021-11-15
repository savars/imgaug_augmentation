from matplotlib import pyplot as plt
import os 
import cv2
from PIL import Image
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


# function from darknet :)

def bbox2points(my_image, bbox):
    hh,ww,dd = my_image.shape
    x,y,w,h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return [int(round(xmin*ww)), int(round(ymin*hh)), int(round(xmax*ww)), int(round(ymax*hh))]

def bboxClass2List(augmented_bbox, bboxes):
    aug_list_of_lists = []
    
    for k in range(0,len(bboxes)):
        aug_bbox_list = augmented_bbox[k][0:3].tolist()[0]+augmented_bbox[k][0:3].tolist()[1]
        aug_list_of_lists.append(aug_bbox_list)
        
    return aug_list_of_lists
    
def toYolo(bboxList, image):
    yolo_bboxes = []
    list_yolo_bboxes = []
    ih, iw, ic = image.shape
    for k in range(0, len(bboxList)):
        
        xc = .5*(bboxList[k][2] + bboxList[k][0])/iw
        yolo_bboxes.append(xc)
        yc = .5*(bboxList[k][3] + bboxList[k][1])/ih
        yolo_bboxes.append(yc)
        wid = (bboxList[k][2] - bboxList[k][0])/iw
        yolo_bboxes.append(wid)
        hei = (bboxList[k][3] - bboxList[k][1])/ih
        yolo_bboxes.append(hei)
        list_yolo_bboxes.append(yolo_bboxes)
        yolo_bboxes = []
    return list_yolo_bboxes


count = 0
bboxes = []
bbox = []

seq = iaa.Sequential([
    iaa.Affine(rotate=(-3, 3)),
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    #iaa.Affine(rotate=(0, 15))
    #iaa.SigmoidContrast(gain=(3, 15), cutoff=(0.4, 0.75)),
    #iaa.Fliplr(1)
    #iaa.AllChannelsCLAHE(),
    #iaa.imgcorruptlike.GlassBlur(severity=2),
])



path = 'path to image AND label folders' # set path to the useful folders
path_to_labels = path + 'original-lbs'
path_to_images = path + 'original-ims'


image_name_list = os.listdir(path_to_images) # list of image names in the folder, will be used later

# for every element in the labels folder, read a corresponding image from image folder (by accessing image list).
# the way they are named (img1.jpg,img1.txt) ensures this works properly.

for element in os.listdir(path_to_labels):
    text_file = path_to_labels + '/' + element      # acces text file for open later
    image = cv2.imread(path_to_images + '/' + image_name_list[count]) # read corresponding image

# read the bounding box text file, parse it, and assign every line of the file to a list. 
# note  that in my method, I had to cast the values extracted to float

    with open(text_file, 'r') as file_input:
        for line in file_input:
            box = line.split(" ")
            l = float(box[1])
            bbox.append(l)
            r = float(box[2])
            bbox.append(r)
            t = float(box[3])
            bbox.append(t)
            b = float(box[4])
            bbox.append(b)
            bboxes.append(bbox) # create a list of lists, which is an input to augmenting function
            bbox = [] # reset the bbox
            
        upscaled_bboxes = []
        for box in bboxes:
            upscaled = bbox2points(image,box)
            xx1,yy1,xx2,yy2 = upscaled
            upscaled_conv = ia.BoundingBox(xx1,yy1,xx2,yy2)
            upscaled_bboxes.append(upscaled_conv)
            
        bbs = BoundingBoxesOnImage(upscaled_bboxes, shape = image.shape).remove_out_of_image('partly')
        # apply augmenting
        img_aug, bbs_aug = seq(image = image, bounding_boxes = bbs)

        bbs_aug_conv = bboxClass2List(bbs_aug, bboxes)
        bbs_aug_conv_lol = toYolo(bbs_aug_conv,image)
        
        # reset bboxes
        bboxes = []
        upscaled_bboxes = [] 
        
        output_image = Image.fromarray(img_aug)
        #output_image.show()
        
        
        
        
        # save the image
        output_image = Image.fromarray(img_aug)
        output_image.save(path + 'aug-outputs/' + image_name_list[count] + '-aug' + '.jpg')
        
        # write to the text files the augmented bboxes
        with open(path + 'aug-outputs/' + image_name_list[count] + '-aug' + '.txt', 'w') as outfile:
            
            # go through every output of bbaug, round it to 6 d-figs (as in original yolo) and write to textfile
            for i in bbs_aug_conv_lol:  
                a_list = list(i)
                a_list = list(map(float, a_list))
                rounded = ['{:.6f}'.format(num) for num in a_list]
                #rounded.remove('0.000000')
                rounded.insert(0,'0')
                str1 = ' '.join(rounded)
                outfile.write(str1 + '\n')
                
    count = count+1 # increment counter for the next image

