import pickle
import numpy as np
import cv2
import os
from collections import namedtuple

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is based on the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,      0,   'void'            , 0       , False        , True        , (  0,  0,  0) ),
    Label(  'blue zone'            ,  1 ,      1  , 'fire'            , 1       , False        , False       , (  120,  28,  109) ),
    Label(  'middle zone'          ,  2 ,      2  , 'fire'            , 1       , False        , False       , (  237,  0,  0) ),
    Label(  'front zone'           ,  3 ,      3  , 'fire'            , 1       , False        , False       , (  252,  255,  164) ),
]

# create a function which maps id to trainId:
id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

############## EXTRA FOR RGB IMAGES
#Function to change rgb to index
color2index = {
    (0,0,0) : 0,
    (64,64,64) : 1,
    (136,136,136) : 2,
    (243,243,243) : 3
}

def rgb2mask(img):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])]
        except:
            print("Color {} Not Found".format(tuple(img[img_id==c][0])))
            pass
    return mask.astype('uint8')
############## END OF EXTRA FUNCTION

data_path = "/root/deeplearning/data/jetfire"
meta_path = "/root/deeplearning/data/jetfire/meta"

if not os.path.exists(meta_path):
    os.makedirs(meta_path)
if not os.path.exists(meta_path + "/label_imgs"):
    os.makedirs(meta_path + "/label_imgs")

################################################################################
# convert all labels to label imgs with trainId pixel values (and save to disk):
################################################################################
train_label_img_paths = []

#img_dir = data_path + "/img/train/"
label_dir = data_path + "/gt/train/"

file_names = os.listdir(label_dir)
for file_name in file_names:
    gtFine_img_path = label_dir + file_name
    gtFine_img = cv2.imread(gtFine_img_path, 1) # (shape: (480, 640))
    gtFine_img = rgb2mask(gtFine_img)
    label_img = id_to_trainId_map_func(gtFine_img) # (shape: (480, 640))
    label_img = label_img.astype(np.uint8)

    cv2.imwrite(meta_path + "/label_imgs/" + file_name, label_img)
    train_label_img_paths.append(meta_path + "/label_imgs/" + file_name)

#img_dir = data_path + "/img/val/"
label_dir = data_path + "/gt/val/"

file_names = os.listdir(label_dir)
for file_name in file_names:
    gtFine_img_path = label_dir + file_name
    gtFine_img = cv2.imread(gtFine_img_path, 1) # (shape: (480, 640))
    gtFine_img = rgb2mask(gtFine_img)
    label_img = id_to_trainId_map_func(gtFine_img) # (shape: (480, 640))
    label_img = label_img.astype(np.uint8)

    cv2.imwrite(meta_path + "/label_imgs/" + file_name, label_img)

################################################################################
# compute the class weigths:
################################################################################
print ("computing class weights")

num_classes = 4

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

# get the total number of pixels in all train label_imgs that are of each object class:
for step, label_img_path in enumerate(train_label_img_paths):
    if step % 100 == 0:
        print (step)

    label_img = cv2.imread(label_img_path, -1)

    for trainId in range(num_classes):
        # count how many pixels in label_img which are of object class trainId:
        trainId_mask = np.equal(label_img, trainId)
        trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += trainId_count

# compute the class weights according to the ENet paper:
class_weights = []
total_count = sum(trainId_to_count.values())
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

print (class_weights)

with open(meta_path + "/class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
