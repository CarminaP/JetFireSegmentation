import torch
import torch.utils.data
import numpy as np
import cv2
import os

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data_path, cityscapes_meta_path):
        self.img_dir = data_path + "/img/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 480
        self.img_w = 640

        self.new_img_h = 240
        self.new_img_w = 320

        self.examples = []
        ## change for single source
        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]
            img_path = self.img_dir + file_name

            label_img_path = self.label_dir + img_id + ".png"

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, 1) # (shape: (480, 640, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (240, 320, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (480, 640))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (240, 320))

        # flip the img and the label with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            label_img = cv2.flip(label_img, 1)

        ########################################################################
        # randomly scale the img and the label:
        ########################################################################
        scale = np.random.uniform(low=0.7, high=2.0)
        new_img_h = int(scale*self.new_img_h)
        new_img_w = int(scale*self.new_img_w)

        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (new_img_w, new_img_h),
                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))

        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))
        ########################################################################

        ########################################################################
        # select a 256x256 random crop from the img and label:
        ########################################################################
        start_x = np.random.randint(low=0, high=(new_img_w - 120))
        end_x = start_x + 120
        start_y = np.random.randint(low=0, high=(new_img_h - 120))
        end_y = start_y + 120

        img = img[start_y:end_y, start_x:end_x] # (shape: (120, 120, 3))
        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (120, 120))
        ########################################################################

        # normalize the img (with the mean and std):
        img = img/255.0
        #Data normalization is done by subtracting the mean from each pixel
        img = img - np.array([0.06254871, 0.09143332, 0.53944639])
        #and then dividing the result by the standard deviation
        img = img/np.array([0.22379429, 0.25679154, 0.19004982]) # (shape: (120, 120, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 120, 120))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 120, 120))
        label_img = torch.from_numpy(label_img) # (shape: (120, 120))

        return (img, label_img)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, data_path, cityscapes_meta_path):
        self.img_dir = data_path + "/img/val/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"

        self.img_h = 480
        self.img_w = 640

        self.new_img_h = 240
        self.new_img_w = 320

        self.examples = []
        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]
            img_path = self.img_dir + file_name

            label_img_path = self.label_dir + img_id + ".png"

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, 1)
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (1024, 2048))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST)

        # normalize the img (with the mean and std):
        img = img/255.0
        #Data normalization is done by subtracting the mean from each pixel
        img = img - np.array([0.06254871, 0.09143332, 0.53944639])
        #and then dividing the result by the standard deviation
        img = img/np.array([0.22379429, 0.25679154, 0.19004982])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)
        label_img = torch.from_numpy(label_img)

        return (img, label_img, img_id)

    def __len__(self):
        return self.num_examples
        

class DatasetEval(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.img_dir = data_path + "/eval/"

        self.img_h = 480
        self.img_w = 640

        self.new_img_h = 240
        self.new_img_w = 320

        self.examples = []
        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]
            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, 1)
        # resize img without interpolation (want the image to still contain
        # the original pixel values):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)

        # normalize the img (with the mean and std):
        img = img/255.0
        #Data normalization is done by subtracting the mean from each pixel
        img = img - np.array([0.06254871, 0.09143332, 0.53944639])
        #and then dividing the result by the standard deviation
        img = img/np.array([0.22379429, 0.25679154, 0.19004982])
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)

        return (img, img_id)

    def __len__(self):
        return self.num_examples