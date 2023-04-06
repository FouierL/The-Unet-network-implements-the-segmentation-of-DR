import cv2
import os
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.transform as trans
from skimage import io
from tensorflow.python.keras import backend as K
import math


def trainGenerator(batch_size,
                   aug_dict,
                   train_path,
                   image_folder,
                   label_folder,
                   image_color_mode="rgb",
                   label_color_mode="grayscale",
                   image_save_prefix="image",
                   label_save_prefix="label",
                   flag_multi_class=True,
                   shuffle=True,
                   num_class=5,
                   save_to_dir=None,
                   target_size=(512, 512),
                   seed=1):
    '''
    can generate image and label at the same time
    use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        shuffle=shuffle,
        color_mode=image_color_mode,
        target_size=target_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        batch_size=batch_size,
        seed=seed)
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        shuffle=shuffle,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, label_generator)

    def adjustData(img, label, flag_multi_class, num_class):
        if (flag_multi_class):
            img = img / 255
            label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
            new_label = np.zeros(label.shape + (num_class,))
            for i in range(num_class):
                new_label[label == i, i] = 1
            label = new_label
        elif (np.max(img) > 1):
            img = img / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return (img, label)

    for (img, label) in train_generator:
        img, label = adjustData(img, label, flag_multi_class, num_class)
        yield (img, label)


def testGenerator(test_path, target_size=(512, 512), as_gray=False):
    filelist = os.listdir(test_path)
    for filename in filelist:
        img = io.imread(os.path.join(test_path, filename), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size, mode='constant')
        if as_gray:
            img = np.reshape(img, img.shape + (1,))
        else:
            img = img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult(save_path, test_path, target_size, npyfile, classes=5):
    bg = [0, 0, 0]
    EX = [255, 0, 0]
    HE = [0, 255, 0]
    MA = [0, 0, 255]
    SE = [255, 255, 0]
    clo = np.array([bg, EX, HE, MA, SE])
    COLOR_DICT = clo
    def draw(img_origin, img_mask, bgr):
        img_origin[np.where(img_mask > 0)] = bgr
        return img_origin
    filelist_test = os.listdir(test_path)
    name = []
    for filename in filelist_test:
        (realname, extension) = os.path.splitext(filename)
        name.append(realname)
    for i, item in enumerate(npyfile):
        cd = []
        if classes == 5:
            img = item
            img_out = np.zeros(img[:, :, 0].shape + (3,))
            img_out_EX = img_out.copy()
            img_out_HE = img_out.copy()
            img_out_MA = img_out.copy()
            img_out_SE = img_out.copy()
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    index_of_class = np.argmax(img[row, col])
                    img_out[row, col] = COLOR_DICT[index_of_class]
                    if index_of_class == 1:
                        img_out_EX[row, col] = [255, 255, 255]
                        img_out_EX = img_out_EX.astype(np.uint8)
                    elif index_of_class == 2:
                        img_out_HE[row, col] = [255, 255, 255]
                        img_out_HE = img_out_HE.astype(np.uint8)
                    elif index_of_class == 3:
                        img_out_MA[row, col] = [255, 255, 255]
                        img_out_MA = img_out_MA.astype(np.uint8)
                    elif index_of_class == 4:
                        img_out_SE[row, col] = [255, 255, 255]
                        img_out_SE = img_out_SE.astype(np.uint8)
            img_out_EX = cv2.resize(img_out_EX, target_size, cv2.INTER_NEAREST)
            img_out_HE = cv2.resize(img_out_HE, target_size, cv2.INTER_NEAREST)
            img_out_MA = cv2.resize(img_out_MA, target_size, cv2.INTER_NEAREST)
            img_out_SE = cv2.resize(img_out_SE, target_size, cv2.INTER_NEAREST)
            cd.append(img_out_EX)
            cd.append(img_out_HE)
            cd.append(img_out_MA)
            cd.append(img_out_SE)
            img_test = io.imread(os.path.join(test_path, '%s.jpg' % name[i]))
            img_test = cv2.resize(img_test, target_size)
            for j, s in enumerate(['EX', 'HE', 'MA', 'SE']):
                path = os.path.join(save_path, s + '_predict')
                if not os.path.exists(path):
                    os.makedirs(path)
                io.imsave(os.path.join(path, '%s_predict.png' % (s + '_' + name[i])), cd[j])
                img_test = draw(img_test, cd[j][:,:,0], COLOR_DICT[j+1])
            io.imsave(os.path.join(save_path, '%s_predict.png' % name[i]), img_test)