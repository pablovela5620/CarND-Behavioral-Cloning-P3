import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

CSV_PATH = "data/driving_log.csv"
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
BATCH_SIZE = 256

np.random.seed(0)


def load_img(img_path):
    '''
    :param img_path:Path to image file 
    :return: image as np.array
    '''
    img = cv2.imread('data/' + img_path.strip())
    return img


def crop_img(img):
    '''
    :param img: image to be cropped
    removes top portion with unnecessary pixels (background) and bottom portion with car in the image
    :return: cropped image
    '''
    img_crop = img[60:135, :, :]
    return img_crop


def img_resize(img):
    '''
    :param img: img to be resized
    :return:resized image (66,200,3) 
    '''
    resized_img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized_img


def img_flip(img, angle):
    '''
    randomly decides to flip image to augment data, angle is accordingly modified with image flip
    '''
    rand_int = np.random.randint(0, 2)
    if rand_int == 0:
        img = np.fliplr(img)
        angle = -angle
        return img, angle
    else:
        return img, angle


def img_change_brightness(img):
    '''
    Random brightness changes 
    '''
    # HSV (Hue, Saturation, Value)
    convt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = 0.25 + np.random.uniform()
    # Value is the brightness of the image
    convt_img[:, :, 2] = convt_img[:, :, 2] * brightness
    # Converted back to rgb
    new_img = cv2.cvtColor(convt_img, cv2.COLOR_HSV2RGB)
    return new_img


# Not currently implemented
# def img_translate(img, angle, range_x=100, range_y=10):
#     trans_x = range_x * np.random.uniform() - (range_x / 2)
#     trans_y = range_y * np.random.uniform() - (range_y / 2)
#     angle += trans_x / range_x * 2 * .2
#     trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
#     new_img = cv2.warpAffine(img, trans_m, (IMAGE_WIDTH, IMAGE_HEIGHT), borderMode=cv2.BORDER_REPLICATE)
#     return new_img, angle
# def low_angle(X_sample, y_sample):
#     angle = y_sample
#     img = load_img(X_sample[0])
#     probability = np.random.uniform()
#     if (abs(angle) < 0.2 and probability < 0.7):
#         camera = np.random.randint(1, 3)
#         if camera == 1:
#             img = load_img(X_sample[1])  # Left Image
#             angle = y_sample + 0.25
#             return img, angle
#         elif camera == 2:
#             img = load_img(X_sample[2])  # Right Image
#             angle = y_sample - 0.25
#             return img, angle
#     return img, angle

def img_choice(X_sample, y_sample):
    '''
    :param X_sample: 
    :param y_sample: 
    :return: 
    '''
    rand_int = np.random.randint(0, 3)
    if rand_int == 0:
        img = load_img(X_sample[0])  # Center Image
        angle = y_sample
        return img, angle
    elif rand_int == 1:
        img = load_img(X_sample[1])  # Left Image
        angle = y_sample + 0.25
        return img, angle
    else:
        img = load_img(X_sample[2])  # Right Image
        angle = y_sample - 0.25
        return img, angle





def preprocess_img(img):
    '''
    Function Crops and Resizes Image
    :param img: image to be pre processed
    :return: preprocessed image
    '''
    # Crops image to remove the top half where there are trees
    img = crop_img(img)

    # Resize image to be fed into network
    img = img_resize(img)

    return img


def data_augment(X_sample, y_sample):
    img, angle = img_choice(X_sample, y_sample)
    if abs(angle < 0.1):
        return None, None
    img, angle = img_flip(img, angle)
    img = img_change_brightness(img)
    img = preprocess_img(img)
    # img = img_translate(img, angle)
    return img, angle


def generator(X_data, y_data, batch_size=BATCH_SIZE):
    num_samples = len(y_data)
    i = 0
    while (True):
        for offset in range(0, num_samples, batch_size):
            X_samples = X_data[offset:offset + batch_size]
            y_samples = y_data[offset:offset + batch_size]

            images = []
            steering_angles = []
            for X_sample, y_sample in zip(X_samples, y_samples):
                img, angle = data_augment(X_sample, y_sample)
                if img is not None:
                    images.append(img)
                    steering_angles.append(angle)
                    i = i + 1

            X_train = np.asarray(images)
            y_train = np.asarray(steering_angles)
            yield shuffle(X_train, y_train, random_state=0)


def valid_generator(X_data, y_data, batch_size=32):
    num_samples = len(y_data)
    while (True):
        for offset in range(0, num_samples, batch_size):
            X_samples = X_data[offset:offset + batch_size]
            y_samples = y_data[offset:offset + batch_size]

            images = []
            steering_angles = []
            for X_sample, y_sample in zip(X_samples, y_samples):
                image, angle = load_img(X_sample[0]), y_sample
                image = preprocess_img(image)
                images.append(image)
                steering_angles.append(angle)

            X_train = np.asarray(images)
            y_train = np.asarray(steering_angles)
            yield shuffle(X_train, y_train, random_state=0)
