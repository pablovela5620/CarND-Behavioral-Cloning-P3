**Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg
[image2]: ./examples/left_2016_12_01_13_30_48_287.jpg
[image3]: ./examples/right_2016_12_01_13_30_48_287.jpg
[image4]: ./examples/histrograph.png
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* helper_functions.py for data augmentation and preprocessing
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The helper_functions.py file contains the code for data augmentation and preprocessing as well as the generator

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a deep convolution neural network based on NVIDIA's end to end Deep learning model

The model includes ELU activation function to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was first to start with a very simple network which was just a single flatten layer
to a fully connected layer with a single output. This was done to ensure that the model could properly interact with the simulator and drive the car

Once I had managed to get the simulator working and the model trained I moved on to a much more powerful network, one that I based of
NVIDIA's end to end Deep learning for self driving cars.

Once NVIDIA's architecture was implemented, I modified the model slightly by adding in a lambda layer in the beginning of the model to normalize the images
during training. I also added dropout layers to avoid overfitting.

The data that was collected was split into training and validation data to gauge the models performance.


The final step was to run the simulator to see how well the car was driving around track one. One major issue was the car would fall into the lake after the first turn past the red markings.
To improve the driving behavior in these cases, I completely removed some of the data under a certain steering angle threshold

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture can be seen below

| Layer (type)            |Output Shape      |Connected to     |
|-------------------------|------------------|-----------------|
|lambda_1 (Lambda)        |(None, 66, 200, 3)|input            |
|conv2d_1 (Conv2D)        |(None, 31, 98, 24)|lambda_1         |
|conv2d_2 (Conv2D)        |(None, 14, 47, 36)|conv2d_1         |
|conv2d_3 (Conv2D)        |(None, 5, 22, 48) |conv2d_2         |
|conv2d_4 (Conv2D)        |(None, 3, 20, 64) |conv2d_3         |
|conv2d_5 (Conv2D)        |(None, 1, 18, 64) |conv2d_4         |
|flatten_1 (Flatten)      |(None, 1152)      |conv2d_5         |
|dense_1 (Dense)          |(None, 100)       |flatten_1        |
|dropout_1 (Dropout)      |(None, 100)       |dense_1          |
|dense_2 (Dense)          |(None, 50)        |dropout_1        |
|dropout_2 (Dropout)      |(None, 50)        |dense_2          |
|dense_3 (Dense)          |(None, 10)        |dropout_2        |
|dropout_3 (Dropout)      |(None, 10)        |dense_3          |
|dense_4 (Dense)          |(None, 1)         |dropout_3        |



#### 3. Creation of the Training Set & Training Process

At the start of the project recording my own data caused alot of struggles so I decided to use the Udacity data set that was released.
Below you can see examples of the left, center, and right driving images:


![Left Image][image2] ![Center Image][image1] ![Right Image][image3]

Using these images, I performed augmentations to help the model generalize.

First I began with a simply randomly choosing between the left, center and right images

```python
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
```

Because the left and right images are slightly off-center their steering angles had to be slightly adjusted by a static constant of 0.25

Following this, a histogram of the steering angles was made to try and see what kind of data is being used.

![alt text][image4]

From the above image it can be seen that the data is biased towards a zero steering angle. Due to this I decided to use a threshold angle and completely ignore any data below this angle

```python
if abs(angle < 0.1):
    return None, None
```

This helped the model during turns on the track.

I continued by randomly flipping the images and adjusting their respective steering angle:

```python
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
```


The last augmentation I performed on the data set involved randomly changing the brightness in the image.
This is done to help generalize the model for data in differing lighting conditions

```python
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
```


Once all the data had been augmented, I proceeded to pre process the data by cropping the top and bottom to remove
unnecessary pixels that only show the track background and the car. After cropping the image is resized to (66,200,3) as done in the NVIDIA model



I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by increasing
 validation loss at higher epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
