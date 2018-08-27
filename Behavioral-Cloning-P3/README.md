# **Behavioral Cloning** 
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model.PNG "Model Visualization"
[image2]: ./examples/center_2018_04_29_14_53_03_503.jpg "Center"
[image3]: ./examples/center_2018_04_29_14_54_20_130.jpg "Center right"
[image4]: ./examples/center_2018_04_29_14_55_41_849.jpg "Center left"
[image5]: ./examples/left_2018_04_29_14_55_40_509.jpg "Left camera"
[image6]: ./examples/right_2018_04_29_14_56_10_389.jpg "Right camera"
[image7]: ./examples/Training.PNG "Training"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
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

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

To have any idea to start this project, [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidiais a great place to start with consists of 9 layers, including normalization layer, 5 convolutional layers and 3 fully connected layers. 
My model will be based on the Nvidiais's network.

The model uses Cropping2D to crop the original image by 50 on the top and 20 on the bottom,
includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as following:

My first step was to use a convolution neural network model similar to the Nvidiais's, I thought this model might be appropriate because it has achived a good result.

In order to gauge how well the model was working, I use the Udacity given sample data, split the image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model using dropout layers.

Then I re-train the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track,  to improve the driving behavior in these cases, I use the left and right camera images together with the center camera image.

At the end of the process, the vehicle is able 

to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture are as follows:
Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

  - To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

 - I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to steer if the car drifts off to the left or the right.

![alt text][image3]
![alt text][image4]

- To fullly use the collected data set, I also use the left and right camera images to train my model to get higher generalization. For example, here are the images,After the collection process, I had 3 times image number.

![alt text][image5]
![alt text][image6]

- The cameras in the simulator capture 160 pixel by 320 pixel images.
Not all of these pixels contain useful information, in order to train model faster I crop each image to focus on only the portion of the image that is useful for predicting a steering angle.

- I also use keras Lambda Layers to parallelize image normalization. In Keras, lambda layers can be used to create arbitrary functions that operate on each image as it passes through the layer.

- I finally randomly shuffled the data set and put 20% of the data into a validation set. 

- The images captured in the car simulator is so large that storing 10,000 simulator images would take over 1.5 GB. That's a lot of memory! Not to mention that preprocessing data can change data types from an int to a float, which can increase the size of the data by a factor of 4.

- Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.

- Finally, I use these training validation datas to training the model. The validation set helps determine if the model is over or under fitting. The ideal number of epochs is maybe 15 according to the following training process picture , I use an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image7]