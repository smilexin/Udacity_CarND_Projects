# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_set_counts.png "Traffic sign counts"
[image2]: ./random_train_samples.png "Random Samples"
[image3]: ./random_train_gray_samples.png "Grayscaling"
[image4]: ./lenet.png "LeNet-5"
[image5]: ./traffic_signs_found.png "Traffic Sign from web"
[image6]: ./softmax_probabilities.png "softmax_probabilities"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

You're reading it! 

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle to load the images, and numpy library to calculate summary statistics of the traffic
signs data set:

The data used are divided into three set: training set, validation set and test set. The sample of Each data set is a dictionary about features and labels key/value pairs.

* The size of training set 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. 

* The flollowing is a bar chart showing the 43 unique classes/labels and the numbers of each class of training set.

![alt text][image1]

* The following is the rgb images of the training data set.
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step, I decided to convert the images to grayscale because rgb images can easily be disturbed by the environment such as night etc. and traffic sign cannot depend on color to classify. Turning into grayscale, on one side can reduce the demention to increase the performance, on the other hand we can easily get the gradient information from grayscale image.

Here is examples of training set traffic sign images of each classes after grayscaling.

![alt text][image3]
We can see that grayscaling somewhat improves the visibility of the images, but some of them still remain very dark.

* As a last step, I normalized the image data so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.

* It's useful to generate additional datas when the data set is not enough in some classes, or some of the images are so hard to be recognized. Using transform、scale、 sharpen、 rotate operations and so on to process on the training sample datas, we can augment the date set, here I leave it to my feature try.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I use the classical LeNet-5 model as introduced in the cause as follows:

![alt text][image4]

My final model consisted of the following layers:

|        Layer          |              Description	        				
| --------------------- | --------------------------------------------- 
| Input                 | 32x32x1 grayscale image   							
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	
| RELU			|												
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	
| Convolution 5x5	| 1x1 stride, VALID padding, outputs 10x10x16
| RELU			|	 	
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16 
| flatten		| outputs 400 							
| Fully connected	| outputs 120
| RELU			|   
| dropout               | keep_prob 0.5 	
| Fully connected	| outputs 84
| RELU			|
| dropout               | keep_prob 0.5 							
| Fully connected	| outputs 43     									

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the key steps are as follows:

* Shuffle the training set
* Tune the epoch = 60, batch size = 100, to train the model
* Use the mu = 0, and sigma = 0.1 when use tf.truncated_normal to get initial convolution kernel
* Use cross entropy to evaluate and update the model parameters
* The learning rate is also an important parameter which will affect the convergence rate, I tune rate to 0.0009.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Training set accuracy of 0.999
* Validation set accuracy of 0.960
* Test set accuracy of 0.944

I choice the well known architecture LeNet-5:

* The LeNet-5 architecture was chosen because this model had proved to be classic which is enough to construct a classifier of traffic signs in relatively not complicated case.
* The reason why I believe it would be relevant to the traffic sign application is that: The LeNet-5 model is a deep learning network using convolutional kernel, which can extract features by itself. So on traffic signs classifier case, LeNet-5 can successfuly extract features from training set.
* The final model's accuracy on the training 0.999, validation 0.960 and test set 0.944, provide evidence that the model works well. The model learns well form the training set, the final test rate is not so high which indicate that further tuning needed to increate the model's accuracy, such as add more convolution layers, reduce fullly connected layers, pooling, dropout optimize and so on. 
 

### Test a Model on New Images

#### 1. Choose eight German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that found on the web: These images turn out to be more clear and brighter then the data set, have different lighting conditions and relatively different background.

![alt text][image5]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).




The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%, Compared to the test accuray of 0.944, and last image was failed classified indicate that the model need further train and tune.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

Here are the results of each image top 3 predictions:

![alt text][image6]

For the last image, the model is miss classified the image, and have a low predict probablity of 14% of the right class.

To get a higher accuracy model, there always a long way to go. Fighting...   

Links:

Udacity traffic signs classifier project link [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Gitbub reference links [kenshiro-o_github](https://github.com/kenshiro-o/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb),  [jeremy-shannon_gitbub](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
