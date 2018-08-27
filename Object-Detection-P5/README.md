# Udacity CarND Term1
---
# Project5: Vehicle Detection and Tracking

## This project goal is to write a software pipeline to detect vehicles in a video. 

- One of the main aim of this project is to familiarize ourself with traditional computer vision and machine learning techniques. Such as, use HOG and special binnings for feature extraction and use Linear SVM classifier for object detection.

- The Deep Learning implementations like YOLO and SSD that utilize convolutional neural network stand out for this purpose but when you are a beginner in this field, its better to start with the classical approach.

For me, I prefer Deep Leaning network like YOLO for It's easy to implement and relatively fast to achive the detection goal, If time permits, I would like to try the classical machine learning method
 to get a better understanding of object detection. Refer to Vehicle\_Detection\_P5.ipynb. 

# 1. Brief Intro
- ###An object detection problem can be approached as either a classification problem or a regression problem. Classification focus on portions of the image to determine objects, generate bounding boxes for regions that have positive classification results. The main mathod are as follows:

	1) Sliding Window + HOG or CNN

	2) Two-stage: Region Proposals + CNN (R-CNN), or Fast-RCNN, Faster-RCNN, refer: [arxiv](https://arxiv.org/abs/1506.01497)

	For example, R-CNN use region proposal methods to first generate potential bounding boxes in an image and then run a CNN classifier on these proposed boxes. After classification, post-processing is used to refine the bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene. These complex pipelines are slow and hard to optimize because each individual component must be trained separately.

- ###Regression focus on the whole image to generate bounding boxes coordinates directly from CNN. The popular network are as follows:
	
	1) One-stage: Tiny-YOLO v1, YOLO v1, v2, v3, refer: [arxiv](https://arxiv.org/abs/1506.02640), [Darknet](https://pjreddie.com/darknet/)
	
	2) SSD, refer:[SSD](https://arxiv.org/abs/1512.02325)

	YOLO reframes object detection as a single regression problem, and go straight from image pixels to bounding box coordinates and class probabilities. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance.

In this project we will implement tiny-YOLO v1. Full details of the network, training and implementation are available in the paper [YOLO](http://arxiv.org/abs/1506.02640).

[//]: # (Image References)
[image1]: ./output_images/YOLO_Dection_System.jpg
[image2]: ./output_images/YOLO-network.jpg
[image3]: ./output_images/Prediction_Value_Structure.jpg
[image4]: ./output_images/Tensor_Value_interpretation.png
[image5]: ./output_images/Tensor_Value_interpretation1.jpg
[image6]: ./output_images/test1.png
[image7]: ./output_images/test6.png
[video1]: ./project_video_out.mp4

# 2. Tiny-YOLO v1
YOLO algorithm use a sigle CNN network to implement a end-to-end object detection system.
![alt text][image1]

## 2.1. Architecture

The model architecture consists of 9 convolutional layers, followed by 3 fully connected layers. Each convolutional layer is followed by a Leaky RELU activation function, with alpha of 0.1. The first 6 convolutional layers also have a 2x2 max pooling layers.

The first 9 convolution layers can be understood as the feature extractor, whereas the last three full connected layers can be understood as the "regression head" that predicts the bounding boxes.

![ale  text][image2]

## 2.2. Use pretrained weights
The weights I used are from the darknet site and are from training the model on the VOC2012 Dataset. This is why there are 20 classes to choose from even though we'd be happy with just a car class.

You can download the pretrained weights from [here](https://drive.google.com/file/d/0B1tW_VtY7onibmdQWE1zVERxcjQ/view) and load them into keras model.

> load_weights(model,'yolo-tiny.weights')

## 2.3. Implementation

- YOLO divides the input image into an **S** x **S** grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts **B** bounding boxes and confidence scores for those boxes. Each grid cell also predicts **C** conditional class probabilities. These probabilities are conditioned on the grid cell containing an object.

![alt text][image3]

- So at test time, the final output vector for each image is a **S** x **S** x (**B** x 5 + **C**) length vector. In Tiny-YOLO network, with **S** goes to 7, **B** goes to 2 and **C** goes to 20, so the output of TOLO network is a vector of 1470 numbers.

- Each bounding box consists of 5 predictions:

![alt text][image4]

- Confidence is defined as (Probability that the grid cell contains an object) multiplied by (Intersection over union of predicted bounding box over the ground truth). If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth. SO the scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.


- For every grid cell, We will get two bounding boxes, which will make up for the starting 10 values of the 1*30 tensor. The remaining 20 denote the number of classes. The values denote the class score, which is the conditional probability of object belongs to class i, if an object is present in the box. Next, we multiply all these class score with bounding box confidence and get class scores for different bounding boxes. We do this for all the grid cells.

![alt text][image5]

## 2.4. Postprocessing
As you can see in the above image, each input image is divided into an **S** x **S** grid and for each grid cell, our model predicts **B** bounding boxes and **C** confidence scores. There is a fair amount of post-processing involved to arrive at the final bounding boxes based on the model's predictions.
### 1) Extract car boxes parameters from the YOLO network

    car_class_number = 6 # Position for class 'car' in the VOC dataset classes
    boxes = []
    SS = S*S  # number of grid cells
    prob_size = SS*C  # class probabilities
    conf_size = SS*B  # confidences for each grid cell

    probabilities = yolo_output[0:prob_size]
    confidence_scores = yolo_output[prob_size: (prob_size + conf_size)]
    cords = yolo_output[(prob_size + conf_size):]

    # Reshape the arrays so that its easier to loop over them
    probabilities = probabilities.reshape((SS, C))
    confs = confidence_scores.reshape((SS, B))
    cords = cords.reshape((SS, B, 4))

    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c = confs[grid, b]
            # bounding box x and y coordinates are offsets of a particular grid cell location,
            # so they are also bounded between 0 and 1.
            # convert them absolute locations relative to the image size
            bx.x = (cords[grid, b, 0] + grid % S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S

            bx.w = cords[grid, b, 2] ** sqrt
            bx.h = cords[grid, b, 3] ** sqrt

            # multiply confidence scores with class probabilities to get class sepcific confidence scores
            p = probabilities[grid, :] * bx.c

            # Check if the confidence score for class 'car' is greater than the threshold
            if p[car_class_number] >= threshold:
                bx.prob = p[car_class_number]
                boxes.append(bx)

### 2) Class score threshold
We reject output from grid cells below a certain threshold (0.2) of class scores.

    # Check if the confidence score for class 'car' is greater than the threshold
    if p[car_class_number] >= threshold:
        bx.prob = p[car_class_number]
        boxes.append(bx)

### 3) Reject overlapping (duplicate) bounding boxes
If multiple bounding boxes, for each class overlap and have an IOU of more than 0.4 (intersecting area is 40% of union area of boxes), then we keep the box with the highest class score and reject the other box(es).

    # combine boxes that are overlap
    # sort the boxes by confidence score, in the descending order
    boxes.sort(key=lambda b: b.prob, reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0:
            continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            # If boxes have more than 40% overlap then retain the box with the highest confidence score
            if box_iou(boxi, boxj) >= 0.4:
                boxes[j].prob = 0

### 4) Drawing the bounding boxes
The predictions (x, y) for each bounding box are relative to the bounds of the grid cell and (w, h) are relative to the whole image. To compute the final bounding box coodinates we have to multiply w & h with the width & height of the portion of the image used as input for the network.

    imgcv1 = im.copy()
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    
    height, width, _ = imgcv1.shape
    for b in boxes:
        w = xmax - xmin
        h = ymax - ymin

        left  = int ((b.x - b.w/2.) * w) + xmin
        right = int ((b.x + b.w/2.) * w) + xmin
        top   = int ((b.y - b.h/2.) * h) + ymin
        bot   = int ((b.y + b.h/2.) * h) + ymin

        if left  < 0:
            left = 0
        if right > width - 1:
            right = width - 1
        if top < 0:
            top = 0
        if bot>height - 1: 
            bot = height - 1

## 3.  Detection Results

### 1) Preprocess of Input Image
Crop, nomerlize, and transpose

    cropped = image[300:650,500:,:]
    cropped = cv2.resize(cropped, (448,448))
    normalized = 2.0*cropped/255.0 - 1
    # The model works on (channel, height, width) ordering of dimensions
    transposed = np.transpose(normalized, (2,0,1))

### 2) Test on image

	test_image = mpimg.imread('test_images/test1.jpg')
	pre_processed = preprocess(test_image)
	batch = np.expand_dims(pre_processed, axis=0)
	batch_output = model.predict(batch)
	boxes = yolo_output_to_car_boxes(batch_output[0], threshold=0.25)
	final = draw_boxes(boxes, test_image, ((500,1280),(300,650)))

The following are the test images with bounding boxes around the cars and car probablity value on the bottom of the each box.

![alt text][image6]

![alt text][image7]

### 3) Define a video pipeline

step1: Pre process the image

setp2: Use the modle to predict

setp3: Post process the model output to extract the car boxes

step4: Draw boxes on the image 

    pre_processed = preprocess(image)
    batch = np.expand_dims(pre_processed, axis=0)
    batch_output = model.predict(batch)
    boxes = yolo_output_to_car_boxes(batch_output[0], threshold=0.20)
    final = draw_boxes(boxes, image, ((500,1280),(300,650)))

### 4) Test on video
Final use moviepy VideoFileClip to apply the pipeline to a video, Refer to project\_video\_out.mp4.
[Video](video0)


## 4. Areas of Improvement
1) There are some frames that the car is not dectected. Use hotmap of last sevaral frames combined with currrent frame prediction to get a more stable detection.

2) Compared to R-CNN, YOLO v1 run faster, but has a higher error rate of localization.

3) In YOLO V1, each grid cell only predict 2 boxes, and has a higer accuracy on reletively larger object, to litter object detection, It has a pool performance. The higer version of YOLO has improved which worth to have a try.

4) The pipeline is only reflects the detection of an object and never the level of object tracking. So there is always a lot of work to do.

## 5. Reference

[YOLO](https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e), [Github1](https://github.com/xslittlegrass/CarND-Vehicle-Detection),
[Github2](https://github.com/subodh-malgonde/vehicle-detection/blob/master/Vehicle_Detection.ipynb)