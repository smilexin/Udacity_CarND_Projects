

#Advanced Lane Finding Project

###The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistorted_image.PNG "Undistorted"
[image2]: ./output_images/Unwarped_binary_image.PNG "Unwarped binary image"
[image3]: ./output_images/Sliding_window_image.PNG "Sliding window"
[image4]: ./output_images/Unwarped_fit_using_pre_frame.PNG "Using pre frame fit"
[image5]: ./output_images/radius_and_center.jpg "Radius and center"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the part two of the IPython notebook located in "./project.ipynb".  

- I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
- Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
- Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.

- I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in part three and part four in nootbook `project.ipynb`). 
- Gradient thresholds include soble in both x and y axis, gradient magnitude and direction threashold; Color threshold used b channel in lab color space and l channel in hls color space. 

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `upwarp()`, which appears in notebook part four in the file `project.ipynb`. The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following:

- src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])

- dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

I verified that my perspective transform was working as expected by transform the test image and its warped counterpart to verify that the lines appear parallel in the warped image. The following is the example of unwarped binary image and its original images.

![alt text][image2]

#### 3. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- The functions sliding_window_polyfit and polyfit_using_prev_fit, which identify lane lines and fit a second order polynomial to both right and left lane lines, which can find in jupyter notebook the fifth part. 
- The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram, but in my final implementation I changed these to quarters of the histogram just left and right of the midpoint. This helped to reject lines from adjacent lanes. 
- The function then identifies night windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image.
- Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. 
 
The image below demonstrates how this process works:

![alt text][image3]


The polyfit_using_prev_fit function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. 
 
The image below demonstrates this:

-  the green shaded area is the range from the previous fit, 
- the yellow lines and red and blue pixels are from the current image:

![alt text][image4]


#### 4. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in sixed part in botebook `project.ipynb`

- The radius of curvature is based upon [this website](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) and calculated in the code cell titled "Radius of Curvature and Distance from Lane Center Calculation" using this line of code (altered for clarity):

- curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
- In this example, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. y_0 is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_meters_per_pixel is the factor used for converting from pixels to meters.
- This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

- lane_center_position = (r_fit_x_int + l_fit_x_int) /2
- center_dist = (car_position - lane_center_position) * x_meters_per_pix
- r_fit_x_int and l_fit_x_int are the x-intercepts of the right and left fits, respectively. This requires evaluating the fit at the maximum y value (719, in this case - the bottom of the image) 

#### 5. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in notebook in part seven and eight in `project.ipynb` in the function `map_data()`.

- A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix Minv and overlaid onto the original image.

Below is an example of the results of the draw_data function, which writes text identifying the curvature radius and vehicle position data onto the original image:

![alt text][image5]
### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The threshold binary image becomes so noisy du to different light, shadows, discoloration.
- Extract the region the lane line located through fillPoly and self defined vertices wasn't work well and i havn't figured out why.
- To make the pipeline more rubust, there are come directions we can go, such as how to fully use the last few frame fits to help find more accurate current fit; dynamically fine tune the threahold parameters to get a highl-quality bianry image and so on.


Thanks to the git hub resource by [Jeremy-shannon](https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines).




