# Project: Traffic Sign Recognition

This project has been prepared by Andre Strobel.

The goal of this project is to train a convolutional neural network to later classify traffic sign images.

Everything has been programmed in Python 3 using Tensorflow.

---

## Content

1. Data set preparation
    1. Loading the existing data
    1. Visualization of the data
1. Convolutional neural network architecture
    1. Model definition
    1. Hyper parameter selection and training
    1. Test of model performance after training
1. Predictions with the trained model
    1. Test images from the web
    1. Exploring the inside of the model
    1. Fun with unknown traffic signs
    1. Searching within larger pictures
1. Discussion

[//]: # (Image References)

[image1]: docu_images/01_01_Random_Images.png
[image2]: docu_images/01_02_Single_Random_Label.png
[image3]: docu_images/01_03_High_Contrast.png
[image4]: docu_images/01_04_Average_Images.png
[image5]: docu_images/02_01_Epochs.png
[image6]: docu_images/02_02_Web_Images.png
[image7]: docu_images/03_01_Image_1.png
[image8]: docu_images/03_02_Plot_1.png
[image9]: docu_images/03_03_Image_2.png
[image10]: docu_images/03_04_Plot_2.png
[image11]: docu_images/03_05_Image_3.png
[image12]: docu_images/03_06_Plot_3.png
[image13]: docu_images/03_07_Image_4.png
[image14]: docu_images/03_08_Plot_4.png
[image15]: docu_images/03_09_Image_5.png
[image16]: docu_images/03_10_Plot_5.png
[image17]: docu_images/03_11_Image_6.png
[image18]: docu_images/03_12_Plot_6.png
[image19]: docu_images/04_01_Stop.png
[image20]: docu_images/04_02_Layer_1.png
[image21]: docu_images/04_03_Layer_2.png
[image22]: docu_images/04_04_Layer_3.png
[image23]: docu_images/05_01_Image_1.png
[image24]: docu_images/05_02_Plot_1.png
[image25]: docu_images/05_03_Image_2.png
[image26]: docu_images/05_04_Plot_2.png
[image27]: docu_images/05_05_Image_3.png
[image28]: docu_images/05_06_Plot_3.png
[image29]: docu_images/05_07_Image_4.png
[image30]: docu_images/05_08_Plot_4.png
[image31]: docu_images/05_09_Image_5.png
[image32]: docu_images/05_10_Plot_5.png
[image33]: docu_images/05_11_Image_6.png
[image34]: docu_images/05_12_Plot_6.png
[image35]: docu_images/06_01_Large.png
[image36]: docu_images/06_02_All_Detect.png
[image37]: docu_images/06_03_Top_Detect.png
[image38]: docu_images/06_04_Predicted_Images.png

---

## 1. Data set preparation

### 1. Loading the existing data

I loaded the provided traffic signs data set and used basic Python operations to get an overview of the content:

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

The numerics of neural networks work best if the mean of the input data is zero. I used the following equations to normalize all input data to be between -1 and 1.

```python
# normalize input data
X_train_norm = np.asarray(((X_train / 127.5) - 1), np.float32)
X_valid_norm = np.asarray(((X_valid / 127.5) - 1), np.float32)
X_test_norm = np.asarray(((X_test / 127.5) - 1), np.float32)
```

As the color of traffic signs is used very intentionally to cluster signs by their importance, I decided that my detection algorithm should not be based on gray scale images only. Therefore, I left the color channel in the input data.

I did not augment the training data set by generating additional input using transformations. This is a potential future improvement.

### 2. Visualization of the data

In order to plot a set of traffic sign images with labels I created the function `plot_traffic_signs`. I used this function to understand the picture content by looking at the training dataset in the following ways.

* 30 random images

![alt text][image1]

* 30 random images of the same random label

![alt text][image2]

* A very high contrast image for each label (selected by looking for the highest contrast in the center of the image)

![alt text][image3]

* The average image for each label (calculated by averaging the individual color channels of all images with the same label)

![alt text][image4]

## 2. Convolutional neural network architecture

### 1. Model definition

I started by adapting the `LeNet` example to take color images as input.

My first intention was to create a very flexible set of functions that defines a typical type of convolutional neural network based on a few input parameters for the graph layout. Unfortunately, I ran into the issue that I couldn't get the same result when choosing the parameters to represent the simple `LeNet` example. My attempts are documented in the functions `LeNet_adjusted_method` and `LeNet_adjusted_inlinemethod`. I suspect that the way Python transfers variables between functions and how I implemented this leads to missing links in the Tensorflow graph during execution.

I reverted back to defining all Tensorflow variables in a single function `LeNet_adjusted3`. The high level structure of the convolutional neural network is shown below as generated by this function.

```
Convolutional layer   1 : [32, 32] input dimension with depth 3 and [28, 28] output dimensions with depth 18
Convolutional layer   2 : [28, 28] input dimension with depth 18 and [20, 20] output dimensions with depth 54
Pooling layer         2 : [20, 20] input dimension with depth 54 and [10, 10] output dimensions with depth 54
Convolutional layer   3 : [10, 10] input dimension with depth 54 and [6, 6] output dimensions with depth 128
Fully connected layer 1 : 4608 input dimensions and 800 output dimensions
Fully connected layer 2 : 800 input dimensions and 84 output dimensions
Fully connected layer 3 : 84 input dimensions and 43 output dimensions
```

I designed the first convolutional layer with a filter size of 5 to detect 18 features instead of 6 like `LeNet`, because I am using 3 color channels in the input data. I intentionally skipped pooling in the first layer to keep as much detail as possible.

The second convolutional layer uses a larger filter size of 9 to detect larger features in the picture. I decided to triple the number of possible features when combining smaller features. To keep the network reasonably small, the second layer uses max pooling with a stride of 2.

The third convolutional layer again uses a filter size of 5 and transforms most of the remaining image structure into a total of 128 features. No pooling is used in the third layer as the size of the network is reasonably small.

The following three fully connected layers transform the features into class probabilities by continuously reducing the dimensions from 4608 to 800 to 84 and finally 43 - one class for each traffic sign label.

Each layer of the convolutional neural network `LeNet_adjusted3` uses *RELU* units followed by *dropout* except the last fully connected layer.

The model pipeline uses the `AdamOptimizer` from Tensorflow for training. The loss function is based on `reduce_mean` from Tensorflow using `softmax_cross_entropy_with_logits`. To further avoid overfitting, the weight matrices of the first and second convolutional layer get regularized using `l2_loss`.

The model accuracy is evaluated by calculating the average difference between the predicted labels and the one hot encoded input labels.

### 2. Hyper parameter selection and training

All layers use random states as initial values with a mean of zero and standard deviation of 0.1. The following hyperparameters were used to train the model that has been used in the following sections. The variable `epochs` defines the number of training epochs. The `batch_size` defines how many inputs are used between every update of the internal parameters. The learning rate `rate` is slightly smaller than the standard `AdamsOptimizer` setting to allow a smoother progression. All layers with *dropout* keep 70 percent of their connections as defined by `keep_prob`. The parameter `beta` is used as factor during regularization of the convolutional weights in the loss function.

```
# define constants
epochs = 50
batch_size = 128
rate = 0.0005
keep_prob = 0.7
beta = 0.1
```

The training progress is shown in the following diagram. After 50 epochs the model achieved an accuracy of 96.0 percent.

![alt text][image5]

### 3. Test of model performance after training

The accuracy on the test data set is 94.3 percent.

## 3. Predictions with the trained model

### 1. Test images from the web

To further test whether the model is good in not only predicting the images on which it has been trained, 6 images of German traffic signs have been found using [Google's image search](https://images.google.com/). I created a function `load_images` to load such a sequence of images.

![alt text][image6]

The function `predict_image_labels` uses the previously described model to predict the labels of these untrained images. The function `check_accuracy` provides a quick check whether the untrained images are accurately predicted. A more thorough check is defined by the function `evaluate_top_k` which is used in the following.

The model accurately predicts each of these untrained traffic signs as shown in the following picture sequence. The top 5 predictions are shown using the average image for each of these labels. The bar charts show the *softmax* probability for each prediction.

![alt text][image7]![alt text][image8]
![alt text][image9]![alt text][image10]
![alt text][image11]![alt text][image12]
![alt text][image13]![alt text][image14]
![alt text][image15]![alt text][image16]
![alt text][image17]![alt text][image18]

### 2. Exploring the inside of the model

In order to visualize the weights in the convolutional layers of my neural network, I evaluated the model with the average *Stop* sign image from the training data set using the function `outputFeatureMap`.

![alt text][image19]

The first convolutional layer uses 3 color channels as input. Therefore, I chose to visualize the weights of each feature map using an *RGB* color image. The 18 feature maps of this layer clearly show that they are distinct by having different average color tones. The images on the left are more green-ish while the ones on the right show more blue and red color tones. Also, the area in which they emphasize on specific color inputs is very different. Especially the images in the center seem to emphasize on diagonal directions.

![alt text][image20]

The second convolutional layer has 54 feature maps for each of the 18 input channels. The following picture only shows the first 250 feature maps as grayscale images (a little more than 25 percent of all feature maps in this layer). Due to the larger size of the filter in the second layer, some of the festures are very detailed while others focus on more general patterns.

![alt text][image21]

The third and final convolutional layer has 128 feature maps for each of the 54 input channels. The following picture only shows the first 250 feature maps as grayscale images (less than 4 percent of all feature maps in this layer). The filter size is again smaller and hence the individual features in these feature maps are coarser.

![alt text][image22]

### 3. Fun with unknown traffic signs

The real fun with convolutional neural networks starts when we use them for something that they have not been trained for. What would they predict when they see US instead of German traffic signs? Let's try it!

For the *Intersection* sign the model probably focuses on the large vertical black line in the center and thinks it might be a *General Caution* sign.

![alt text][image23]![alt text][image24]

The *Pedestrian Crossing* sign clearly puts the model out of its comfort zone and it cannot predict anything with a high probability.

![alt text][image25]![alt text][image26]

I think it is facinating that my model accurately predicts a *Yellow Stop* sign to be a *Stop* sign with pretty high probability although my model also takes into account the color.

![alt text][image27]![alt text][image28]

Now here it is: Who would have guessed that the US *Traffic Light* sign is close to the German *Traffic Light* sign? Well, I didn't and so does the model. It doesn't even consider it as part of the top 5 predictions.

![alt text][image29]![alt text][image30]

And it gets better: The *Right Turn Ahead* signs are pretty close besides being edgy versus round and the model nails it.

![alt text][image31]![alt text][image32]

Well, and here is the "Right Turn" sign that is really for away from anything in the training data set. Similar to the first sign in this fun sequence, the model probably focuses on the vertical black line in the middle and predicts a *General Caution* sign. Well, it's always good to be cautious with your predictions - or not?

![alt text][image33]![alt text][image34]

### 4. Searching within larger pictures

In real life we probably don't have the luxury of predicting mug shots of traffic signs. We have to find them in a larger image. And now they can be larger or smaller. I created a little algorithm in the function `check_large_images` that scans a larger image with different scaling levels and tries to find traffic signs of any size in any position.

The below pictures show the original image, a picture in which I marked all areas in which potential traffic signs have been predicted and finally a picture in which only the top predictions have been marked. The predicted areas have a white outline for the most likely predictions and a black outline for the least likely predictions. As expected the most likely predictions for traffic signs are in the area of the actual traffic sign in the larger image.

![alt text][image35]![alt text][image36]![alt text][image37]

But which specific traffic signs does the model predict? It accurately lists the *Yield* and *Roundabout Mandatory* signs. Somehow it sees things that I don't see though and gives the *Priority Road* sign the highest probability to be in this larger image.

![alt text][image38]

## 4. Discussion

Most images used to test my model are nice frontal shots of sunny day traffic signs. Looking at some of the results makes it very obvious that traffic signs that are covered by obstacles or snow or images taken at night during rain would challenge my model extremely.

At the same time I tried to not overfit my model and have fun evaluating its predictions on formerly unknown images. I could have used more input data to train the model by e.g. augmenting the training data set. While rotational and perspective transformations definitely would improve its predictability, I would stay away from darkening or blurring the existing training data set for augmentation as this could also lead to overfitting - unless I had access to a much larger training data set and overfitting would become less of an issue.

I am looking forward to learning higher level neural network programming packages like [Keras](https://keras.io/) which will make it easier to define the network architecture and experiment with different architectures in less time - my own attempt to create something like this with minimum effort failed.

In summary I was surprised to see with which little effort (although much knowledge and fast hardware) one can detect traffic signs in images.