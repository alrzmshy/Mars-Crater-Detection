# Mars-Crater-Detection
Implementing YOLO to detect craters on Mars. </br>


<p float="left">
  <img src="https://raw.githubusercontent.com/alrzmshy/Mars-Crater-Detection/master/test1.png" width="224" hspace="40"/>
  <img src="https://raw.githubusercontent.com/alrzmshy/Mars-Crater-Detection/master/test2.png" width="224" hspace="40"/> 
  <img src="https://raw.githubusercontent.com/alrzmshy/Mars-Crater-Detection/master/test7.png" width="224" />
</p>

(Adapted from https://github.com/experiencor/basic-yolo-keras , many thanks to him for his help.)</br>
Pre-trained weights are provided in his repo.

## Steps for setting up the training:

### 0. Download the data:

Run: `python download_data.py`
This will create a directory named `data` and downloads the images and the labels.

### 1. Extract the images and create annotations:

Run: `python save_images.py --dataset` and `python annotation.py --dataset` </br>
Note that the choice of parameter dataset should be either train or test.

### 2. Launch the training:

Run: `python train.py`</br>

I could not get the loss to converge without the pre-trained weights. They are necessary for a good initialization. The training was done on a NVIDIA GTX 1080 Ti. Some sample detections from the images in the test set are shown above.</br>




