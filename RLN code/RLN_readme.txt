RLN is the companion code to our paper:
Incorporating physical model into deep learning network for resolution enhancement in optical microscopy

RLN is a 3D fully convolutional deep learning incorporating the Richardson-Lucy deconvolution formula to restore and enhance the resolution of fluorescence microscopy image.

System Requirements:
Ubuntu 16.04
Python 3.6+
NVIDIA GPU
CUDA 10.1.243 and cuDNN 7.6.4
Tensorflow 1.14.0

Dependencies Installation:
If using conda, conda install tifffile -c conda-forge.
Otherwise, pip install tifffile.


Main folder:
--   train
      --input
      --(input2) for dual-input mode
      --ground truth
      --model_rl
      --output_rl
      --labels.txt
--   test
      --input
      --(input2) for dual-input mode
      --(ground truth) for validation
      --output_rl
      --labels.txt

During training, put training dataset into the input folder and ground truth folder, each input/ground truth (input2) image pair share the same file name. Labels.txt summary the name of all the image pairs. model_rl folder is used to save the trained model. The output_rl model is used to save the output during training, this is not necessary and just to supervise the training procedure.

During testing, put testing dataset into the input folder and list the testing data name into the labels.txt. The output_rl model is used to save the test result. If you have ground truth, you can try validation to metric the difference between the output and ground truth.


makedata3D_train_single.py and makedata3D_train_dual.py are the data loading and preprocessing files for single-input and dual-input training respectively.
makedata3D_test_single.py and makedata3D_test_dual.py are the data loading and preprocessing files for single-input and dual-input testing respectively.


After preparing the dataset and relative folders, you can set the main parameters for train in the RLN-single.py or RLN-dual.py:

mode: TR:train ; VL:validation, with known ground truth ; TS: test , no ground truth
relative folders:
data_dir = '/home/liyue/newdata1/'  # the main folder including the train and test folders
model_path='/home/liyue/newdata1/train/model_rl/new_single_used/' #the folder for the trained model to be saved
train_output='/home/liyue/newdata1/train/output_rl/' #the train output saved folder
test_output='/home/liyue/newdata1/test/output_rl/'  # the validating output or testing output folders

train_iter_num=iter_per_epoch*epochs
test_iter_num=testing_data_numbers
train_batch_size=as you want
test_batch_size=1

and you can set the learning rate in:
self.learning_rate = tf.train.exponential_decay(0.02,self.global_step,1000,0.9,staircase=False)

After you setting these parameters, you can run: python RLN-single.py or python RLN-dual.py
