# Image-Compressed-Sensing-using-Convolutional-Neural-Network
In Traditional image acquisition, the analog image is first acquired using a dense
set of samples based on the Nyquist-Shannon sampling theorem, of which the
sampling ratio is no less than twice the bandwidth of the signal, then compress the
signal to remove redundancy by a computationally complex compression method
for storage or transmission.

Compressive Sensing theory shows that a signal can be recovered from many
fewer measurements than suggested by Nyquist-Shannon Sampling theorem when
the signal is sparse in some domain.

We have Implemented a CS framework using Convolutional Neural Network
(CSNet) that includes a sampling network and a reconstruction network, which are
optimized jointly.

The sampling network adaptively learns the floating-point Sampling Matrix
from the training images, making the CS measurements retain more image
structural information for better reconstruction.

The reconstruction network learns an end-to-end mapping between the CS
measurements and the reconstructed images. It consists of an Initial
reconstruction network and a non-linear deep reconstruction network based
on residual learning. The reconstruction network can effectively utilize inter-block
information and avoid blocking artifacts.

# Compressed-sensing-using-Convolutional-Neural-Network-Architecture
![Compressed-sensing-using-Convolutional-Neural-Network-Architecture](/Static/CS%20Net%20Framework.PNG)

# Results
We are getting very good results with high PSNR and SSIM values on test images

Average PSNR - 25.402

Average SSIM - 0.543

Left Image represents input image

Right Image represents regenerated image after compressed sensing

![Results](/Static/predicted45000.PNG)

![Results](/Static/Capture2.PNG)

![Results](/Static/Capture3.PNG)

![Results](/Static/Capture4.PNG)

# Reference
We referred Image Compressed Sensing using Convolutional Neural Network by Wuzhen Shi , Feng Jiang and Debin Zhao IEEE 2019.

Research paper is referred below :-

![Reference](/Static/Image%20Compressed%20Sensing%20Using%20Convolutional%20Neural%20Network%20(2).pdf)

# Instructions to run code
1. Clone the repository on your machine
2. Download cudann to run gpu on your machine only for Nvidia graphic card
3. Download python, pytorch and all the other necessary libraries used my me in my code
4. Dowload training dataset from  BSDS500 (The Berkeley Segmentation Dataset and Benchmark) database website.
5. Change the locations of directories in python files data_augmentation.py , training.py of training dataset according to location of training dataset you downloaded on your machine.
6. Change location of directory to save images after data augmentation according to your preference in data_augmentation.py
7. Open command prompt in the file you cloned in your machine.
8. Run commands in order advised below.
9. `python data_augmentation.py`
10. `python training.py`
11. `python testing.py`
