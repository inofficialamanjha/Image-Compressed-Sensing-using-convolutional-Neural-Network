# ABSTRACT

Traditional Image acquisition systems based on Nyquist-Shannon sampling theorem require high acquisition energy per image along with high acquisition time. This system may not be favored in some image processing applications when the data acquisition device must be simple (e.g., inexpensive resource-deprived sensors) or when oversampling can harm the object being captured (e.g., medical imaging). The emerging technology of compressed sensing depicts a new paradigm for image acquisition and reconstruction that is highly efficient.

In the study of Compressed sensing (CS), the two main challenges are the design of the sampling matrix and development of the reconstruction method. On the one hand, the usually used random sampling matrices (e.g., GRM) are signal independent, which ignore the characteristics of the signal. On the other hand, the state-of-the-art image CS methods (e.g., GSR and MH) achieve quite good performance, however, with much higher computational complexity.

To deal with two challenges, we have implemented an image CS framework using Convolutional Neural Network (CSNet), that includes a sampling network and a reconstruction network, which are optimized jointly.

Experimental results demonstrate that CSNet offers state-of-the-art reconstruction quality while achieving fast running speed. The learned sampling matrices can improve the traditional image CS reconstruction method as it retains more image-structural information.


# INTRODUCTION

In Traditional image acquisition, the analog image is first acquired using a dense set of samples based on the Nyquist-Shannon sampling theorem, of which the sampling ratio is no less than twice the bandwidth of the signal, then compress the signal to remove redundancy by a computationally complex compression method for storage or transmission.

Compressive Sensing theory shows that a signal can be recovered from many fewer measurements than suggested by Nyquist-Shannon Sampling theorem when the signal is sparse in some domain.

We have Implemented a CS framework using Convolutional Neural Network (CSNet) that includes a sampling network and a reconstruction network, which are optimized jointly.

The sampling network adaptively learns the floating-point Sampling Matrix from the training images, making the CS measurements retain more image structural information for better reconstruction.

The reconstruction network learns an end-to-end mapping between the CS measurements and the reconstructed images. It consists of an Initial reconstruction network and a non-linear deep reconstruction network based on residual learning. The reconstruction network can effectively utilize inter-block information and avoid blocking artifacts.

The Learned model does not require the Sampling matrix to be transmitted from encoder to decoder as opposed to traditional CS methods. Also, the learned sampling matrix can improve traditional image CS reconstruction methods.


# METHODOLOGY


## Traditional Image Acquisition

In Traditional Image Acquisition, the analog image is first acquired using a dense set of samples based on the Nyquist-Shannon sampling theorem.

![Image 1](https://user-images.githubusercontent.com/75173703/116242056-2b10ae80-a783-11eb-9ad9-044c881793ad.PNG)

After acquiring, the image (dense image) is transformed over a basis Ψ (e.g., Fast Fourier transform or wavelet basis), for which most of the locations in the spatial domain of the image have small coefficients.

The pixels in the transformed image, are then truncated, and a vast amount of information (95% in Image 1) is removed. The acquired image (sparse image representation) is easy to store as a dictionary of {key, value} pairs where the key is the spatial orientation of pixel and value is the transform coefficient. This process is called compression.

The image data in the form of dictionary is comparatively easy to transfer over computer networks and comparatively acquires less amount of memory.

To reconstruct the image, inverse basis transform Ψ-1 is applied over the compressed sparse Image. This process is called decompression.


## Compressed Sensing Background

Compressed sensing is a form of data-acquisition through which we can reconstruct high-resolution images by sampling comparatively much lower amount of information as compared to the traditional Image acquisition method.

The CS theory permits linear projection of a high dimensional signal into a dimension much lower than that of the original signal while allowing exact recovery of the signal from the projections when the signal is sparse in some domain.

Concretely, suppose that x ∈ RN X 1 is a real-valued signal (Original Image), and Φ ∈ RM X N is a sampling matrix, M<<N, the CS measurement acquisition process is expressed as

y = Φx

where y ∈ R(M X 1) is the Compressed sensed measurement (CS measurement). Since the signal x is sparse in some domain Ψ, the CS theory shows that correctly recovering x is possible.

![Fig 2](https://user-images.githubusercontent.com/75173703/116242326-6f9c4a00-a783-11eb-8765-013bfe0aeaeb.PNG)

The most straightforward formulation of CS reconstruction can be expressed as :

![Eq 1](https://user-images.githubusercontent.com/75173703/116242436-89d62800-a783-11eb-8992-33100b85a3a1.PNG)

where Ψx are the sparse coefficients with respect to domain Ψ, with the subscript p is usually set to 1 or 0, characterizing the sparsity of the vector Ψx.

![Fig 3](https://user-images.githubusercontent.com/75173703/116242566-a83c2380-a783-11eb-9497-4e5a9c937495.PNG)

There are numerous strategies proposed for solving this optimization problem in the literature e.g., convex optimization, gradient-descent methods, greedy algorithms, Landweber (PL) algorithms.

Although so many efforts have been made, the requirement of iterative computation makes these traditional methods have high computational complexity. These CS methods take several seconds to several minutes to reconstruct a high quality image. Also, the sampling matrix should be transmitted from the encoder to the decoder.


## Compressed Sensing using Convolutional Neural Network

![Fig 4 - 1](https://user-images.githubusercontent.com/75173703/116242659-c43fc500-a783-11eb-8dd9-d31b19bf0eac.PNG)
![Fig 4 - 2](https://user-images.githubusercontent.com/75173703/116242747-dc174900-a783-11eb-917b-d1ada8ac3528.PNG)

Fig 4. Shows the framework of CSNet. CSNet uses CNN to implement three functions, i.e.
- Block-based compressed sampling.
- Initial signal reconstruction.
- Non-Linear Signal reconstruction.

The sampling network is used to learn the sampling matrix and acquire CS Measurements. CSNet performs Block by block sampling, to effectively use both Intra-block and Inter-block correlation to optimize the reconstructed image instead of an image block.

The reconstruction network contains a linear Initial reconstruction network and a non-linear reconstruction network, which learns an end-to-end mapping from the CS measurements to the reconstructed images.

In the training phase, the sampling network and the reconstruction network form an end-to-end network for joint optimization.

In the application phase, the sampling network is used as an encoder to generate CS measurements, and the reconstruction network is used as a decoder to reconstruct images.


## CSNet - Sampling Network

## CSNet - Initial Reconstruction Network

## CSNet - Deep Reconstruction Network

# TRAINING ALGORITHM & METHODOLOGY

## Joint Optimization Training Algorithm

## Training Methodology and loss function

The sampling and the reconstruction network introduced above form a CNN-based framework for image CS.

Given the input image x, our goal is to obtain the CS measurements y using the sampling network S and then recover the original input image x accurately from y by using the reconstruction network R.

The network is optimized jointly. The input and the label are all image x itself for training the CSNet. The training dataset can be represented as { xi , xi }Ki=1.
The mean square error is adopted as the cost function of CSNet.

We have two objectives to minimize:

1. Initial Reconstructed Image

Loss Function:

![Eq 2](https://user-images.githubusercontent.com/75173703/116243316-75def600-a784-11eb-91cc-f94afa41d651.PNG)

Where θ and ϕ are the parameters of the sampling network and reconstruction network needed to be trained.

S( xi ; θ ) are the CS measurements, and I ( S( xi ; θ ) is the initial reconstructed output with respect to image xi.


2. Final Reconstructed Image

Loss Function :

![Eq 3](https://user-images.githubusercontent.com/75173703/116243391-85f6d580-a784-11eb-8e71-b0b31af6ce79.PNG)

Where R ( S( xi ; θ ) ; ϕ ) is the Final reconstructed output.

Training is carried out by optimizing equations 1. And 2. Simultaneously using Adaptive moment estimation ( Adam ).

It should be noted that we train the sampling network and the reconstruction network jointly, but they can be used independently.


# TRAINING DETAILS

Our Training dataset has 211 images that are the 200 training images and 11 test images from the BSDS500 (The Berkeley Segmentation Dataset and Benchmark) database. The data augmentation technology is used to increase the training dataset.

The training images are prepared as 96 x 96 pixel sub-images. Finally, we have trained the model by randomly selecting
1. 1000 sub-images
2. 10,000 sub-images
3. 45,000 sub-images.

The Network parameters were set as follows:
1. Block size, B = 32.
2. Number of channels per image, l = 1.
3. Spatial size of the kernel, f = 3.
4. Number of feature maps in the Deep reconstruction network, d = 64.
5. Amount of non-linear mapping layer in the Deep reconstruction net., n = 5.
6. Sampling Ratio = 0.1.

For optimizing target functions, we use Adam Optimizer.

We train our model for 100 epoches, and each epoch iterates 1400 times with a batch size of 64. The learning rate of the epochs are as follows:
1. First 50 epoches, 10-3.
2. 51 to 80 epoches, 10-4.
3. 81 to 100 epoches, 10-5.

# EXPERIMENTAL RESULTS

We test these methods on a Test set of 11 Images from BSDS500 Dataset.

![Table 1](https://user-images.githubusercontent.com/75173703/116241554-a32aa480-a782-11eb-8816-43c1899d0503.PNG)

As shown in Table 1, The Average value of PSNR and SSIM increases with an increasing number of Images in the image set.

On a training set of **45,000** Images the Average value of **PSNR** is **25.402 dB**, and that of **SSIM** is **0.543**.

Therefore, CSNet offers state-of-the-art **reconstruction quality** while achieving **faster running speeds**. The learned sampling matrices have successfully applied to improve the traditional CS methods.

![Fig 5](https://user-images.githubusercontent.com/75173703/116241755-e1c05f00-a782-11eb-9298-82dd84548dea.PNG)

![Fig 6](https://user-images.githubusercontent.com/75173703/116241819-ef75e480-a782-11eb-802c-0b3f7810ddef.PNG)


# CONCLUSION

CSNet reconstructed images achieve high average Peak to Signal Ratio (PSNR) and Structural similarity index measure (SSIM), depicting that the learned CS measurements retain more image structural information for better reconstruction, offering state-of-the-art reconstruction quality, while achieving fast running speed.

# REFERENCES

1. [**W. Shi, F. Jiang, S. Liu and D. Zhao, "Image Compressed Sensing Using Convolutional Neural Network," in IEEE Transactions on Image Processing, vol. 29, pp. 375-388, 2020, doi: 10.1109/TIP.2019.2928136.**](https://ieeexplore.ieee.org/document/8765626)
2. [**BSDS500 Dataset**](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
3. [**Steve Brunton - Youtube Channel**](https://www.youtube.com/c/Eigensteve/)

# CODE RUN INSTRUCTIONS

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
