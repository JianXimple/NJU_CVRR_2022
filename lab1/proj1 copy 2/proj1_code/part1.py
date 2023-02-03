#!/usr/bin/python3

from cmath import pi
from cmath import exp
from select import kevent
from typing import Tuple
from cv2 import mean, sqrt

import numpy as np
import math
def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution
    
    Returns:
        kernel: 1d column vector of shape (k,1)
    
    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """
    mean = np.floor (ksize / 2)
    kernel=np.zeros((ksize,1))
    sum=0
    for i in range(ksize):
          x=1/(2*pi**0.5*sigma)*math.exp(-1/(2*sigma**2)*(i-mean)**2)
          kernel[i]=float(x)
          sum=sum+x
    # for i in range(ksize):
    #       x=kernel[i]
    #       kernel[i]=x/sum


   
    # raise NotImplementedError(
    #     "`create_Gaussian_kernel_1D` function in `part1.py` needs to be implemented"
    # )
    
    return kernel/sum

def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each 
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability 
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    ksize=cutoff_frequency*4+1
    kernel=np.zeros((ksize,ksize))
    mean=np.floor(ksize/2)
    kx=create_Gaussian_kernel_1D(ksize=ksize,sigma=cutoff_frequency)
    ky=create_Gaussian_kernel_1D(ksize=ksize,sigma=cutoff_frequency)
    kernel=np.multiply(kx,np.transpose(ky))
    # raise NotImplementedError(
    #     "`create_Gaussian_kernel_2D` function in `part1.py` needs to be implemented"
    # )

    ### END OF STUDENT CODE ####
    ############################

    return kernel


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.
    
    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    if image.ndim==3:
      (m,n,c)=image.shape
      (a,b)=filter.shape
      filtered_image=np.zeros(image.shape)
      padlengthx = int((a-1)/2)
      padlengthy = int((b-1)/2)
      #print(padlengthx,padlengthy)
      pre=np.pad(image,((padlengthx,padlengthx),(padlengthy,padlengthy),(0,0)),'constant',constant_values=(0,0))
      for k in range(c):
            p=pre[...,k]
            q=filtered_image[...,k]
            [rows,cols]=p.shape
            for i in range(padlengthx,rows-padlengthx):
                  for j in range(padlengthy,cols-padlengthy):
                        conv_input=p[i-padlengthx:i+padlengthx+1,j-padlengthy:j+padlengthy+1]
                        conv_output=conv_input*filter
                        conv_sum=np.sum(conv_output)
                        q[i-padlengthx,j-padlengthy]=conv_sum
    else:
      (m,n)=image.shape
      (a,b)=filter.shape
      filtered_image=np.zeros(image.shape)
      padlengthx = int((a-1)/2)
      padlengthy = int((b-1)/2)
      pre=np.pad(image,((padlengthx,padlengthx),(padlengthy,padlengthy)),'constant',constant_values=(0,0))
      p=pre
      q=filtered_image
      [rows,cols]=p.shape
      for i in range(padlengthx,rows-padlengthx):
            for j in range(padlengthy,cols-padlengthy):
                  conv_input=p[i-padlengthx:i+padlengthx+1,j-padlengthy:j+padlengthy+1]
                  conv_output=conv_input*filter
                  conv_sum=np.sum(conv_output)
                  q[i-padlengthx,j-padlengthy]=conv_sum

    
    # raise NotImplementedError(
    #     "`my_conv2d_numpy` function in `part1.py` needs to be implemented"
    # )

    ### END OF STUDENT CODE ####
    ############################
    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencies: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    low_frequencies=my_conv2d_numpy(image=image1,filter=filter)
    temp=my_conv2d_numpy(image=image2,filter=filter)
    high_frequencies=image2-temp
    hybrid_image=low_frequencies+high_frequencies
    hybrid_image=np.clip(hybrid_image,0,1)
    ############################
    ### TODO: YOUR CODE HERE ###

    # raise NotImplementedError(
    #     "`create_hybrid_image` function in `part1.py` needs to be implemented"
    # )

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
