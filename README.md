# CUDA-image-blurring
Implementation of an image blurring kernel. With and without the shared memory.
## System Specifications
• Azure NC6 </br>
• Cores: 6 </br>
• GPU: Tesla K80 </br>
• Memory: 56 GB </br>
• Disk: 380 GB SSD </br>
The Tesla K80 delivers 4992 CUDA cores with a dual-GPU design, up to 2.91 Teraflops of double- precision and up to 8.93 Teraflops of single-precision performance.

## Implementation Details
We will implement a simple blurring. This is also known as a box linear filter. The operation samples neighboring pixels of the input image and calculates an output image with average value. In this implementation, values that are located outside of the bounds of the input image are given zero. The implementation can work with any size of images. Blur box size can be changed before running the code.
To run the code smoothly, please follow command line arguments given below. </br>
  •argv[1]: IMAGENAME </br>
  •argv[2]: BLURTYPE </br>
BLURTYPE represents the memory type. To run the code with unshared memory, type ‘0’ for second argument. To run it with shared memory, type ‘1’ for second argument.
An example command can be like below.</br>
  •./imageBlur 1.ppm 0 </br>
  
  ## Results and Graphs
  The test images are checkboard images with different sizes. </br>
  • 1.ppm = 800x600 </br>
  • 2.ppm = 1600x1200 </br>
  • 3.ppm = 2400x1800 </br>
  • 4.ppm = 3200x2400 </br>
  • 5.ppm = 4000x3000 </br>
  • 6.ppm = 4800x3600 </br>
  • 7.ppm = 5600x4200 </br>
  • 8.ppm = 6400x4800 </br>
  • 9.ppm = 7200x5400 </br>
  • 10.ppm = 8000x6000 </br>
  
