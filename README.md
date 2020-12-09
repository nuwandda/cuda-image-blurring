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
  ![Graph1](https://github.com/nuwandda/CUDA-image-blurring/blob/main/graph1.jpg "Graph 1") </br>
  Kernel block size in the graph above is 16x16 and blur size is 1. It spans three rows
( Row-1, Row, Row+1 ). As we can see in the graph, both shared and unshared memory implementations finish in near times at the first 3 images. However, after the image 3.ppm, the difference gets bigger. The process time of unshared memory increases much more than the process time of shared memory. The unshared memory approach process time increases almost x2 with each image. In contrast, the shared memory approach process increases almost x1.7. Nonetheless, shared memory processes have small computation time and this makes time consumption less. We can say that shared memory approach is averagely 10 times faster than unshared memory approach. </br>
 ![Graph2](https://github.com/nuwandda/CUDA-image-blurring/blob/main/graph2.jpg "Graph 2") </br>
 Blur size in the graph above is 1. It spans three rows ( Row-1, Row, Row+1 ). As we can see in the graph, increasing block size decreases the process times. For unshared memory approach, the block size changes affect process time more rather than shared memory approach. After 16x16 block size, the shared memory approach does not change much like the other memory type. For unshared memory approach, change from 4x4 to 32x32 reduces process time to half of the 4x4. For shared memory approach, change from 4x4 to 32x32 reduces process time for almost a third of the 4x4. Change gets really small after 8x8 for shared memory and we can say that the effect is not that much. Hence, block size increase has more effect on unshared memory approach.</br>
![Graph3](https://github.com/nuwandda/CUDA-image-blurring/blob/main/graph3.jpg "Graph 3") </br> 
Kernel block size in the graph above is 16x16. As we can. See in the graph, blur size changes affect unshared memory approach the most. Unshared memory process times increases for almost x4 with the first change. Then, it increases for almost x2.5. Shared memory process times increases for almost x4 with the first change. It is almost the same with the unshared one. Then, it increases for almost x2.6. The process time changes are very close to each other for both memory approaches. We can say that blur size changes have the same effect on memory approaches.</br>
