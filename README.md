# Parameterized Extraction of Tiles in Multilevel Gigapixel Images

This is the source code described in the paper "Parameterized Extraction of Tiles in Multilevel Gigapixel Images".

### Abstract
In many image domains using multilevel gigapixel images, each image level may reveal different information. E.g., a histological image will reveal specific diagnostic information at different resolutions. By incorporating all levels in deep learning models, the accuracy can be improved. It is necessary to extract tiles from the image since it is intractable to process an entire gigapixel image at full resolution at once. Therefore, a sound method for finding and extracting tiles from multiple levels is essential both during training and prediction. In this paper, we have presented a method to parameterize and automate the task of extracting tiles from different scales with a region of interest (ROI) defined at one of the scales. The proposed method makes it easy to extract different datasets from the same group of gigapixel images with different choices of parameters, and it is reproducible and easy to describe by reporting the parameters. The method is suitable for many image domains and is demonstrated here with different parameter settings using histological images from urinary bladder cancer. An efficient implementation of the method is openly provided via Github.

![alt text](images/wsi_example1_v5.png?raw=true)

### Requirements

The code was built using the following Python packages:

python==3.6.7  
numpy==1.18.5  
opencv-python==4.4.0.42  
scikit-image==0.17.2  
scipy==1.4.1  
pyvips==2.1.12
opencv-python==4.1.0.25

### Link to paper
TBA

### How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper if you use it in your research.
```
R. Wetteland, K. Engan, and T. Eftestøl, “Parameterized extraction of tiles in multilevel gigapixel images,” Accepted for publication, 12th International Symposium on Image and Signal Processing and Analysis (ISPA 2021). IEEE, 2021.
```
