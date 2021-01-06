# Extracting tiles in multilevel gigapixel images

This is the source code described in the paper "Extracting tiles in multilevel gigapixel images".

### Abstract
In many image domains using multilevel gigapixel images, each image level may reveal different information. E.g., a histological image will reveal specific diagnostic information at different resolutions. By incorporating all levels in deep learning models, the accuracy is improved. Therefore, a sound method for finding and extracting tiles from multiple levels, i.e., resolutions, is essential both during training and prediction. This paper presents procedures that can be used to handle extraction and visualization of tiles in gigapixel images, and efficient implementation is openly provided. A set of parameters are defined, controlling the behavior of the procedures. The proposed procedures keep the correct scaling and physical location of the tiles on each level. The presented procedures work very well and have already been used to successfully extract several large ($>$2 million tiles) histological datasets, and can also be used on multilevel gigapixel images from other domains.


### Requirements

The code was built using the following Python packages:

python==3.6.7  
numpy==1.18.5  
opencv-python==4.4.0.42 
scikit-image==0.17.2  
scipy==1.4.1
pyvips==2.1.12

### Link to paper
TBA

### How to cite our work
TBA
