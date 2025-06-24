
Acne Detection on ACNE04 + Cross-Domain Acne Classification
1. Acne Detection on ACNE04
Dataset Introduction
ACNE04 is a publicly available dataset that focuses on acne vulgaris detection. It consists of 1450 high-resolution facial images, each accompanied by precise bounding box annotations for acne lesions. This dataset serves as an essential resource for developing and evaluating acne detection algorithms.
Dataset link: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/
Model Evaluation
We conducted acne detection experiments using three state-of-the-art models:

YOLOv5: A highly efficient object detection framework known for its speed and accuracy.
GitHub Repository: https://github.com/KeruZheng/YOLO-Acne-Detection-on-ACNE04-and-Cross-Domain-Acne-Classification_3
Contents: The repository includes detailed parameter settings for training, the full training code, and evaluation scripts. It allows users to reproduce our experimental results or fine-tune the model for their own applications.
Faster R-CNN: A classic two-stage object detection model that excels in precise object localization.
GitHub Repository: https://github.com/KeruZheng/Faster-RCNN-Acne-Detection-on-ACNE04-and-Cross-Domain-Acne-Classification_2
Contents: Users can find the customized parameter configurations, the training codebase optimized for acne detection, and instructions for running evaluations on the ACNE04 dataset.
DINO: A cutting-edge model leveraging self-supervised learning techniques for object detection.
GitHub Repository: https://github.com/KeruZheng/DINO-Acne-Detection-on-ACNE04-and-Cross-Domain-Acne-Classification
Contents: The repository houses the model's training parameters, the implementation code, and guidelines for conducting detection experiments on acne lesions.

Each repository provides a comprehensive set of resources, enabling researchers and developers to understand, replicate, and extend our work on acne detection using different model architectures.
2. Cross-Domain Acne Classification
### 2.2 Model Versions & Training Logic  


#### 2.2.1 v1: Facenet + Basic Multi-Augmentation (Baseline)  
- **Training Approach**:  
  - Backbone: `facenet-pytorch` (face feature extraction model).  
  - Augmentations: Color jitter, blur, rotation, flip, crop to simulate skin tone/pose variations.  
- **Goal**: Establish a baseline for comparing advanced techniques.  


#### 2.2.2 v2: Facenet + Advanced Augmentations  
- **Training Approach**:  
  - Inherits basic augmentations from v1, adds:  
    - HSV adjustment (color space manipulation)  
    - Cutout (occlusion simulation)  
    - Affine transformations (geometric distortion)  
- **Goal**: Enhance model robustness against texture shifts and real-world image corruptions.  


#### 2.2.3 v3: Facenet + CycleGAN Cross-Domain Style Transfer  
- **Training Pipeline**:  
  1. **DermNet Data Expansion**:  
     - Select 20 acne images from DermNet, expand to 1000 via rotation/slicing.  
  2. **CycleGAN Integration**:  
     - ACNE04 (target domain) learns DermNet's texture/lighting styles.  
  3. **Advanced Augmentations**:  
     - HSV adjustment, Cutout, Affine to simulate clinical scenarios.  
- **Key Innovation**: Cross-domain style transfer to bridge ACNE04 ↔ DermNet distribution gaps.  


### 2.3 Training Set Strategy  
- **Sample Ratio Experiments**:  
  - Tested 1:1 positive/negative ratio vs. real-world ratio (acne as minority class in DermNet).  
  - **Conclusion**: Real-world ratio outperformed in cross-domain tasks, better adapting to clinical data distributions (e.g., acne as 1/22 dermatological classes in DermNet).  


### 2.4 Evaluation Metrics  
- **Key Indicators**:  
  - mAP (Mean Average Precision)  
  - Precision-Recall curves  
  - IoU (Intersection over Union)  
- **Notable Result**:  
  v3 achieved the best cross-domain generalization, leveraging CycleGAN to improve feature transfer between datasets.  
![image](https://github.com/user-attachments/assets/500b87c9-2d58-4ddb-b2f5-a33595cdee06)


We trained five distinct versions of models for cross-domain acne classification, aiming to address the challenges of domain shift between different datasets. The key aspects of each version are as follows:

... (Your detailed description) ...

This README provides an overview of our research efforts in acne detection and cross-domain classification. For more in-depth information, please refer to the respective GitHub repositories and the associated documentatio
