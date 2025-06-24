
# Acne Detection on ACNE04 + Cross-Domain Acne Classification

## Acne Detection on ACNE04
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


## Cross-Domain Acne Classification
### odel Versions & Training Logic  

![image](https://github.com/user-attachments/assets/500b87c9-2d58-4ddb-b2f5-a33595cdee06)
#### v1: Facenet + Basic Multi-Augmentation (Baseline)  
- **Training Approach**:  
  - Backbone: `facenet-pytorch` (face feature extraction model).  
  - Augmentations: Color jitter, blur, rotation, flip, crop to simulate skin tone/pose variations.  
- **Goal**: Establish a baseline for comparing advanced techniques.  


#### v2: Facenet + Advanced Augmentations  
- **Training Approach**:  
  - Inherits basic augmentations from v1, adds:  
    - HSV adjustment (color space manipulation)  
    - Cutout (occlusion simulation)  
    - Affine transformations (geometric distortion)  
- **Goal**: Enhance model robustness against texture shifts and real-world image corruptions.  


#### v3: Facenet + CycleGAN Cross-Domain Style Transfer  
- **Training Pipeline**:  
  1. **DermNet Data Expansion**:  
     - Select 20 acne images from DermNet, expand to 1000 via rotation/slicing.  
  2. **CycleGAN Integration**:  
     - ACNE04 (target domain) learns DermNet's texture/lighting styles.  
  3. **Advanced Augmentations**:  
     - HSV adjustment, Cutout, Affine to simulate clinical scenarios.  
- **Key Innovation**: Cross-domain style transfer to bridge ACNE04 ↔ DermNet distribution gaps.  


### Training Set Strategy  
- **Sample Ratio Experiments**:  
  - Tested 1:1 positive/negative ratio vs. real-world ratio (acne as minority class in DermNet).  
  - **Conclusion**: Real-world ratio outperformed in cross-domain tasks, better adapting to clinical data distributions (e.g., acne as 1/22 dermatological classes in DermNet).  


###  Evaluation Metrics  
- **Key Indicators**:  
  - mAP (Mean Average Precision)  
  - Precision-Recall curves  
  - IoU (Intersection over Union)  
- **Notable Result**:  
  v3 achieved the best cross-domain generalization, leveraging CycleGAN to improve feature transfer between datasets.  

![image](https://github.com/user-attachments/assets/1662f7be-ddd7-448b-a18f-6e9398f2abb3)
 🔍 Failure Analysis & Model Comparison
❌ v1 & v2: Why They Fail
The v1 and v2 models demonstrate zero Precision, Recall, and F1 Score, primarily due to:

Extreme Class Imbalance
The datasets contain a large number of negative samples, causing the models to heavily bias towards negative predictions. This results in a misleadingly high Accuracy, as only negative cases are correctly classified — a phenomenon known as the "High Accuracy Illusion."

Insufficient Feature Learning
Simple augmentations fail to address the nuanced and subtle nature of acne lesions. Characteristics such as fuzzy lesion edges and low contrast are not well captured, leading to complete misclassification of positive samples.

✅ v3: A Robust Solution
The v3 model leverages CycleGAN-based cross-domain style transfer and advanced augmentation techniques, leading to significant performance improvements in both diversity and feature discrimination:

🌈 Data Diversity
By incorporating data from DermNet, v3 enriches the stylistic diversity missing in ACNE04.

This enables the model to learn from varied clinical textures, such as acne under different lighting or skin tones.

Additionally, the dataset becomes more balanced, reducing bias toward negative samples.

🧠  Feature Discrimination
Augmentations like HSV adjustment, Cutout, and Affine transforms force the model to focus on key acne features (e.g., shape, texture).

These transformations reduce reliance on irrelevant cues like background and lighting.

As a result, v3 achieves higher AUROC (0.6206), outperforming both v1 and v2.

📌 Recommendation
The v1/v2 models suffer from the "high Accuracy illusion" and are ineffective for practical acne detection.

In contrast, v3 successfully addresses both data imbalance and weak feature learning through cross-domain data fusion and targeted augmentations.

✅ We strongly recommend adopting v3 for its superior generalization performance and clinical applicability in real-world dermatological scenarios.




This README provides an overview of our research efforts in acne detection and cross-domain classification. For more in-depth information, please refer to the respective GitHub repositories and the associated documentatio
