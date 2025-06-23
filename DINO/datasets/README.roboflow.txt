
Acne04 Detection - v5 2024-05-09 12:20pm
==============================

This dataset was exported via roboflow.com on May 12, 2024 at 6:14 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3398 images.
Acne are annotated in COCO format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -13° to +13° horizontally and -13° to +13° vertically
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Salt and pepper noise was applied to 0.1 percent of pixels


