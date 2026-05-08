# HW3 - Visual Recognition using Deep Learning (Instance Segmentation of Medical Cell Images using Mask R-CNN)
Titouan GAGNEUX, ID : 114550830

---

# Introduction

This project was completed for Homework 3 of the *Visual Recognition using Deep Learning* course at NYCU. The objective of this assignment is to solve an instance segmentation task on medical images containing different types of cells. The goal is to detect and segment each individual cell instance belonging to four possible classes.

The challenge dataset contains colored microscopy images associated with segmentation masks stored in `.tif` format. Each unique pixel value inside a mask corresponds to a different object instance. The evaluation metric used in the competition is AP50.

For this homework, a Mask R-CNN based pipeline was implemented using PyTorch and Torchvision. The complete workflow includes dataset preprocessing, mask extraction, model training, inference generation, post-processing into COCO RLE format, and submission generation for the Codabench competition.

---

# Environment Setup

## Requirements

Recommended Python version: **Python 3.9+**

Install the required dependencies with:
```bash
pip install -r requirements.txt
```

Main libraries used:

- torch
- torchvision
- numpy
- matplotlib
- opencv-python
- pycocotools
- tifffile
- tqdm
- scikit-learn

---

# Method

## Data Preprocessing

The dataset contains raw microscopy images and several mask files corresponding to the different cell classes. Each mask file stores instance IDs where every unique pixel value represents a separate cell instance.

The preprocessing pipeline performs the following operations:

- Load `.tif` images and masks
- Extract every unique instance from masks
- Generate binary masks for each object
- Compute bounding boxes automatically
- Assign the correct class label
- Convert data into the format expected by Mask R-CNN

To improve the model generalization capability, several augmentations were applied during training:

- Random Horizontal Flip
- Random Vertical Flip
- Color Jitter
- Random Rotation

These augmentations help the model become more robust to variations in orientation, illumination, and texture.

---

## Model Architecture

The implemented model is based on **Mask R-CNN** with a **ResNet-50 FPN backbone** pretrained on ImageNet.

The architecture contains:

- Backbone: ResNet-50
- Neck: Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- Bounding Box Head
- Classification Head
- Segmentation Mask Head

The final prediction head was modified to support the four target classes from the dataset.

Pretrained weights were used to accelerate convergence and improve segmentation performance.

---

## Training Configuration

Main hyperparameters used during training:

- Batch size: 2
- Optimizer: SGD
- Learning rate: 0.005
- Momentum: 0.9
- Weight decay: 0.0005
- Scheduler: StepLR
- Epochs: 10

The model was trained using GPU acceleration on Google Colab.

A validation split was used to evaluate the model locally before submission. COCOEval was also used to avoid blind submissions and estimate AP50 performance during development.

---

# Inference Pipeline

Inference on the test set follows several steps:

1. Load trained checkpoint
2. Run forward prediction on test images
3. Extract masks, labels, bounding boxes, and confidence scores
4. Filter low-confidence predictions
5. Convert masks into RLE encoding
6. Generate the final `test-results.json` submission file

The generated JSON file follows the COCO-style instance segmentation format required by the competition platform.

---

# Usage

## Training

Run the training script to start model training.

```python
python train.py
```

---

## Inference

Run the inference script to generate predictions on the test set.
```python
python inference.py
```

This generates the `test-results.json` file which can then be compressed into a `.zip` file and submitted to Codabench.

---

# Results

## Quantitative Results

The final model achieved approximately:
- Train loss: 0.6798
- Val loss:   0.7911

| Metric | Score |
| Public AP50 | 0.31 |
| Validation AP50 | ~0.30 |

The obtained score is above the weak baseline provided in the homework instructions and approaches the strong baseline range.

---

## Qualitative Results

The model is capable of detecting multiple cell instances and generating accurate segmentation masks for most visible objects.

The predictions generally show:

- Good localization of cell regions
- Accurate instance separation
- Robust predictions on dense areas
- Correct detection of multiple classes

Some failure cases remain on:

- Very small cells
- Overlapping instances
- Blurry regions
- Highly crowded images

<img width="1487" height="512" alt="image" src="https://github.com/user-attachments/assets/1ce9384a-cdb8-47af-8f1f-a70405f644b1" />
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/2f0d37c6-85ce-482f-a004-1a7d2d7f6435" />


---

# Additional Experiments

## Data Augmentation

Different augmentation strategies were tested to improve robustness. In particular, horizontal and vertical flips helped the model generalize better to different cell orientations.

Color augmentation also slightly improved validation performance by reducing sensitivity to illumination differences between images.

## Plot algorithm

An additional visualization notebook was implemented to better inspect the dataset and the model predictions.

This notebook allows:

- Displaying training images together with their associated segmentation masks
- Visualizing each individual cell instance separately
- Inspecting predicted masks generated by the model during inference
- Comparing raw images and predicted instances for qualitative evaluation

These visualizations were particularly useful for validating the preprocessing pipeline, checking annotation consistency, and analyzing model behavior on difficult samples.

---

## Confidence Threshold Tuning

Different confidence thresholds were tested during inference.

A lower threshold increased recall but produced more false positives, while a higher threshold generated cleaner masks but missed small objects. A compromise threshold was selected to maximize AP50 performance.

---

## Pretrained Backbone

Using pretrained ImageNet weights significantly improved convergence speed and final performance compared to training from scratch.

Without pretrained weights, the model struggled to learn meaningful features due to the relatively small dataset size.

---

# Conclusion

This project implemented a complete instance segmentation pipeline using Mask R-CNN for medical cell segmentation.

The final system successfully performs:

- Instance detection
- Cell classification
- Pixel-wise segmentation
- COCO-format prediction generation

Despite the limited dataset size, the model achieved competitive AP50 performance through transfer learning, data augmentation, and careful preprocessing.

Future improvements could include:

- Larger backbones
- Better augmentation strategies
- Transformer-based segmentation models
- Advanced loss functions
- Ensemble methods

---

# References

- Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick.  
  *Mask R-CNN*, ICCV 2017.

- Torchvision Documentation

- COCO Evaluation Documentation

```bash
pip install -r requirements.txt
