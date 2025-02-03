# wheres_waldo
## Goal
The goal of this project is to find Waldo using a machine learning model trained on less than 50 images. The aim was to keep the dataset small to avoid overfitting and to ensure the model generalizes well to unseen images.

## Method

### 1. Scrape
- **Image Collection**: Collected 50 images of "Where's Waldo" that are roughly the same size and are already solved to avoid manual annotation.

### 2. Annotate
- **Custom Annotation Tool**: Developed a custom annotation tool to quickly annotate the images and create the dataset.

### 3. Augment
- **Data Augmentation**: Expanded the dataset from less than 50 images to over 2000 images using various augmentation techniques.

### 4. Train
- **Model Training**: Trained the model using Darknet, achieving a 100% Mean Average Precision (MAP).

### 5. Test
- **Sliding Window Script**: Implemented a sliding window script to detect Waldo in images of various sizes. The script successfully found Waldo in images that were not part of the training dataset.

## Result
The model successfully identified Waldo in images that were not included in the training dataset, demonstrating its ability to generalize well with a small dataset.
