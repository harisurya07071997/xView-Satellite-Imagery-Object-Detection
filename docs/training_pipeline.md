
## Overview

This section provides an overview of the training pipeline, including dataset preparation, preprocessing, model training, and key engineering decisions involved in the workflow.

---

## Dataset Analysis

The xView dataset consists of high-resolution GeoTIFF imagery with annotations stored in large-scale GeoJSON files. Understanding the dataset characteristics is a critical step for designing effective preprocessing pipelines, selecting suitable model architectures, and addressing key challenges.


### Extreme Class Imbalance

![Class Imbalance Analysis](docs/images/instance_per_category.png)

The dataset exhibits a highly skewed distribution across both object instance counts and image-level coverage.

- Dominant classes such as **Building** (~316K instances across 727 images) and **Small Car** (~211K instances across 692 images) appear frequently and are widely distributed throughout the dataset.
- In contrast, rare classes such as **Railway Vehicle** (17 instances across 6 images) and **Straddle Carrier** (64 instances across 17 images) appear only in a limited number of images.

This indicates that a small subset of classes dominates both the spatial distribution and overall annotation density within the dataset.

#### Key Observations

- Severe class imbalance can lead to biased model learning toward dominant categories.
- Rare classes are more prone to underfitting and poor generalization.
- The imbalance necessitates careful sampling, augmentation, and training strategies.


### Variation in Object Size, Shape, and Scale

![Object Size Variation](docs/images/object_size_variation.png)

The dataset contains significant variation in bounding box dimensions, object shapes, and spatial scales.

Most bounding boxes are concentrated in the lower range of width and height values, indicating that the dataset is heavily dominated by small objects. A smaller number of outliers extend to very large dimensions, representing large-scale structures and infrastructure.

The width-height distribution also indicates the presence of both square-like and elongated rectangular objects.

#### Key Observations

- The concentration of bounding boxes near the origin highlights a strong **small-object detection challenge**.
- The coexistence of extremely small and very large objects introduces a significant **multi-scale detection problem**.
- Large variation in object dimensions motivates the use of:
  - image tiling strategies,
  - multi-scale training,
  - and anchor box optimization.
- The distribution of object shapes provides useful insights for anchor tuning and model configuration.


### Dataset Analysis Workflow

Run the notebook below to perform detailed exploratory data analysis (EDA) and convert the original GeoJSON annotations into a structured CSV format.

```text
training/yolo_src/dataset/Dataset_Analysis.ipynb
```

#### Generated Outputs

```text
training/yolo_src/dataset/Original/train_labels/xview_labels.csv
```

---

## Dataset Preparation

This section describes the preprocessing strategies used to prepare the xView dataset for robust and scalable model training.

The xView dataset contains extremely high-resolution satellite imagery with significant variation in object size, object density, and spatial distribution. Directly resizing large images for training can lead to severe information loss, particularly for tiny objects such as vehicles and small infrastructure components.

To address these challenges, the preprocessing pipeline focuses on:
- preserving fine-grained spatial information,
- improving small-object visibility,
- preventing train-validation data leakage,
- and ensuring balanced dataset representation during training.


### Stratified Train-Validation Split

A custom image-level stratified splitting strategy was implemented instead of a standard random split.

The split strategy:
- groups annotations at the image level to prevent data leakage,
- performs stratification using image-level object statistics,
- and ensures validation coverage across all object classes.

The preprocessing pipeline categorizes images based on average bounding box area distribution, helping maintain balanced representation of object scales across both training and validation datasets.

#### Why This Strategy?

Standard random splitting can lead to:
- poor representation of rare classes,
- scale distribution mismatch between train and validation sets,
- and biased evaluation metrics.

The custom stratified splitting approach improves:
- dataset consistency,
- evaluation reliability,
- and model generalization across varying object scales.


### Tile-Based Image Preprocessing

The original xView images are extremely large and cannot be efficiently processed directly during GPU training due to memory limitations.

Instead of resizing entire images — which can significantly degrade small-object visibility — a tile-based preprocessing strategy was implemented.

The preprocessing pipeline:
- divides large satellite images into smaller overlapping tiles,
- preserves spatial resolution and fine-grained object details,
- and applies boundary padding to handle edge regions consistently.

#### Why Tiling Instead of Resizing?

Direct image resizing introduces:
- loss of spatial detail,
- reduced visibility of tiny objects,
- and degraded small-object detection performance.

The tile-based strategy helps:
- preserve object resolution,
- improve GPU memory efficiency,
- increase small-object representation during training,
- and enable scalable processing of very large satellite imagery.

Overlapping tiles also help reduce boundary artifacts and improve object continuity across neighboring regions.


### Annotation Export

After preprocessing:
- tiled images are generated for training and validation,
- annotations are converted into YOLO/COCO-compatible formats,
- and dataset artifacts are organized for model training.


### Dataset Preparation Workflow

Run the preprocessing notebook below to:
- perform stratified train-validation splitting,
- apply tile-based preprocessing,
- and export annotations into YOLO/COCO format.

```text
training/yolo_src/dataset/Dataset_Preparation.ipynb
```

### Generated Outputs

- Processed train/validation image tiles
- YOLO/COCO-format annotation files
- Structured dataset directories for training

---


## Training Configuration

This section describes the training configuration used for model development, including hyperparameter settings, augmentation strategies, experiment modes, and configurable training workflows.

The training pipeline was designed to support:
- scalable experimentation,
- reproducible training,
- hyperparameter tuning,
- and flexible fine-tuning workflows for satellite imagery object detection.

All major training parameters are managed through a centralized configuration file.


### Configuration File

The primary training configuration is managed through:

```text
training/yolo_src/model/Config.yaml
```

The configuration file contains:
- dataset paths,
- model checkpoints,
- image size configuration,
- batch size settings,
- optimizer parameters,
- augmentation settings,
- training epochs,
- and experiment-related configurations.

Centralizing training parameters into a single configuration file improves:
- reproducibility,
- experiment management,
- and easier hyperparameter tuning.


### Hyperparameter Configuration

Several training hyperparameters were tuned to improve learning stability and small-object detection performance on the xView dataset.

The configuration includes:
- learning rate,
- batch size,
- image size,
- optimizer configuration,
- weight decay,
- confidence thresholds,
- and augmentation probabilities.

Due to the extreme scale variation and class imbalance present in satellite imagery datasets, careful hyperparameter selection was important for:
- stable convergence,
- better feature learning,
- and improved generalization across object scales.


### Augmentation Strategy

Data augmentation was used extensively to improve model robustness and generalization.

The augmentation pipeline includes:
- horizontal and vertical flipping,
- scaling,
- rotation-based transformations,
- cutmix and copypaste augmentation,
- and color-space augmentations.

These augmentations help the model become more robust to:
- varying object orientations,
- scale changes,
- environmental conditions,
- and viewpoint variations commonly present in aerial imagery.

Augmentation is particularly important for satellite imagery because:
- objects appear at arbitrary orientations,
- object sizes vary significantly,
- and training samples for certain classes are limited.


### Training Modes

The training pipeline supports multiple execution modes for flexible experimentation and model development.

#### Standard Training

Used for regular model training from initialization or pretrained weights.

```bash
python trainer.py --do_training
```


#### Hyperparameter Tuning

Used for running hyperparameter optimization experiments and WandB sweeps.

```bash
python trainer.py --do_tuning
```

This mode enables:
- automated experiment tracking,
- parameter search,
- and comparative evaluation across training runs.


#### Resume Training from Checkpoint

Used for resuming interrupted training sessions or continuing long-running experiments.

```bash
python trainer.py --do_resume_from_checkpoint
```

This is particularly useful for:
- large-scale training workloads,
- long GPU sessions,
- and iterative experimentation.


### Experiment Tracking

Training experiments were tracked using Weights & Biases (WandB) for:
- experiment monitoring,
- hyperparameter comparison,
- metric visualization,
- and training reproducibility.

Tracked metrics include:
- training loss,
- validation loss,
- mAP metrics,
- precision,
- recall,
- and learning curves.

---

## Evaluation

This section summarizes the model evaluation strategy and validation performance on the xView dataset.

The evaluation pipeline focuses on:
- object detection accuracy,
- localization quality,
- and robustness across varying object scales.

Validation was performed on the stratified validation split generated during dataset preparation.


### Evaluation Metrics

The following metrics were used to evaluate model performance:

| Metric | Description |
|---|---|
| mAP@50 | Mean Average Precision at IoU threshold 0.50 |
| mAP@50-95 | Mean Average Precision averaged across IoU thresholds |
| Precision | Ratio of correct positive detections |
| Recall | Ability to detect ground-truth objects |
| F1 Score | Balance between precision and recall |

These metrics provide insights into:
- detection accuracy,
- localization performance,
- and overall model generalization.


### Validation Results

| Metric | Score |
|---|---|
| mAP@50 | 0.22 |
| mAP@50-95 | 0.13 |
| Precision | 0.69 |
| Recall | 0.64 |


### Validation Predictions

#### Sample Prediction 1

![Validation Result 1](docs/images/sample_prediction_1.png)


#### Sample Prediction 2

![Validation Result 2](docs/images/sample_prediction_2.png)


### Performance Observations

#### Confusion Matrix

![Confusion matrix analysis](docs/images/confusion_matrix.png)


The confusion matrix indicates that the model has learned strong coarse-level semantic separation across major object groups such as Aircraft, Vehicles, Ships, Railway objects, and Structures. However, the model still struggles with fine-grained classification between visually similar subclasses.

The severe class imbalance in the xView dataset also introduces bias toward dominant categories, resulting in:
- weaker performance on rare classes,
- increased false negatives for minority categories,
- and misclassification toward frequently occurring classes.

From the evaluation metrics:
- **mAP@0.5 remains relatively strong**, indicating good coarse object detection capability.
- **mAP@[0.5:0.95] shows degradation**, reflecting challenges in precise localization and fine-grained classification.

Overall, the model demonstrates:
-  Strong coarse-level generalization
-  Moderate localization reliability
-  Limited fine-grained class discrimination


### Benchmarking

Inference benchmarking was performed to measure:
- preprocessing latency,
- model inference time,
- postprocessing overhead,
- and total pipeline latency.

Detailed benchmarking results are available in:

```text
docs/benchmarks.md
```


### Evaluation Workflow

Run the evaluation notebook below to generate validation predictions and performance metrics:

```text
training/yolo_src/evaluation/Evaluate.ipynb
```

Generated outputs:
- validation predictions
- evaluation metrics
- benchmark visualizations
- performance analysis artifacts


