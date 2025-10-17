# Infant Audio Classifier: PCA and Logistic Regression from Scratch

## Project Overview
This project implements an **Infant Audio Classifier** using a subset of the "Baby Cry Pattern Archive" dataset from Kaggle. The goal is to classify infant audio into three categories: **silence**, **noise**, and **laughter**.

The project demonstrates the impact of **feature scaling**, **dimensionality reduction**, and **manual implementation of machine learning algorithms**. 

Two main Jupyter Notebooks are included:

1. **Feature Extraction Notebook** (`notebooks/01_feature_extraction.ipynb`):  
   - Extracts audio features from raw audio files.  
   - Original feature set had ~1.3 million features per sample.  
   - Features were reduced to **9 meaningful features**:
     - Audio_Mean, Audio_Std  
     - MFCC_Mean, MFCC_Std  
     - Spec_Mean, Spec_Std  
     - Chroma_Mean, Chroma_Std  
     - Tempo  
   - Processed features are saved into a CSV file for modeling.

2. **PCA and Logistic Regression Notebook** (`notebooks/02_pca_logistic_regression.ipynb`):  
   - Implements **Logistic Regression** and **PCA** from scratch, without using scikit-learn implementations.  
   - Demonstrates:
     - Impact of feature scaling (accuracy improved from ~0.83 → 0.95)
     - Effect of PCA on dimensionality reduction (9 → 8 features)
     - Comparison between manual implementation and scikit-learn results
   - Plots accuracy vs number of principal components.

## Dataset
The dataset is available on Kaggle: [Baby Cry Pattern Archive](https://www.kaggle.com/datasets/mennaahmed23/baby-cry/data).  
Only the following three categories were used:

- **silence** (108 audio samples)  
- **noise** (108 audio samples)  
- **laugh** (108 audio samples)  

> **Note:** The full dataset is **not included** in this repository. Please download it from the Kaggle link above. A small subset of 3 audio files per class is included for testing purposes.

## Folder Structure
```
InfantAudioClassifier/
├── data/
│   ├── laugh
│   ├── noise
│   └── silence
├── features/
│   └── features_with_labels.csv
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   └── 02_pca_logistic_regression.ipynb
├── LICENSE
├── README.md
└── requirements.txt
````

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Zahra-Ghaffary/Infant-Audio-Classifier.git
cd Infant-Audio-Classifier
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebooks in Jupyter or VSCode to reproduce results.

## Results Highlights

* **Baseline Logistic Regression (no scaling, no PCA):** ~83% accuracy

* **Sklearn Logistic Regression (scaled, no PCA):** ~95% accuracy

* **Manual Logistic Regression (scaled, no PCA):** ~95% accuracy

* **Sklearn Logistic Regression with PCA (scaled, 99% variance):** ~97% accuracy

* **Manual Logistic Regression with Manual PCA (scaled, 8 features):** ~95% accuracy

* Accuracy is stable for 7–9 principal components; drops significantly below 6 components.

## Features

* Manual implementation of **Gradient Descent** for Logistic Regression
* Manual implementation of **PCA** using covariance matrix and eigen decomposition
* Comparison with **scikit-learn** implementations to validate correctness
* Demonstrated accuracy improvement and effect of dimensionality reduction

## References

* Original dataset: [Baby Cry Pattern Archive on Kaggle](https://www.kaggle.com/datasets/mennaahmed23/baby-cry/data)
* Feature extraction code adapted from: [Baby Cry Detection Capstone Project](https://github.com/raviatkumar/Baby-Cry-Detection-Audio-data/tree/main)
