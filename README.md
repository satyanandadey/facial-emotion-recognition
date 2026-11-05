# Facial Emotion Recognition – Geometry-Based Approach

## 1. Project Overview

This project implements a geometry-based facial emotion recognition system using facial landmarks.

The system detects facial landmarks, extracts geometric information from facial features, and uses machine learning algorithms to classify facial expressions into emotion categories.

The geometry branch uses the **dlib 68-point facial landmark predictor** to identify important facial points and extract geometric features for emotion classification.

## 2. Objectives

- Detect faces in facial expression images.
- Extract 68 facial landmarks using dlib.
- Calculate geometric features from facial landmarks.
- Train machine learning classifiers for emotion recognition.
- Evaluate model performance using classification metrics.
- Study the effect of landmark quality and different facial regions on classification performance.

## 3. Dataset

The project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset.

The dataset contains facial expression images belonging to seven emotion categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

The dataset is organized into training, validation, and test partitions.

## 4. Technologies and Libraries

- **Python** – Main programming language
- **dlib** – Face detection and 68-point facial landmark prediction
- **OpenCV** – Image processing
- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and CSV handling
- **Scikit-learn** – Machine learning models and evaluation
- **Matplotlib and Seaborn** – Data visualization
- **Joblib** – Model serialization and loading
- **Jupyter Notebook** – Development environment

## 5. Methodology

The geometry-based emotion recognition pipeline consists of the following steps:

### Step 1: Face Detection

The dlib frontal face detector identifies faces in the input images.

### Step 2: Facial Landmark Extraction

The pretrained dlib 68-point shape predictor identifies important facial landmarks around the eyes, eyebrows, nose, mouth, and jawline.

### Step 3: Geometric Feature Extraction

Geometric measurements are calculated using facial landmark coordinates.

These measurements represent facial structure and expression-related changes.

The final geometry-based feature representation contains 25 features.

### Step 4: Data Preparation

The extracted landmarks and geometric features are processed and organized into datasets for model training and evaluation.

### Step 5: Model Training

Machine learning classifiers, including Support Vector Machine (SVM) and Random Forest, are used for emotion classification.

### Step 6: Evaluation

The models are evaluated using metrics such as:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Classification reports
- Confusion matrices

Additional experiments investigate facial-region contributions, landmark quality, robustness, and inference latency.

## 6. Experimental Results

### Landmark Extraction

| Metric | Result |
|---|---:|
| Official test images | 7,178 |
| Successfully extracted test samples | 5,378 |
| Landmark extraction coverage | 74.92% |

### Random Forest Test Performance

The following results are calculated on successfully processed test images.

| Metric | Result |
|---|---:|
| Accuracy | 56.32% |
| Macro F1-score | 0.5180 |
| Weighted F1-score | 0.5531 |

When failed face detections are also counted as incorrect predictions, the overall test accuracy is approximately **42.2%**.

### Facial Region Ablation

The experiments investigate the contribution of different facial regions.

| Facial Region | Macro F1-score |
|---|---:|
| Lower face (mouth and jaw) | 0.4497 |
| Upper face | 0.3485 |
| Nose | 0.3154 |

The combined 25-feature geometry model achieved the best validation macro F1-score of **0.4947** among the evaluated feature configurations.

## 7. Project Structure

```text
MiniProject/
│
├── assets/
│   └── shape_predictor_68_face_landmarks.dat
│
├── data/
│   ├── FER-2013/
│   ├── manifests/
│   └── processed/
│
├── demo_images/
│
├── models/
│   └── dlib_geometry_model_expanded.joblib
│
├── results/
│   ├── audits/
│   ├── experiments/
│   └── predictions/
│
├── Geometry_Branch.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── shape_predictor_68_face_landmarks.dat.bz2
```

## 8. Installation and Setup

### Prerequisites

- Python 3.x
- pip
- Jupyter Notebook or Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone <https://github.com/satyanandadey/facial-emotion-recognition.git>
cd MiniProject
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** dlib may require additional installation or build dependencies depending on the operating system and Python environment.

### Step 3: Prepare the Landmark Predictor

The project uses the pretrained dlib 68-point facial landmark predictor.

The required file is:

```text
assets/shape_predictor_68_face_landmarks.dat
```

The notebook includes logic to download and extract the compressed predictor file when necessary.

### Step 4: Prepare the Dataset

Place the FER-2013 dataset in:

```text
data/FER-2013/
```

Ensure that its directory structure matches the paths expected by the notebook.

### Step 5: Run the Notebook

Open `Geometry_Branch.ipynb` in Jupyter Notebook or VS Code.

Run the notebook cells sequentially, starting from the first section.

Run it with `MiniProject/` as the working directory so that the relative project paths resolve correctly.

## 9. Saved Model

The trained geometry-based model is saved at:

```text
models/dlib_geometry_model_expanded.joblib
```

The saved model can be loaded using Joblib for subsequent predictions without retraining.

## 10. Limitations

- Landmark extraction does not succeed for every image.
- Facial expressions can be difficult to distinguish using geometric measurements alone.
- Variations in head pose, lighting, occlusion, and facial structure can affect landmark detection and classification.
- The geometry-based approach may not capture all the visual information available in facial images.

## 11. Conclusion

This project demonstrates a geometry-based approach to facial emotion recognition using dlib facial landmarks and machine learning.

The experiments show that geometric information from different facial regions contributes differently to emotion classification. The combined geometric feature representation performs better than the individual facial-region configurations evaluated.

Although the approach provides an interpretable way to study facial expressions, landmark detection failures and the limited representation of facial appearance remain important challenges.

---

**Project:** Facial Emotion Recognition  
**Approach:** Geometry-Based Facial Landmark Analysis  
**Dataset:** FER-2013  
**Framework:** Python and Scikit-learn