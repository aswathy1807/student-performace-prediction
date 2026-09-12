# Student Performance Prediction

## Project Description

This project uses Machine Learning to predict whether a student is likely to be placed based on academic and behavioral factors.

The project uses a Decision Tree Classification algorithm to classify students into two categories:

- Placed
- Not Placed

## Dataset

The dataset contains 10,000 student records and includes information about:

- Study hours
- Attendance
- Sleep hours
- Internet usage
- Assignments completed
- Previous score
- Exam score
- Placement status

## Machine Learning Technique

This is a supervised machine learning classification problem.

### Algorithm

Decision Tree Classifier

The model was trained using the following features:

- Study hours
- Attendance
- Sleep hours
- Internet usage
- Assignments completed
- Previous score

The exam score was excluded from the model because the analysis showed that placement status was directly determined by exam score in this dataset.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset structure.
3. Checked for missing values.
4. Checked for duplicate records.
5. Encoded placement status:
   - Not Placed = 0
   - Placed = 1
6. Removed exam score from the model features to avoid target leakage.

The dataset contained no missing values or duplicate records.

## Exploratory Data Analysis

Several visualizations were created:

- Placement status distribution
- Study hours vs placement
- Attendance vs placement
- Previous score vs placement
- Exam score vs placement
- Correlation heatmap
- Feature importance
- Confusion matrix
- Decision tree visualization

## Model Training

The dataset was divided into:

- 80% training data
- 20% testing data

The Decision Tree Classifier was configured with a maximum depth of 5.

## Model Performance

The model achieved an accuracy of:

**87.05%**

### Classification Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Not Placed | 0.62 | 0.54 | 0.58 |
| Placed | 0.91 | 0.93 | 0.92 |

## Feature Importance

The most important features identified by the model were:

1. Study hours
2. Assignments completed
3. Previous score
4. Attendance
5. Sleep hours
6. Internet usage

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## How to Run

### 1. Clone the repository

git clone YOUR_REPOSITORY_URL

### 2. Create a virtual environment

python -m venv venv

### 3. Activate the virtual environment

For Git Bash:

source venv/Scripts/activate

### 4. Install the required libraries

pip install -r requirements.txt

### 5. Run the project

python student_performance.py