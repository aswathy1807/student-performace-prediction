# Student Performance Prediction
# Step 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset

data = pd.read_csv("dataset/student_performance.csv")

# Step 3: Display the first 5 rows

print("First 5 rows of the dataset:")
print(data.head())

# Step 4: Basic information about the dataset
#EDA
print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nStatistical Summary:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Step 5: Data Preprocessing

# Convert placement status into numerical values
data["placement_status"] = data["placement_status"].map({
    "Not Placed": 0,
    "Placed": 1
})

print("\nPlacement Status after encoding:")
print(data["placement_status"].value_counts())

#checking encoded data

print("\nFirst 5 rows after preprocessing:")
print(data.head())

# Step 6: Exploratory Data Analysis

# Plot 1: Placement Status Distribution

plt.figure(figsize=(6, 4))

sns.countplot(x="placement_status", data=data)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

plt.xticks([0, 1], ["Not Placed", "Placed"])

plt.tight_layout()
plt.savefig("plots/placement_distribution.png")
plt.show()

# Plot 2: Study Hours vs Placement Status

plt.figure(figsize=(6, 4))

sns.boxplot(x="placement_status", y="study_hours", data=data)

plt.title("Study Hours vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Study Hours")

plt.xticks([0, 1], ["Not Placed", "Placed"])

plt.tight_layout()
plt.savefig("plots/study_hours_vs_placement.png")
plt.show()

# Plot 3: Attendance vs Placement Status

plt.figure(figsize=(6, 4))

sns.boxplot(x="placement_status", y="attendance", data=data)

plt.title("Attendance vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Attendance (%)")

plt.xticks([0, 1], ["Not Placed", "Placed"])

plt.tight_layout()
plt.savefig("plots/attendance_vs_placement.png")
plt.show()

# Plot 4: Previous Score vs Placement Status

plt.figure(figsize=(6, 4))

sns.boxplot(x="placement_status", y="previous_score", data=data)

plt.title("Previous Score vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Previous Score")

plt.xticks([0, 1], ["Not Placed", "Placed"])

plt.tight_layout()
plt.savefig("plots/previous_score_vs_placement.png")
plt.show()

# Plot 5: Exam Score vs Placement Status

plt.figure(figsize=(6, 4))

sns.boxplot(x="placement_status", y="exam_score", data=data)

plt.title("Exam Score vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Exam Score")

plt.xticks([0, 1], ["Not Placed", "Placed"])

plt.tight_layout()
plt.savefig("plots/exam_score_vs_placement.png")
plt.show()

# Plot 6: Correlation Heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# Step 7: Define Features and Target

X = data.drop(["placement_status", "exam_score"], axis=1)
y = data["placement_status"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Step 8: Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Step 9: Create the Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# Step 10: Make Predictions

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

# Step 11: Calculate Accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Step 12: Classification Report

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Placed", "Placed"]
))

# Step 13: Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()

# Step 14: Feature Importance (model accuracy 100%) means exam score is too important, so we will check feature importance to see which features are most important for predicting placement status.

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))

importance.plot(kind="bar")

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()



# Step 16: Analyze Exam Score and Placement

print("\nExam Score by Placement Status:")

print(
    data.groupby("placement_status")["exam_score"]
    .agg(["min", "max", "mean", "median"])
)

print("\nStudents with exam score <= 70:")
print(
    data[data["exam_score"] <= 70]["placement_status"]
    .value_counts()
)

print("\nStudents with exam score > 70:")
print(
    data[data["exam_score"] > 70]["placement_status"]
    .value_counts()
)

# Step 15: Visualize the Decision Tree

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree for Student Placement Prediction")
plt.tight_layout()
plt.savefig("plots/decision_tree.png", dpi=300)
plt.show()