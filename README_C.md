
# Binary Classification Model

This project involves developing a machine learning model to classify data into one of two categories based on input features. The implementation is done using Python and standard machine learning libraries.

---

## Objective

To train and evaluate a binary classification model, and save the trained model for future use on similar data.

---

## Methodology and Assumptions

- A structured dataset with numeric features and a binary target column is used.
- The data is preprocessed using pandas and split into training and test sets.
- A classification model such as Random Forest is used for training.
- Model performance is evaluated using accuracy, precision, recall, F1 score, and confusion matrix.
- The model is saved using the `pickle` module.

Assumptions:

- The target variable contains two distinct classes (e.g., 0 and 1).
- Input data is either numeric or transformed to numeric format.
- All required libraries are listed in `requirements.txt`.

---

## Steps to Reproduce

### 1. Download the Files

Ensure the following files are in the same directory:
- `Binary_classification_model.ipynb`
- `Classification_final_model.pkl`
- `i_imputer.pkl`
- `s_selected_features.pkl`
- `README_C.md`
- `requirements.txt`

### 2. Open the Notebook

Use Jupyter Notebook to open and run the file:

```
jupyter notebook Binary_classification_model.ipynb
```

### 3. Run the Notebook

Execute all cells to:
- Load and preprocess the dataset
- Train the classification model
- Evaluate its performance
- Save the trained model and supporting files

---

### 4. Making Predictions with the Trained Model

After training and saving the model, predictions can be made on new data as shown below:

```python
import pickle

with open("Classification_final_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(new_data)
```

Replace `new_data` with your actual input data in the same format as the training features.

If preprocessing was done using saved tools (e.g., imputer or selected features), load them similarly:

```python
with open("i_imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("s_selected_features.pkl", "rb") as f:
    selector = pickle.load(f)
```

---

## Evaluation Metrics

- Accuracy: Proportion of correct predictions
- Precision: Ratio of correctly predicted positive observations to total predicted positives
- Recall: Ratio of correctly predicted positive observations to all actual positives
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: Summarizes the number of correct and incorrect predictions

---

## Files Included

- `Binary_classification_model.ipynb`: Jupyter notebook containing the implementation
- `Classification_final_model.pkl`: Saved model
- `i_imputer.pkl`: Saved imputer used during preprocessing
- `s_selected_features.pkl`: Saved selected features object
- `README_C.md`: Documentation file
- `requirements.txt`: List of Python libraries required

---

## Requirements

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```
