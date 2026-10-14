# dm_project

# Food Waste Analysis and Prediction Using Machine Learning

## Project Overview

This project uses machine-learning techniques to analyze food-waste data, predict the quantity of food wasted, classify food-waste amounts into categories, and discover patterns among similar records.

The workflow includes data inspection and preprocessing, regression, classification, clustering, model evaluation, and visualization. The implementation is provided in a Jupyter Notebook and is suitable for running in Google Colab or a local Jupyter environment.

## Objectives

- Explore the food-waste dataset and inspect its structure and missing values.
- Predict the quantity of food wasted using regression models.
- Classify food-waste records into **Low**, **Medium**, and **High** categories.
- Group similar food-waste records using unsupervised clustering.
- Compare model performance using appropriate evaluation metrics.
- Visualize predictions, classification results, clusters, and potential outliers.

## Machine-Learning Techniques

### 1. Regression

Regression predicts the numerical value of `Wastage Food Amount`.

Models used:
- **XGBoost Regressor**
- **Linear Regression**
- **Random Forest Regressor**

Evaluation metrics:
- **MAE (Mean Absolute Error):** average absolute prediction error.
- **RMSE (Root Mean Squared Error):** prediction error measured in the target variable's units, with larger errors penalized more heavily.
- **R² Score:** measures how well the model explains variation in the target.

The notebook also plots actual values against predicted values and compares the regression models.

### 2. Classification

The notebook converts the numerical `Wastage Food Amount` values into **Low**, **Medium**, and **High** categories using quantile-based grouping.

Models used:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **K-Nearest Neighbors (KNN) Classifier**

Evaluation and visualization:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

These metrics help compare the models and identify which categories are being predicted correctly. The category boundaries are based on the dataset distribution, so they should not automatically be interpreted as universal food-waste thresholds.

### 3. Clustering

Clustering groups records based on similarity without using predefined category labels.

Algorithms used:
- **K-Means**
- **Hierarchical Clustering**
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)

The notebook compares clustering results, visualizes clusters after reducing the feature space to two dimensions, and identifies records marked as noise by DBSCAN (`-1` labels).

## Workflow

1. Load the food-waste dataset.
2. Inspect the dataset shape, columns, data types, sample rows, and missing values.
3. Prepare features and target variables for each machine-learning task.
4. Handle missing values and encode categorical features.
5. Split the data into training and testing sets where appropriate.
6. Train regression and classification models.
7. Evaluate and compare model performance.
8. Apply clustering algorithms to discover groups and potential outliers.
9. Visualize and interpret the results.

## Repository Structure

```text
.
├── dm_capstone_project.ipynb
├── food_wastage_data.csv   # Add the dataset locally; it may not be included in this repository
└── README.md
```

## Dataset

The notebook loads a CSV file named `food_wastage_data.csv`. Make sure you have permission to use and share your dataset before uploading it to GitHub.

In Google Colab, upload the CSV file and ensure the path used in the notebook points to it. The current notebook uses:

```python
df = pd.read_csv('/content/food_wastage_data.csv')
```

If you run the notebook locally, update the path to the location of the CSV file on your computer, for example:

```python
df = pd.read_csv('food_wastage_data.csv')
```

The regression workflow uses `Wastage Food Amount` as its numerical target. The classification workflow derives Low/Medium/High categories from this amount. Clustering uses the prepared features to find similar records without relying on those category labels.

## Requirements

The notebook uses Python and common data-science and machine-learning libraries, including:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- Jupyter Notebook or Google Colab

Install the main libraries with:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

Depending on the notebook environment, some libraries may already be installed.

## How to Run

### Option A: Google Colab

1. Upload `dm_capstone_project.ipynb` to Google Colab.
2. Upload `food_wastage_data.csv` to the Colab session.
3. Confirm that the CSV path in the notebook is correct.
4. Run the notebook cells in order, from top to bottom.
5. Review the printed evaluation metrics and generated plots.

### Option B: Local Jupyter Notebook

1. Clone or download this repository.
2. Place `food_wastage_data.csv` in the project directory, or update the dataset path in the notebook.
3. Install the requirements.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `dm_capstone_project.ipynb` and run all cells in order.

## Results and Interpretation

The notebook produces model evaluation metrics and visualizations for comparing the algorithms. Use the outputs generated by your own notebook run to report the actual best-performing model.

- For **regression**, lower MAE and RMSE generally indicate smaller errors; a higher R² is generally preferable when comparing models on the same test split.
- For **classification**, consider accuracy together with precision, recall, and F1-score, particularly if category sizes differ.
- For **clustering**, compare the group assignments and visualizations carefully. Cluster IDs are labels, not rankings, and DBSCAN may identify noise points.

**Note:** Results depend on the dataset, preprocessing, and model parameters. No performance scores are claimed here because they should be taken from the notebook's actual execution outputs.

## Future Improvements

- Add a data dictionary describing each dataset column.
- Validate the models on additional data and use cross-validation where appropriate.
- Tune model hyperparameters systematically.
- Add clearer plots and a consolidated summary of the best model for each task.
- Investigate whether cluster patterns are meaningful and useful for food-waste reduction.
- Add reproducible experiment settings and document the dataset source.

## Website Link
https://dmcapstoneproject-2026.streamlit.app 
