# Predictive Model Using Machine Learning

This project demonstrates a basic machine learning workflow with:

- Linear Regression for a simple regression task
- Decision Tree and Random Forest classifiers for a classification task
- Accuracy and confusion matrix evaluation
- ROC curve visualization for model comparison

## Project Structure

- `main.py` — trains the models and saves evaluation plots
- `requirements.txt` — Python dependencies
- `results/` — generated plots and CSV metrics

## Run the project

1. Open a terminal in this folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   python main.py
   ```

## Output

The script creates the following files inside the `results` folder:

- `linear_regression_plot.png`
- `confusion_matrices.png`
- `roc_curve.png`
- `model_metrics.csv`

These results provide a quick comparison of model performance and visual insight into predictive quality.
