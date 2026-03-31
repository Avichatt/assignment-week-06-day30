# SUV Purchase Prediction - Logistic Regression

This repository contains my solution for the **Week 06 · Day 30** assignment: "Logistic Regression & End-to-End ML Pipeline (Pandas, Preprocessing, Model Training, Evaluation)" using the SUV Purchase Dataset.

## Project Structure

*   `part-a.py`: Contains the core Machine Learning pipeline including Data Loading, Data Preprocessing (handling missing values, encoding categorical variables), Train-Test Split (80/20), Feature Scaling, and Model Training with Logistic Regression.
*   `part-b.py`: Extends part A by implementing Model Evaluation (calculating Accuracy and generating a Confusion Matrix). It also visualizes the decision boundary on the dataset and experiments with different Train-Test splits (70/30, 75/25) for performance comparison.
*   `part-c.md`: Contains theoretically framed interview questions regarding Logistic Regression, Confusion Matrices, and the corresponding code for train-test splitting and feature scaling.
*   `part-d.md`: Documents an AI-Augmented Task exploring an AI-generated prompt on the SUV dataset and evaluating its correctness and completeness.
*   `suv_data.csv`: A representative sample of the SUV dataset structured to make the code reproducible and executable locally out-of-the-box.

## Pre-requisites

Make sure you have Python installed, and install the required modules by running:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to Run

1.  Clone this repository.
2.  Navigate to the directory containing the project.
3.  Run the conceptual script:
    ```bash
    python part-a.py
    ```
4.  Run the evaluation script to generate metric responses and decision boundary plots:
    ```bash
    python part-b.py
    ```

## Dataset Reference

Original Kaggle Dataset Reference: [SUV Purchase Dataset](https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset)
*(Note: `suv_data.csv` is included to demonstrate code functionality directly).*
