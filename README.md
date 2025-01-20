# MNIST Digit Classification

This project explores classification techniques on the MNIST dataset, focusing on digits 0 and 8. The workflow includes data preprocessing, feature extraction, and implementation of both manual thresholding and logistic regression.

## Key Features
- **Dataset**: Used the MNIST dataset containing 70,000 grayscale images of digits (28x28 pixels).
- **Thresholding**: Derived custom features based on the mean pixel value of the center region (4x4) for binary classification.
- **Logistic Regression**: Built a logistic regression model for improved classification accuracy.
- **Visualization**: Generated histograms, scatter plots, and decision boundary visualizations to interpret results.

## Results
- Explored different thresholds (20, 30, and 50) for manual classification and analyzed their performance.
- Logistic regression achieved high accuracy:
  - Training accuracy: 97.5%
  - Validation accuracy: 96.8%
  - Testing accuracy: 96.2%

## Technologies Used
- **Python**: Core programming language.
- **Libraries**: 
  - `numpy` for numerical operations.
  - `matplotlib` for data visualization.
  - `tensorflow.keras` for loading the MNIST dataset.
  - `scikit-learn` for logistic regression and performance evaluation.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/mnist-digit-classification.git
   cd mnist-digit-classification
2. Install dependencies:    
   pip install -r requirements.txt

3. Run the script:
   python mnist_classification.py
