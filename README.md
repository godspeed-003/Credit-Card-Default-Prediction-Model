# Credit Card Default Prediction Model

## Overview
This project implements a machine learning model using XGBoost to predict credit card defaults based on various transaction, customer, and bureau data attributes. The model achieves significant predictive performance through careful preprocessing of missing values and feature engineering.

## Model Performance
- Train AUC Score: 0.8742
- Test AUC Score: 0.8456
- Cross-validation mean AUC: 0.8521

## Features
- Comprehensive missing value analysis and handling
- Feature preprocessing for different data categories:
  - Transaction attributes
  - Onus attributes
  - Bureau attributes
  - Bureau enquiry attributes
- StandardScaler implementation for feature scaling
- XGBoost classifier with optimized parameters
- Feature importance analysis
- Model performance evaluation using AUC-ROC metric

## Requirements
All dependencies are listed in requirements.txt. To install them:

```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-default-prediction.git
cd credit-card-default-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your data files in the appropriate data folders:
   - Training data: `Dev_data_to_be_shared/Dev_data_to_be_shared.csv`
   - Validation data: `validation_data_to_be_shared/validation_data_to_be_shared.csv`

2. Run the model:
```bash
python model.py
```

## Model Details

### Data Preprocessing
- Missing value handling strategy:
  - Transaction attributes: Filled with 0
  - Onus attributes: Filled with median values
  - Bureau attributes: Filled with median values
  - Bureau enquiry attributes: Filled with 0

### Model Configuration
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
```

### Output
The model generates:
- Train and Test AUC scores
- Top 10 most important features analysis
- Predictions file: `credit_card_predictions.csv`

## Files Description
- model.py: Main script containing the model implementation
- `requirements.txt`: List of Python dependencies
- `Dev_data_to_be_shared.csv`: Development dataset
- `validation_data_to_be_shared.csv`: Validation dataset
- `credit_card_predictions.csv`: Output predictions file

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- scikit-learn
- XGBoost
- Pandas
- NumPy
