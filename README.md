# Numerai_Models
Weekly Model Predictions for Numerai Hedge Fund Competition

## Documentation

### Week 2, Round 331 (Current)
- Manual Grid Search and Cross Validation Algorithm
- New Models: KNN Regression, Regression Tree

### Week 1, Round 330
- Data Pipeline Setup
- Model Setup (under sklearn)
- New Models: OLS, Ridge, Lasso

## General Introduction

### Activate Virtual Environment
1. Navigate to project directory
2. Run command "source ./venv/bin/activate"

### Package Requirements
1. Activate virtual environment (see above)
2. Navigate to project directory
3. Run command "pip3 freeze > requirements.txt"
4. Open "requirements.txt"

### Data
- "id" = Unique stock identifier
- "era" = Unique timestamp of each observation
- "feature..." = Prediction features
- "target" = Prediction target

### Submission
- Frequency: Weekly
- Start: Saturday, 20:00 CET
- End: Monday, 16:30 CET

### Scoring (Timing)
- 1st Score: Friday after submission deadline
- Final Score: 20 days after first score; Thursdaxy 5 weeks after submission deadline

### Scoring (Criteria)
1. TC: True Contribution to Meta Model
2. CORR: Correlations between predictions and targets
3. MMC: Meta Model Contribution
4. FNC: Feature Neutral Contribution

## Methodology

### Data Preprocessing
1. Missing Data: Features with missing data are removed from train, test and live data set
2. Factor Neutralization: Features are non-stationary, hence features with high exposure to target at one point in time might not have the same exposure at another point in time, leading to inconsistent predictive power of the model in the long-run. Substrating the linear relationship of high-exposure features from prediction neutrlaizes such features and ensures consistent predictive power over time. 

## Notes

### Git Repository
1. Stage local changes: git add .
2. Commit changes: git commit -m "commit message"
3. Push changes to main repository: git push