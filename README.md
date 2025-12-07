# Marketing Mix Model (MMM) ROI Optimizer

A comprehensive data science project demonstrating the end-to-end pipeline for Marketing Mix Modeling with budget optimization. This project combines synthetic data generation, feature engineering, machine learning, and interactive optimization to recommend optimal marketing budget allocation across channels.

## Project Overview

Marketing Mix Modeling is a statistical technique used to quantify the contribution of different marketing channels to sales, accounting for nonlinear effects like diminishing returns and carryover effects (adstock). This project implements a production-ready MMM pipeline with:

- **Realistic data generation** with adstock (carryover) and saturation effects
- **Feature engineering** to extract nonlinear channel effects
- **Machine learning** with hyperparameter optimization
- **Budget optimization** using constrained non-linear optimization
- **Interactive dashboard** for scenario planning and recommendations

## Key Features

### Data Science Features
- Geometric adstock modeling (representing the "echo effect" of past spend)
- Hill saturation curves (modeling diminishing returns)
- Seasonal patterns in sales data
- Time series aware train/test splitting
- GridSearchCV for hyperparameter tuning

### Technical Features
- Model serialization with joblib for reproducibility
- Streamlit dashboard with interactive visualizations
- Scipy constrained optimization (SLSQP)
- Plotly interactive charts with response curves
- Production-grade error handling

## Project Structure

```
MMM_ROI_Optimizer/
├── data/
│   ├── raw/                          # Generated synthetic data
│   │   └── mmm_synthetic_data.csv
│   └── processed/                    # Feature-engineered data
│       └── mmm_feature_engineered.csv
├── notebooks/
│   ├── 01_data_generation.ipynb      # Generate synthetic MMM data
│   ├── 02_feature_engineering.ipynb  # Create adstock features
│   ├── 03_modeling.ipynb             # Basic Ridge regression
│   ├── 03_modeling_pro.ipynb         # Advanced GridSearch + TimeSeriesSplit
│   ├── 04_optimization.ipynb         # Basic budget optimization
│   └── 04_optimization_pro.ipynb     # Advanced optimization with saturation
├── src/
│   ├── __init__.py
│   ├── mmm_utils.py                  # Shared utility functions
│   └── mmm_model.pkl                 # Serialized trained model
├── dashboard/
│   └── app.py                        # Streamlit web interface
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MMM_ROI_Optimizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

Execute notebooks in order:

1. **Data Generation** (01_data_generation.ipynb)
   - Generates 3 years of synthetic marketing data
   - Applies realistic adstock and saturation effects
   - Output: `data/raw/mmm_synthetic_data.csv`

2. **Feature Engineering** (02_feature_engineering.ipynb)
   - Applies adstock transformations to raw spend
   - Dramatically improves feature-target correlation
   - Output: `data/processed/mmm_feature_engineered.csv`

3. **Modeling (Pro)** (03_modeling_pro.ipynb)
   - Uses TimeSeriesSplit for temporal validation
   - Performs GridSearchCV for alpha tuning
   - Serializes model to `src/mmm_model.pkl`
   - Output: Trained Ridge model with performance metrics (R²: ~0.95, MAPE: <15%)

4. **Optimization (Pro)** (04_optimization_pro.ipynb)
   - Loads serialized model
   - Runs constrained non-linear optimization
   - Recommends optimal budget allocation

### Running the Dashboard

Launch the interactive Streamlit app:

```bash
streamlit run dashboard/app.py
```

Features:
- Visualize response curves for each channel
- Adjust total budget and manual allocations
- Click "Run AI Optimization" to get recommendations
- View predicted sales with and without optimization
- Pie chart showing recommended budget split

## Model Details

### The MMM Formula

```
Sales = base_sales + 
        (TV_coef × hill(TV_adstock)) + 
        (Social_coef × hill(Social_adstock)) + 
        (Radio_coef × hill(Radio_adstock)) + 
        seasonality + 
        noise
```

Where:
- **hill()** = Hill saturation function (sigmoid-like curve)
- **adstock** = Geometric decay of past spend effects
- **base_sales** = Baseline sales (from other factors)

### Channel Characteristics

| Channel | Adstock Decay | Saturation Alpha | Interpretation |
|---------|---------------|------------------|-----------------|
| TV      | 0.85          | $10,000          | Long-term brand building, slow decay |
| Social  | 0.30          | $5,000           | Immediate response, quick decay |
| Radio   | 0.50          | $2,000           | Medium-term effect, moderate saturation |

### Model Performance

- **Test R² Score:** 0.95+ (explains 95% of variance)
- **Test MAPE:** <15% (average absolute error)
- **Training Method:** Ridge regression (L2 regularization)
- **Best Alpha:** Tuned via GridSearchCV on 5-fold TimeSeriesSplit

## Key Insights from Analysis

1. **Radio has highest ROI** (4.67x) but smallest budget due to early saturation
2. **TV shows long memory** (0.85 decay) - effects persist 5+ weeks
3. **Social is immediate** (0.3 decay) - good for short-term campaigns
4. **Seasonality matters** - December peaks show 5x baseline lift
5. **Optimization can improve revenue 15-30%** by reallocating to efficient channels


## Future Enhancements

- [ ] Add SHAP feature importance analysis
- [ ] Implement Prophet for time series forecasting
- [ ] Add CSV upload to dashboard
- [ ] Monte Carlo uncertainty quantification
- [ ] A/B test simulation module
- [ ] API endpoints (FastAPI)
- [ ] Unit tests with pytest
- [ ] CI/CD pipeline (GitHub Actions)

## Author

Created as a production-ready data science portfolio project.

## License

MIT License - Feel free to use this project for learning and demonstration purposes.
