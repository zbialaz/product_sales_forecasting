# Product Sales Forecasting

A comprehensive time series analysis and forecasting project for product sales data using Python and advanced statistical models.

## 📊 Project Overview

This project implements a complete time series forecasting pipeline for product sales data, featuring exploratory data analysis, outlier detection, feature engineering, and multiple forecasting models including ARIMA and SARIMA.

### Key Features

- **Comprehensive EDA**: Statistical analysis and visualizations of sales patterns
- **Outlier Detection**: Advanced algorithms to identify and handle anomalies
- **Feature Engineering**: Temporal features creation (trends, seasonality, lags)
- **Multiple Models**: ARIMA, SARIMA, and baseline models implementation
- **Model Comparison**: Systematic evaluation and performance comparison
- **Automated Optimization**: Grid search for optimal model parameters

## 🗂️ Project Structure

```
product_sales_forecasting/
├── data/
│   └── product_sales.csv          # Raw sales data
├── notebook/
│   └── product_sales.ipynb        # Main analysis notebook
├── requirements.txt               # Python dependencies
├── venv/                         # Virtual environment
└── README.md                     # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter Notebook**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Time Series**: statsmodels
- **Data Source**: kaggle

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/product_sales_forecasting.git
   cd product_sales_forecasting
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the analysis notebook**
   Navigate to `notebook/product_sales.ipynb` and run the cells

## 📈 Analysis Pipeline

### 1. Data Loading and Exploration
- Load and inspect the sales dataset
- Basic statistical summary and data quality checks
- Missing values and data type analysis

### 2. Exploratory Data Analysis (EDA)
- Time series decomposition (trend, seasonality, residuals)
- Sales distribution analysis
- Seasonal patterns and trends identification
- Correlation analysis

### 3. Outlier Detection
- Statistical methods for anomaly detection
- Visualization of outliers in time series
- Impact assessment and handling strategies

### 4. Feature Engineering
- **Temporal Features**: day of week, month, quarter, year
- **Lag Features**: previous periods sales
- **Rolling Statistics**: moving averages and standard deviations
- **Trend Components**: linear and polynomial trends

### 5. Model Development

#### Baseline Models
- **Naive Forecast**: Last value propagation
- **Seasonal Naive**: Last seasonal value
- **Moving Average**: Simple and weighted averages
- **Linear Trend**: Linear regression on time

#### Advanced Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA with seasonal components
- **Grid Search Optimization**: Automated parameter tuning

### 6. Model Evaluation
- **Metrics**: MAE, MSE, RMSE, MAPE, AIC, BIC
- **Residual Analysis**: Normality and autocorrelation tests
- **Cross-validation**: Time series split validation
- **Performance Comparison**: Statistical significance tests

### 7. Forecasting and Visualization
- Future predictions with confidence intervals
- Forecast visualization and interpretation
- Model diagnostics and validation

## 📊 Key Functions

### Core Analysis Functions
- `create_temporal_features()`: Generate time-based features
- `detect_sales_outliers()`: Identify anomalies in sales data
- `create_baseline_models()`: Implement simple forecasting models
- `evaluate_model()`: Comprehensive model evaluation metrics

### Advanced Modeling Functions
- `diagnose_arima_problem()`: ARIMA model diagnostics
- `create_optimized_arima()`: Automated ARIMA optimization
- `create_optimized_sarima()`: Seasonal ARIMA with grid search
- `final_model_comparison()`: Compare all models performance

## 📈 Results and Insights

The analysis provides:

1. **Sales Patterns**: Identification of seasonal trends and patterns
2. **Outlier Impact**: Quantification of anomalies effect on forecasts
3. **Model Performance**: Comparative analysis of different approaches
4. **Optimal Parameters**: Best-fit model configurations
5. **Future Predictions**: Reliable sales forecasts with confidence intervals

## 📝 Usage Examples

### Quick Start
```python
# Load the notebook and run all cells for complete analysis
# Key results will be displayed in the output cells

# For custom analysis:
from notebook.product_sales import *

# Load data
df = pd.read_csv('data/product_sales.csv')

# Create features
df_features = create_temporal_features(df)

# Detect outliers
outliers = detect_sales_outliers(df_features)

# Build and evaluate models
models = create_baseline_models(df_features)
arima_model = create_optimized_arima(train_data, test_data)
```
