# 2D Time Series Forecasting for Cohort-Based Data

## Overview

This repository implements a novel two-dimensional (2D) time series forecasting approach designed to enhance predictive accuracy in small data environments, particularly for subscription-based and cohort-level analyses. Our method transforms traditional time series data into a 2D matrix representation, providing robust forecasting capabilities for businesses with limited historical data.

### Key Features

- 2D time series modeling for cohort-based data
- ARIMAX-based forecasting algorithms with enhanced accuracy
- Optimized for small data environments with limited historical information
- Specialized for user subscription and revenue prediction workflows
- Superior performance compared to traditional forecasting methods

## Research Paper

This implementation is based on the research paper:
**"Enhancing Forecasting with a 2D Time Series Approach for Cohort-Based Data"**
- Authors: Yonathan Guttel, Nachi Lieder, Orit Moradov, Osnat Greenstein-Messica
- Company: Lightricks
- Publication Year: 2025 (Forthcoming)

### Key Innovations

- Transforms standard time series representation into a two-dimensional matrix
- Combines cohort resolution with prediction horizon
- Demonstrates superior performance in long-term forecasting
- Adaptable to various industries with limited historical data
- Reduces prediction errors by incorporating cohort-specific patterns

## Installation

### Prerequisites

- Python 3.8+
- Poetry (recommended for dependency management)

```bash
# Set up Python environment with pyenv (recommended)
pyenv shell 3.9.16

# Initialize Poetry environment
poetry env use $(pyenv which python)
poetry install
```

## Project Structure

```
├── config.py            # Configuration settings and parameters
├── main.py              # Main script for running model inference
├── model.py             # 2D Time Series ARIMA model implementation
├── preprocessing.py     # Data preprocessing and transformation utilities
├── pyproject.toml       # Poetry project configuration
├── poetry.lock          # Poetry dependency lock file
├── Data/                # Data directory
│   ├── __init__.py      # Python package indicator
│   ├── customer_info.csv       # Customer demographic data
│   ├── customer_product.csv    # Product usage data
│   ├── customer_cases.csv      # Customer interaction data
│   ├── product_info.csv        # Product metadata
│   ├── adjusted_data.csv       # Preprocessed data
│   └── results.csv             # Model output results
├── notebooks/           # Jupyter notebooks for analysis and demonstration
│   ├── __init__.py      # Python package indicator
│   └── kaggel_customer_data_modelling.ipynb  # Example notebook
└── README.md            # Project documentation
```

## Configuration

The `config.py` file allows customization of model parameters:

- `first_record`: Starting date for analysis (YYYY-MM-DD format)
- `max_month_since_attribution`: Maximum number of months to track in cohort analysis
- `target_value`: Target variable for prediction (e.g., 'revenue', 'active_users')
- `features`: Additional features to include in prediction models
- `months_to_predict`: Forecast horizon (number of months to predict)

## Usage

### Data Preprocessing

```python
from preprocessing import add_month0_data, add_future_records

# Load and preprocess your cohort data
raw_df = pd.read_csv('Data/customer_info.csv')
processed_df = add_month0_data(raw_df)
full_df = add_future_records(processed_df)
```

### Model Inference

```python
from model import run_inference

# Run predictions for a specific time period
predictions = run_inference(processed_df, prediction_month)

# Save results to CSV
predictions.to_csv('Data/results.csv', index=False)
```

## Command-Line Interface

The package provides a convenient command-line interface for running forecasts:

### Parameters

- `prediction_time` (`-pt`): The date for prediction (format: 'YYYY-MM-DD')
  - Default: Current date
- `horizon_steps` (`-hs`): Number of steps ahead to forecast
  - Default: 12
- `step_unit` (`-su`): Time unit for forecasting
  - Allowed values: 'Y' (Year), 'M' (Month), 'D' (Day), 'W' (Week), 'H' (Hour), 'm' (Minute), 's' (Second)
  - Default: 'M' (Month)
- `data_path` (`-dp`): Path to input CSV data file (required)
- `save_path` (`-sp`): Path to save forecasting results (required)

### Usage Examples

```bash
# Basic usage
python main.py -dp Data/adjusted_data.csv -sp Data/results.csv

# Specify prediction date and horizon
python main.py -pt 2024-01-01 -hs 12 -dp Data/adjusted_data.csv -sp Data/results.csv

# Use weekly forecasting instead of monthly
python main.py -su W -dp Data/adjusted_data.csv -sp Data/results.csv
```

## Performance Metrics

Our 2D forecasting model demonstrates significant improvements over traditional approaches:

- Lower Mean Absolute Error (MAE)
- Reduced Root Mean Square Error (RMSE)
- Consistent Symmetric Mean Absolute Percentage Error (sMAPE)

### Comparative Performance

| Dataset                    | Metric | 2D Model | Linear Regression | XGBoost | Prophet |
|----------------------------|--------|----------|-------------------|---------|---------|
| **Applications**           | MAE    | 0.06     | 0.28              | 1.10    | 1.07    |
| **Applications**           | RMSE   | 0.24     | 0.53              | 1.05    | 1.03    |
| **Applications**           | sMAPE  | 6.45     | 27.32             | 182.66  | 51.85   |
| **Customer Subscription**  | MAE    | 0.03     | 0.19              | 0.07    | 1.07    |
| **Customer Subscription**  | RMSE   | 0.17     | 0.44              | 0.27    | 1.03    |
| **Customer Subscription**  | sMAPE  | 3.28     | 24.39             | 7.87    | 52.81   |

## Limitations

While our approach offers significant advantages, users should be aware of the following limitations:

- Iterative prediction approach limits computational parallelization
- Increased complexity in uncertainty estimation compared to traditional methods
- Requires careful feature selection to avoid overfitting in small data environments
- May require domain-specific adjustments for optimal performance

## Future Work

Planned enhancements for this project include:

- Integration of external factors and market indicators
- Validation across diverse industries and data types
- Improved uncertainty estimation techniques
- Parallel computation optimizations
- Interactive visualization tools for forecast exploration

## Citation

If you use this work in academic research, please cite:
```
Guttel, Y., et al. (2025). Enhancing Forecasting with a 2D Time Series Approach for Cohort-Based Data.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or collaboration opportunities, please contact the authors at the email addresses provided in the research paper.