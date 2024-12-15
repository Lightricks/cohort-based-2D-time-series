# 2D Time Series Forecasting for Cohort-Based Data

## Overview

This repository implements a novel two-dimensional (2D) time series forecasting approach designed to enhance predictive accuracy in small data environments, particularly for subscription-based and cohort-level analyses.

### Key Features

- 2D time series modeling for cohort-based data
- ARIMAX-based forecasting
- Supports small data environments
- Handles user subscription and revenue prediction

## Research Paper

The implementation is based on the research paper:
**"Enhancing Forecasting with a 2D Time Series Approach for Cohort-Based Data"**
- Authors: Yonathan Guttel, Nachi Lieder, Orit Moradov, Osnat Greenstein-Messica
- Company: Lightricks

### Key Innovations

- Transforms time series representation into a two-dimensional matrix
- Combines cohort resolution with prediction horizon
- Demonstrates superior performance in long-term forecasting
- Adaptable to various industries with limited historical data

## Installation

### Prerequisites

- Python 3.8+
- Poetry (optional)

```bash
pyenv shell 3.9.16
poetry env use $(pyenv which python)
poetry install
```


## Project Structure

```
├── preprocessing.py     # Data preprocessing utilities
├── model.py             # 2D Time Series ARIMA model implementation
├── config.py            # Configuration settings
├── main.py              # Main script for running inference
├── pyproject.toml       # Poetry project configuration
├── poetry.lock          # Poetry lock file
├── data/                # Sample data files
    └── results/         # Sample results files
├── notebooks/           # Jupyter notebooks for data analysis
└── README.md            # Project documentation
 
```

## Configuration

Edit `config.py` to customize:
- `first_record`: Starting date for analysis
- `max_month_since_attribution`: Maximum months to track
- `target_value`: Target variable for prediction
- `features`: Additional features for prediction
- `months_to_predict`: Forecast horizon

## Usage

### Preprocessing

```python
from preprocessing import add_month0_data, add_future_records

# Preprocess your cohort data
processed_df = add_month0_data(raw_df)
full_df = add_future_records(processed_df)
```

### Model Inference

```python
from model import run_inference

# Run predictions for a specific month
predictions = run_inference(processed_df, prediction_month)
```

## Command-Line Interface
This script provides a command-line interface for running time series forecasting using a 2D ARIMA model.

## Parameters
- `prediction_time` (optional): The date for prediction, format 'YYYY-MM-DD'. 
  - Default: Current date
- `horizon_steps` (optional): Number of steps ahead to forecast
  - Default: 12
- `step_unit` (optional): Time unit for forecasting
  - Allowed values: 'Y' (Year), 'M' (Month), 'D' (Day), 'W' (Week), 'H' (Hour), 'm' (Minute), 's' (Second)
  - Default: 'M' (Month)
- `data_path` (required): Path to input CSV data file
- `save_path` (required): Path to save forecasting results

## Usage Examples
```bash
# Basic usage
python script.py -dp input_data.csv -sp results.csv

# Specify prediction date and horizon
python script.py -pt 2024-01-01 -hs 12 -dp input_data.csv -sp results.csv

# Specify different step unit
python script.py -su W -dp input_data.csv -sp results.csv
```

## Performance Metrics

The 2D model demonstrates:
- Lower Mean Absolute Error (MAE)
- Reduced Root Mean Square Error (RMSE)
- Consistent Symmetric Mean Absolute Percentage Error (sMAPE)

### Comparative Performance

| Dataset | Metric | 2D Model | Linear Regression | XGBoost | Prophet |
|---------|--------|----------|------------------|---------|---------|
| Applications | MAE | 0.06 | 0.28 | 1.10 | 1.07 |
| Customer Subscription | RMSE | 0.17 | 0.44 | 0.27 | 1.03 |

## Limitations

- Iterative approach limits computational parallelization
- Complexity in uncertainty estimation
- Requires careful feature selection

## Future Work

- Integrate external factors
- Validate across diverse industries
- Improve uncertainty estimation techniques

## Citation

If you use this work in academic research, please cite:
```
Guttel, Y., et al. (2024). Enhancing Forecasting with a 2D Time Series Approach for Cohort-Based Data.
```

## License

This project is licensed under the MIT License

## Contact

For questions or collaboration, contact the authors at the emails provided in the research paper.