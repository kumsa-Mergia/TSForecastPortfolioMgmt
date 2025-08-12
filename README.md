# TSForecastPortfolioMgmt

## Overview

The objective of this project is to equip users with skills to preprocess financial data, develop time series forecasting models, analyze market trends, and optimize investment portfolios. The project provides hands-on experience leveraging data-driven insights to enhance portfolio performance, minimize risks, and capitalize on market opportunities.

## Project Structure

```

TSForecastPortfolioMgmt/
│
├── data/                            # Raw and processed datasets
│
├── notebooks/                       
│   ├── eda_tsla_bnd_spy.ipynb     
│   ├── Forecast Future Market Trends.ipynb
│   └── forecast_tsla.ipynb          
│   └── PortfolioOptimizer.ipynb
│   └── Strategy Backtesting.ipynb
│
├── src/                            # Source code modules
│   ├── data_preparation/
│   │   └── financial_data_analyzer.py  # Financial data preprocessing and analysis
│   ├── forecasting/
│   │   └── StockForecaster.py         # Time series forecasting models
│
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules

````

## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/kumsa-Mergia/TSForecastPortfolioMgmt.git
   cd TSForecastPortfolioMgmt
   ````

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

* Use notebooks in the `notebooks/` directory to explore data and run forecasting models interactively.
* Use the classes and functions in `src/` for programmatic access to financial data analysis, forecasting, and portfolio optimization.

Example:

```python
from src.forecasting.StockForecaster import StockForecaster

forecaster = StockForecaster("TSLA", start="2015-01-01")
forecaster.fetch_data()
forecaster.plot_price_history()
forecaster.forecast_arima(order=(5,1,2), future_steps=30)
```


To install  'pypfopt' use
```
!pip install git+https://github.com/robertmartin8/PyPortfolioOpt.git
```
## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

