# Part 1: Retail Sales Forecasting Challenge

This directory contains all materials for the machine learning challenge to analyze and forecast retail sales data.

## Contents

```
Part1/
├── dataset/
│   └── retail_sales_data.csv   # Historical sales records
├── Case_Assessment.ipynb      # Jupyter notebook with EDA, modeling, and analysis
├── ML_Challenge.pdf           # Challenge specification and requirements
└── README.md                  # This file
```

### `dataset/retail_sales_data.csv`
- Contains daily sales figures for multiple stores and product categories.
- Typical columns: `date`, `store_id`, `product_id`, `units_sold`, `revenue`

### `Case_Assessment.ipynb`
- Interactive Jupyter notebook guiding you through:
  1. **Data Loading** and **Preprocessing**
  2. **Exploratory Data Analysis** 
  3. **Feature Engineering** 
  4. **Model Development** 
  5. **Evaluation**

### `ML_Challenge.pdf`
- Detailed description of the challenge objectives, deliverables, and timeline:
  - Forecast next-quarter sales for each store-product combination
  - Identify top 5 fastest-growing products
  - Propose inventory optimization strategies based on forecast results
  - Deliver a brief report summarizing methodology, results, and recommendations

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KimathiNewton/Zeromargin.git
   cd Zeromargin/Part1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter scikit-learn statsmodels prophet
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook Case_Assessment.ipynb
   ```

## How to Use

1. Open **`Case_Assessment.ipynb`** in Jupyter.
2. Follow the notebook cells sequentially:
   - Run **Data Loading** to inspect the dataset.
   - Execute **EDA** cells to visualize trends and anomalies.
   - Proceed to **Modeling** sections to train and validate forecasting models.
   - Review the **Evaluation** section for performance metrics and decision thresholds.
3. Refer to **`ML_Challenge.pdf`** for any clarifications on objectives or evaluation criteria.




