# 📊 Customer Value Forecasting & Offer Optimization

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

*End-to-end ML system for predicting customer revenue and optimizing promotional strategies*

[Live Demo](#) • [Portfolio](https://decodedbyfarhan.tech) • [LinkedIn](https://www.linkedin.com/in/farhan16/)

</div>

---

## 🎯 Project Overview

**Business Challenge:** Companies need to identify which customers will generate the most value in the future to optimize marketing spend and promotional strategies.

**Solution:** A predictive machine learning system that estimates future customer revenue and automatically segments users into actionable groups for targeted marketing campaigns.

### 🔄 Full Data Science Lifecycle

```
Data Collection → EDA → Feature Engineering → Model Training → Evaluation → Deployment
```

---

## ✨ Key Features

- 📈 **Revenue Prediction** - Forecast future customer value with high accuracy
- 👥 **Customer Segmentation** - Automatic grouping into 4 value tiers
- 🎁 **Offer Optimization** - Data-driven promotional strategies
- 📊 **Interactive Dashboard** - Streamlit web application
- 🔄 **Production Ready** - Complete ML pipeline with preprocessing

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="25%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="50" height="50"/>
<br><strong>Python</strong>
</td>
<td align="center" width="25%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="50" height="50"/>
<br><strong>Pandas</strong>
</td>
<td align="center" width="25%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="50" height="50"/>
<br><strong>NumPy</strong>
</td>
<td align="center" width="25%">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="80" height="50"/>
<br><strong>Scikit-learn</strong>
</td>
</tr>
<tr>
<td align="center" width="25%">
<strong>CatBoost</strong>
<br>ML Model
</td>
<td align="center" width="25%">
<strong>Matplotlib</strong>
<br>Visualization
</td>
<td align="center" width="25%">
<strong>Streamlit</strong>
<br>Web App
</td>
<td align="center" width="25%">
<strong>Pickle</strong>
<br>Model Serialization
</td>
</tr>
</table>

---

## 📂 Project Structure

```
customer-value-forecasting/
│
├── 📓 notebooks/
│   ├── 01_Load&EDA.ipynb                    # Initial data exploration
│   ├── 02_Clean_Transactions.ipynb          # Data cleaning pipeline
│   ├── 03_Visualization&Charts.ipynb        # Exploratory visualizations
│   ├── 04_Time_series_split.ipynb           # Train/test splitting
│   ├── 05_RFM_Dataset.ipynb                 # RFM analysis
│   ├── 06_Creating_Labels.ipynb             # Target variable creation
│   ├── 07_Building_Pipelines.ipynb          # Feature engineering pipeline
│   ├── 08_Model_Training&Evaluation.ipynb   # Model development
│   ├── 09_Segmentation&Analyze.ipynb        # Customer segmentation
│   ├── 10_Offer_Strategy.ipynb              # Marketing optimization
│   └── 11_Final.ipynb                       # End-to-end pipeline
│
├── 🤖 models/
│   ├── preprocess_pipeline.pkl              # Feature preprocessing
│   └── revenue_model.pkl                    # Trained CatBoost model
│
├── 🌐 Streamlit/
│   └── app.py                               # Interactive web application
│
├── 📊 figures/                               # Generated visualizations
│
├── 📄 Forcasting_Model.ipynb                # Main notebook
│
└── 📖 README.md                             # Project documentation
```

---

## 🔍 Feature Engineering

Key features engineered from transactional data to capture customer behavior:

| Feature Category | Features | Purpose |
|-----------------|----------|---------|
| 💰 **Revenue Patterns** | Revenue last 30/90 days | Recent spending behavior |
| 📈 **Trends** | Purchase trend, growth rate | Spending trajectory |
| 👤 **Customer Profile** | Tenure, avg order value | Customer lifecycle stage |
| 🎁 **Promotional** | Mean discount, promo sensitivity | Price responsiveness |
| 📊 **RFM Metrics** | Recency, Frequency, Monetary | Customer engagement level |

### RFM Analysis
- **Recency** - Days since last purchase
- **Frequency** - Number of transactions
- **Monetary** - Total revenue contribution

---

## 🤖 Model Development

### Models Evaluated

| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| Random Forest | ✅ | ✅ | ✅ | Medium |
| LightGBM | ✅ | ✅ | ✅ | Fast |
| **CatBoost** | ⭐ Best | ⭐ Best | ⭐ Best | Medium |

**Final Model:** CatBoost Regressor
- Superior performance on metrics
- Handles categorical features natively
- Robust to overfitting
- Built-in regularization

### Model Performance Metrics
- **MAE (Mean Absolute Error)** - Average prediction error
- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- **R² Score** - Variance explained by the model

---

## 👥 Customer Segmentation

Customers are automatically grouped into **4 value tiers** based on predicted revenue:

<table>
<tr>
<td align="center" width="25%">
<h3>💎 VIP Customers</h3>
<p>Top 10% revenue generators</p>
<strong>Strategy:</strong> Premium service, exclusive offers
</td>
<td align="center" width="25%">
<h3>🌟 High Value</h3>
<p>Next 20% of customers</p>
<strong>Strategy:</strong> Loyalty programs, upselling
</td>
<td align="center" width="25%">
<h3>📊 Medium Value</h3>
<p>Middle 40% segment</p>
<strong>Strategy:</strong> Engagement campaigns, cross-selling
</td>
<td align="center" width="25%">
<h3>🎯 Low Value</h3>
<p>Bottom 30% of customers</p>
<strong>Strategy:</strong> Activation campaigns, re-engagement
</td>
</tr>
</table>

---

## 🌐 Streamlit Application

Interactive web application for real-time predictions and analysis.

### Features

- 📤 **Upload Data** - CSV file with customer transactions
- 🔮 **Generate Predictions** - Instant revenue forecasts
- 📊 **Visualizations** - Interactive charts and insights
- 👥 **Auto-Segmentation** - Automatic customer grouping
- 💾 **Export Results** - Download predictions as CSV

### Running the Application

```bash
# Navigate to Streamlit directory
cd Streamlit

# Run the app
streamlit run app.py

# Access at http://localhost:8501
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

```bash
# Clone the repository
git clone https://github.com/farhann-16/customer-value-forecasting.git
cd customer-value-forecasting

# Install required packages
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn catboost matplotlib streamlit
```

### Quick Start

```python
import pickle
import pandas as pd

# Load trained models
with open('models/preprocess_pipeline.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/revenue_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your data
data = pd.read_csv('customer_data.csv')

# Preprocess and predict
features = preprocessor.transform(data)
predictions = model.predict(features)

# View results
results = pd.DataFrame({
    'customer_id': data['customer_id'],
    'predicted_revenue': predictions
})
print(results)
```

---

## 💼 Business Use Cases

### 1. Customer Lifetime Value (CLV) Estimation
Predict total revenue a customer will generate over their lifetime.

### 2. Marketing Campaign Optimization
- Identify high-value customers for premium campaigns
- Target medium-value customers for engagement
- Re-activate low-value customers with special offers

### 3. Personalized Promotional Offers
- VIP: Exclusive products, early access
- High Value: Loyalty rewards, volume discounts
- Medium Value: Cross-sell opportunities
- Low Value: Re-engagement discounts

### 4. Revenue Forecasting
Predict future revenue streams for financial planning and budgeting.

### 5. Churn Prevention
Identify at-risk high-value customers before they churn.

---

## 📊 Sample Results

```
Customer Segmentation Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIP Customers:      10% (500 customers)
High Value:         20% (1,000 customers)
Medium Value:       40% (2,000 customers)
Low Value:          30% (1,500 customers)

Model Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAE:    $45.23
RMSE:   $67.89
R²:     0.87
```

---

## 📈 Key Insights

- 📊 Top 10% of customers generate **60% of total revenue**
- 🎁 Customers with **higher discount sensitivity** show **25% lower CLV**
- ⏰ **Recency** is the strongest predictor of future purchases
- 📅 Customers active in last 30 days are **3x more likely** to purchase again

---

## 🔮 Future Enhancements

- [ ] Add deep learning models (LSTM for time series)
- [ ] Implement A/B testing framework
- [ ] Real-time prediction API
- [ ] Customer churn prediction module
- [ ] Multi-product recommendation engine
- [ ] Automated model retraining pipeline
- [ ] Dashboard with Power BI/Tableau integration

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Farhan Diwan**

<div align="center">

[![Portfolio](https://img.shields.io/badge/Portfolio-decodedbyfarhan.tech-blue?style=for-the-badge&logo=google-chrome)](https://decodedbyfarhan.tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-farhan16-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/farhan16/)
[![GitHub](https://img.shields.io/badge/GitHub-farhann--16-181717?style=for-the-badge&logo=github)](https://github.com/farhann-16)

</div>

---

## 🙏 Acknowledgments

- CatBoost team for the excellent ML library
- Streamlit for the intuitive web framework
- Open-source community for inspiration and tools

---

<div align="center">

### ⭐ If this project helped you, please give it a star!

**Made with ❤️ and ☕ by Farhan**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=farhann-16.customer-value-forecasting)

</div>
