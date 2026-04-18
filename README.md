# 🎯 Cross-Sell Intelligence Dashboard

A Streamlit-powered web application that identifies cross-selling opportunities by analyzing sales transactions, product metadata, and customer data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 🚀 Deploy to Streamlit Cloud (1-click)

1. **Fork this repo** to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your forked repo, branch `main`, and set main file path to `app.py`
4. Click **Deploy** — it'll be live in ~2 minutes

---

## 🖥️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/cross-sell-intelligence.git
cd cross-sell-intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
cross-sell-intelligence/
├── app.py                        # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Dark theme configuration
├── utils/
│   ├── __init__.py
│   ├── engine.py                 # Core recommendation engine (ML logic)
│   ├── data_loader.py            # CSV/Excel ingestion & validation
│   └── charts.py                 # Plotly chart builders
└── data/
    └── sample/
        ├── transactions.csv      # Sample transaction data
        ├── products.csv          # Sample product metadata
        └── customers.csv         # Sample customer metadata
```

---

## 📊 Input Data Format

### Transactions (`transactions.csv` / `.xlsx`)
| Column | Type | Description |
|--------|------|-------------|
| `CustomerID` | string | Unique customer identifier |
| `ProductID` | string | Unique product identifier |
| `Quantity` | numeric | Units purchased |
| `UnitPrice` | numeric | Price per unit |
| `TransactionDate` | date | Optional, for time analysis |

### Products (`products.csv` / `.xlsx`)
| Column | Type | Description |
|--------|------|-------------|
| `ProductID` | string | Unique product identifier |
| `ProductName` | string | Human-readable product name |
| `ProductLine` | string | Top-level grouping (e.g. "Fluid Systems") |
| `ProductGroup` | string | Mid-level grouping (e.g. "Pumps") |
| `ProductClass` | string | Lowest grouping (e.g. "Centrifugal") |
| `StandardPrice` | numeric | Optional fallback price |

### Customers (`customers.csv` / `.xlsx`)
| Column | Type | Description |
|--------|------|-------------|
| `CustomerID` | string | Unique customer identifier |
| `CustomerName` | string | Company/customer name |
| `Segment` | string | Enterprise / Mid-Market / SMB |
| `Region` | string | Geographic region |
| `Industry` | string | Industry vertical |

---

## 🧠 Recommendation Engine

### Hierarchical Gap Analysis
The engine follows a strict priority hierarchy:

| Level | Scope | Priority Weight | Description |
|-------|-------|-----------------|-------------|
| **L1** | Product Line | 1.00 | Customer owns some products in a line but not others |
| **L2** | Product Group | 0.65 | Customer owns products in a group but not all |
| **L3** | Product Class | 0.35 | Customer owns products in a class but not all |
| **L4** | New Idea (CF) | Dynamic | Collaborative filtering suggests new product lines |

### Propensity Score Calculation
```
propensity = min(0.95, adoption_rate × priority_weight × 1.5 + 0.05)
```
Where `adoption_rate` = % of all customers who purchased the product.

### Predicted Revenue
```
predicted_revenue = avg_unit_price × avg_quantity × propensity_score
```

### Collaborative Filtering (Level 4 / "New Idea")
- Builds a customer-product purchase matrix
- Computes cosine similarity between customers
- For each customer, finds top-N most similar customers
- Suggests products (from entirely new Product Lines) that similar customers buy
- Scores suggestions by weighted purchase frequency

---

## 📈 Dashboard Features

### Executive Summary
- Total projected revenue across all customers
- Number of customers with identified gaps
- Total recommendations generated
- Average propensity score

### Ranked Opportunity List
- Customers sorted by total predicted revenue opportunity
- Per-customer: top product, number of recommendations, avg propensity

### Customer Detail View
- Current product ownership
- All recommendations with propensity scores and predicted revenue
- Expandable cards with full recommendation context

### Analytics Suite
- Revenue opportunity by Region (horizontal bar chart)
- Revenue split by Customer Segment (donut chart)
- Revenue by Recommendation Level (bar chart)
- Propensity score distribution (histogram)
- Conversion probability vs. Deal size (scatter plot)
- Top 10 customers by opportunity (bar chart)
- Customer × Product purchase matrix (heatmap)

---

## ⚙️ Configuration

Edit `.streamlit/config.toml` to customize the theme. All chart colors are defined in `utils/charts.py` in the `PALETTE` dictionary.

---

## 📝 License

MIT — free to use and modify.
