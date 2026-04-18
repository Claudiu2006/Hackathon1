# 🎯 Cross-Sell Intelligence Dashboard

A Streamlit web app that identifies cross-selling opportunities across a large customer and product base using hierarchical gap analysis and collaborative filtering.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 🚀 Deploy to Streamlit Cloud

1. **Fork this repo** to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file: `app.py`
4. Click **Deploy**

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
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── .streamlit/config.toml        # Dark theme
├── utils/
│   ├── engine.py                 # ML recommendation engine
│   ├── data_loader.py            # File ingestion & validation
│   └── charts.py                 # Plotly chart builders
└── data/sample/
    ├── Sales.xlsx
    ├── Products.xlsx
    └── Customers.xlsx
```

---

## 📊 Required Column Names

### Sales.xlsx
| Column | Description |
|--------|-------------|
| `key_source_soldto` | Customer join key |
| `key_source_material_pl` | Product join key |
| `Value` | Transaction revenue |
| `QTY` | Quantity sold |
| `date` | Transaction date (optional) |

### Products.xlsx
| Column | Description |
|--------|-------------|
| `key_source_material_pl` | Product join key |
| `key_pl` | Product Line (L1 hierarchy) |
| `keytext_productgroup` | Product Group (L2 hierarchy) |
| `keytext_productclass` | Product Class (L3 hierarchy) |
| `division` | Division grouping |
| `material` | Material name/code |

### Customers.xlsx
| Column | Description |
|--------|-------------|
| `key_source_soldto` | Customer join key |
| `soldto` | Customer name |
| `danfoss_region_2` | Region |
| `text_country` | Country |
| `customer_type_2` | Customer type |
| `sf_primary_segment` | Primary segment |
| `sf_primary_application` | Primary application |

---

## 🧠 Recommendation Engine

### Level 1 — Product Line Gap (Highest Priority)
Identifies Product Lines the customer hasn't purchased, where they already buy in the same **Division**. Propensity is weighted by market-wide PL adoption rate.

### Level 4 — New Idea via Collaborative Filtering
Uses cosine similarity on a customer × Product Line matrix to find similar customers, then suggests Product Lines those customers buy that the target customer doesn't.

### Revenue Prediction
```
predicted_revenue = avg_revenue_per_PL_transaction × propensity_score
propensity = min(0.92, adoption_rate × 1.4 + 0.05)   # L1
propensity = min(0.75, similarity_score × 0.55 + 0.08) # L4
```

---

## 📈 Dashboard

- **Executive Summary** — Total opportunity vs actual YTD revenue + uplift %
- **Ranked Lead Table** — Customers sorted by predicted revenue opportunity
- **All Recommendations** — Full list with propensity scores, reasons, and predicted revenue
- **Analytics** — Region, segment, country, customer type, level, and propensity charts
- **Customer Detail** — Per-customer owned PLs, recommendations with full context
- **Raw Data Explorer** — Browse Sales / Products / Customers / Merged data

---

## ⚙️ Performance Notes

- The app analyses the **top N customers by revenue** (configurable in sidebar, default 500)
- Full 8,493-customer run takes ~60–90 seconds; 500 customers takes ~5–10 seconds
- The customer × PL matrix is 8,493 × 23 — tractable for real-time Streamlit use

---

## 📝 License

MIT
