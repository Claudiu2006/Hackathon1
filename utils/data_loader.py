"""
Data ingestion utilities – handles CSV and Excel uploads.
"""

import pandas as pd
import io
from typing import Optional, Tuple


REQUIRED_TRANSACTION_COLS = {"CustomerID", "ProductID", "Quantity", "UnitPrice"}
REQUIRED_PRODUCT_COLS = {"ProductID", "ProductLine", "ProductGroup", "ProductClass"}
REQUIRED_CUSTOMER_COLS = {"CustomerID"}


def _read_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel from Streamlit UploadedFile."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def load_transactions(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        missing = REQUIRED_TRANSACTION_COLS - set(df.columns)
        if missing:
            return None, f"Transactions file missing columns: {missing}"
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0)
        return df, None
    except Exception as e:
        return None, str(e)


def load_products(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        missing = REQUIRED_PRODUCT_COLS - set(df.columns)
        if missing:
            return None, f"Products file missing columns: {missing}"
        return df, None
    except Exception as e:
        return None, str(e)


def load_customers(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        missing = REQUIRED_CUSTOMER_COLS - set(df.columns)
        if missing:
            return None, f"Customers file missing columns: {missing}"
        return df, None
    except Exception as e:
        return None, str(e)


def load_sample_data() -> tuple:
    """Load bundled sample datasets for demo mode."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    transactions = pd.read_csv(os.path.join(base, "transactions.csv"))
    products = pd.read_csv(os.path.join(base, "products.csv"))
    customers = pd.read_csv(os.path.join(base, "customers.csv"))
    return transactions, products, customers



