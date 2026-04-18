"""
Data ingestion - handles CSV and Excel uploads.
Real schema column names are defined in engine.py
"""
import pandas as pd
from typing import Optional, Tuple


def _read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file).lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file, engine="openpyxl")


def load_sales(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        required = {"key_source_soldto", "key_source_material_pl", "Value", "QTY"}
        missing  = required - set(df.columns)
        if missing:
            return None, f"Sales file missing columns: {missing}"
        return df, None
    except Exception as e:
        return None, str(e)


def load_products(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        required = {"key_source_material_pl", "key_pl", "keytext_productgroup", "keytext_productclass"}
        missing  = required - set(df.columns)
        if missing:
            return None, f"Products file missing columns: {missing}"
        return df, None
    except Exception as e:
        return None, str(e)


def load_customers(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = _read_file(uploaded_file)
        if "key_source_soldto" not in df.columns:
            return None, "Customers file missing column: key_source_soldto"
        return df, None
    except Exception as e:
        return None, str(e)


def load_real_data(sales_path, products_path, customers_path) -> tuple:
    """Load from file paths (used in app with real uploaded files)."""
    sales     = pd.read_excel(sales_path,     engine="openpyxl")
    products  = pd.read_excel(products_path,  engine="openpyxl")
    customers = pd.read_excel(customers_path, engine="openpyxl")
    return sales, products, customers
