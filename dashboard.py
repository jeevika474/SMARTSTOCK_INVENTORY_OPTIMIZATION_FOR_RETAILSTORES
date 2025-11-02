import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")

# -------------------------
# 1. Load Forecast Results
# -------------------------
if not os.path.exists("data/forecast_results.csv"):
    st.error("⚠ Run forecasting.py first to generate forecast_results.csv!")
    st.stop()

df = pd.read_csv("data/forecast_results.csv")
df.columns = df.columns.str.strip()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

required_cols = ["product_name", "date", "forecast_best"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

st.title("📦 Milestone 4: Smart Inventory Dashboard")

# -------------------------
# 2. Sidebar Controls
# -------------------------
st.sidebar.header("Inventory Parameters")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost", 1, 20, 2)
service_level = st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)
z = {"90%": 1.28, "95%": 1.65, "99%": 2.33}[service_level]

# -------------------------
# 3. Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Inventory", "Stock Alerts", "Reports"])

# ---- Forecasts Tab ----
with tab1:
    st.subheader("📈 Demand Forecasts")
    prod = st.selectbox("Select Product", df["product_name"].unique())
    sub = df[df["product_name"] == prod]

    plt.figure(figsize=(8, 4))
    plt.plot(sub["date"], sub["forecast_best"], label="Forecast", color="teal")
    plt.title(f"Forecasted Demand for {prod}")
    plt.xlabel("Date")
    plt.ylabel("Forecast Quantity")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())

# ---- Inventory Tab ----
with tab2:
    st.subheader("🧮 Inventory Planning")
    plan = []
    for p in df["product_name"].unique():
        d = df[df["product_name"] == p]
        avg = d["forecast_best"].mean() / 30
        dem = d["forecast_best"].sum()
        std = d["forecast_best"].std()
        eoq = np.sqrt((2 * dem * oc) / hc)
        ss = z * std * np.sqrt(lead)
        rop = (avg * lead) + ss
        plan.append({
            "Product": p,
            "AvgDailySales": round(avg, 2),
            "TotalDemand": round(dem, 2),
            "EOQ": round(eoq, 2),
            "SafetyStock": round(ss, 2),
            "ReorderPoint": round(rop, 2)
        })
    inv = pd.DataFrame(plan)
    st.dataframe(inv)

# ---- Stock Alerts Tab ----
with tab3:
    st.subheader("🚨 Stock Alerts")
    inv["CurrentStock"] = np.random.randint(10, 100, len(inv))
    inv["Action"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder 🚨", "OK ✅")
    st.dataframe(inv[["Product", "CurrentStock", "ReorderPoint", "Action"]])
    st.bar_chart(inv.set_index("Product")[["CurrentStock", "ReorderPoint"]])

# ---- Reports Tab ----
with tab4:
    st.subheader("📄 Reports & Downloads")
    st.write("You can download the daily reorder report below:")
    st.download_button("📥 Download Report", inv.to_csv(index=False), "daily_reorder_report.csv")
    st.write("Summary Statistics:")
    st.write(inv.describe())

# ---- Upload New Data ----
st.sidebar.header("Upload New Data")
upl = st.sidebar.file_uploader("Upload New Sales Data (CSV)", type="csv")
if upl:
    new = pd.read_csv(upl)
    new.columns = new.columns.str.strip()
    st.sidebar.success("✅ File uploaded successfully!")
    st.sidebar.info("Re-run forecasting.py to refresh predictions.")