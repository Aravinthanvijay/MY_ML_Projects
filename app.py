import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import calendar
import os

# ======================================================
# ✅ Load Models, Weights, Feature Columns & Label Encoders
# ======================================================
MODEL_DIR = r"D:\MY_ML_VSC\Models"

rf_model = joblib.load(os.path.join(MODEL_DIR, "model_compressed.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
weights = joblib.load(os.path.join(MODEL_DIR, "ensemble_weights.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

rf_w, xgb_w, lgb_w = weights["rf_w"], weights["xgb_w"], weights["lgb_w"]
total_w = rf_w + xgb_w + lgb_w

# ======================================================
# ✅ Page Configuration
# ======================================================
st.set_page_config(page_title="Rossmann Sales Predictor", page_icon="📈", layout="wide")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='title'>🏪 Rossmann Store Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict store sales between <b>Aug 1 – Sep 17, 2015</b> (Rossmann official test period)</p>", unsafe_allow_html=True)

# ======================================================
# ✅ Sidebar Inputs
# ======================================================
st.sidebar.markdown("<h2 class='sidebar-title'>📋 Store & Date Input</h2>", unsafe_allow_html=True)

store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, step=1)
date_input = st.sidebar.date_input(
    "Select Date",
    min_value=datetime.date(2015, 8, 1),
    max_value=datetime.date(2015, 9, 17),
    value=datetime.date(2015, 8, 1)
)
day_of_week = date_input.weekday() + 1
promo = st.sidebar.selectbox("Promo Active?", [0, 1])

state_holiday_map = {"0": 0, "a": 1, "b": 2, "c": 3}
state_holiday_str = st.sidebar.selectbox("State Holiday", list(state_holiday_map.keys()))
state_holiday = state_holiday_map[state_holiday_str]
school_holiday = st.sidebar.selectbox("School Holiday?", [0, 1])

# ======================================================
# ✅ Load Store Info
# ======================================================
store_df = pd.read_csv(r"D:\MY_ML_VSC\Datasets\store.csv")
if store_id not in store_df["Store"].values:
    st.error("❌ Store ID not found in store.csv!")
    st.stop()

store_info = store_df[store_df["Store"] == store_id].iloc[0].to_dict()

# ======================================================
# ✅ Feature Engineering
# ======================================================
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

month = date_input.month
year = date_input.year
day = date_input.day
week_of_year = date_input.isocalendar()[1]
quarter = (month - 1) // 3 + 1
day_of_year = date_input.timetuple().tm_yday
is_weekend = 1 if day_of_week in [6, 7] else 0
is_month_start = 1 if day == 1 else 0
is_month_end = 1 if day == calendar.monthrange(year, month)[1] else 0
season = get_season(month)

input_data = {
    "Store": store_id,
    "DayOfWeek": day_of_week,
    "Date": date_input,
    "Promo": promo,
    "StateHoliday": state_holiday,
    "SchoolHoliday": school_holiday,
    "Year": year,
    "Month": month,
    "Day": day,
    "WeekOfYear": week_of_year,
    "Quarter": quarter,
    "DayOfYear": day_of_year,
    "IsWeekend": is_weekend,
    "IsMonthStart": is_month_start,
    "IsMonthEnd": is_month_end,
    "Season": season
}

input_data.update(store_info)
df_input = pd.DataFrame([input_data])

# Handle Missing Columns
for col in ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']:
    if col not in df_input.columns:
        df_input[col] = 0

df_input['CompetitionDistance_Scaled'] = np.log1p(df_input['CompetitionDistance'])
df_input['CompetitionOpenMonths'] = 12 * (df_input['Year'] - df_input['CompetitionOpenSinceYear']) + (
    df_input['Month'] - df_input['CompetitionOpenSinceMonth'])
df_input['CompetitionOpenMonths'] = df_input['CompetitionOpenMonths'].apply(lambda x: x if x > 0 else 0)
df_input['Promo2ActiveMonths'] = 12 * (df_input['Year'] - df_input['Promo2SinceYear']) + (
    (df_input['WeekOfYear'] - df_input['Promo2SinceWeek']) / 4)
df_input['Promo2ActiveMonths'] = df_input['Promo2ActiveMonths'].apply(lambda x: x if x > 0 else 0)
df_input['PromoEffect'] = df_input['Promo'] * df_input['Promo2']
df_input['IsPromoActive'] = ((df_input['Promo'] == 1) | (df_input['Promo2'] == 1)).astype(int)

drop_cols = ['Date', 'PromoInterval']
df_input.drop(columns=drop_cols, inplace=True, errors='ignore')
df_input = df_input.fillna(0)

# Label Encoding
for col in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval', 'Season']:
    if col in df_input.columns and col in label_encoders:
        try:
            df_input[col] = label_encoders[col].transform(df_input[col])
        except ValueError:
            df_input[col] = 0

# Align columns
for col in feature_columns:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[[col for col in feature_columns if col in df_input.columns]]

# ======================================================
# ✅ Predict
# ======================================================
rf_pred = rf_model.predict(df_input)[0]
xgb_pred = xgb_model.predict(df_input)[0]
lgb_pred = lgbm_model.predict(df_input)[0]
final_pred_log = (rf_pred * rf_w + xgb_pred * xgb_w + lgb_pred * lgb_w) / total_w
final_pred = np.expm1(final_pred_log)

# ======================================================
# ✅ Display Results
# ======================================================
st.markdown("<div class='result-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='result-title'>📊 Prediction Result</h2>", unsafe_allow_html=True)
st.metric(label="Predicted Sales (€)", value=f"{final_pred:,.2f}")
st.caption(f"Prediction for Store {store_id} on {date_input.strftime('%Y-%m-%d')}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.info("ℹ️ Date is restricted between **2015-08-01** and **2015-09-17** (Rossmann test period).")

import joblib
joblib.dump(rf_model, "model_compressed.pkl", compress=3)
