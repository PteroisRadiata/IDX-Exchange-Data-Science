"""Streamlit UI for the final XGBoost sale-price model, with a comparable-properties panel.

Run from the project folder with:
    streamlit run streamlit_model_only.py

Expected layout (all created by the notebook):
    models/xgboost_school_hoa_flooring_final.joblib
    models/xgboost_school_hoa_flooring_final.json     (feature contract)
    comparable_sales_history.csv                      (display-only comps)

Comparable sales are NOT model inputs. They are found by similarity search over the
historical sales file and shown next to the model's estimate.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_school_hoa_flooring_final.joblib"
METADATA_PATH = MODEL_PATH.with_suffix(".json")
HISTORY_PATH = BASE_DIR / "comparable_sales_history.csv"

# ---- Comparable-search settings ------------------------------------------------
N_COMPS = 3
COMP_WINDOW_MONTHS = 12     # only sales this recent (relative to newest sale in the file)
# Lower score = better comp. Each term is roughly "how many km-equivalents of mismatch".
W_AREA = 10.0               # per unit of |log(comp area / subject area)|  (0.1 ≈ 10% diff)
W_BED = 0.5                 # per bedroom of difference
W_BATH = 0.5                # per bathroom of difference
W_MONTH = 0.1               # per month of age of the sale

BINARY_COLUMNS = ["AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"]
BINARY_LABELS = {
    "AttachedGarageYN": "Attached garage",
    "FireplaceYN": "Fireplace",
    "NewConstructionYN": "New construction",
    "PoolPrivateYN": "Private pool",
    "ViewYN": "View",
}
FLOORING_TYPES = ["Wood", "Tile", "Carpet", "Laminate", "Vinyl"]

st.set_page_config(page_title="Local Sale Price Predictor", layout="wide")
st.title("Sale price estimate")
st.caption("Model UI prototype · model and prediction run locally")


# ============================================================
# LOADERS
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_feature_list(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


@st.cache_data
def load_history(path):
    df = pd.read_csv(path, dtype={"PostalCode": str})
    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df = df.dropna(subset=["CloseDate", "ClosePrice", "LivingArea", "Latitude", "Longitude"])
    df = df[df["LivingArea"] > 0].reset_index(drop=True)
    df["PricePerSqft"] = df["ClosePrice"] / df["LivingArea"]
    return df


missing = [p for p in (MODEL_PATH, METADATA_PATH, HISTORY_PATH) if not p.exists()]
if missing:
    st.error("Missing required file(s): " + ", ".join(str(p) for p in missing))
    st.stop()

model = load_model(str(MODEL_PATH))
MODEL_FEATURES = load_feature_list(str(METADATA_PATH))
history = load_history(str(HISTORY_PATH))


# ============================================================
# HELPERS
# ============================================================

def yes_no(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not reported"
    return "Yes" if value else "No"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. lat2/lon2 may be arrays."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(np.asarray(lat2, dtype=float))
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def zip_centroid(hist, postal_code):
    """Median coordinates of past sales in a ZIP; used when no coordinates are entered."""
    rows = hist[hist["PostalCode"] == postal_code]
    if rows.empty:
        return None
    return float(rows["Latitude"].median()), float(rows["Longitude"].median())


def find_comparables(hist, lat, lon, living_area, bedrooms, bathrooms,
                     address="", n=N_COMPS, window_months=COMP_WINDOW_MONTHS):
    """Return the n most similar recent sales with Distance_km and Score columns."""
    newest = hist["CloseDate"].max()
    cutoff = newest - pd.DateOffset(months=window_months)
    pool = hist[hist["CloseDate"] >= cutoff].copy()

    # Never comp a property against its own earlier sale record.
    if address.strip():
        same = pool["UnparsedAddress"].fillna("").str.strip().str.lower() == address.strip().lower()
        pool = pool[~same]
    if pool.empty:
        return pool

    pool["Distance_km"] = haversine_km(lat, lon, pool["Latitude"], pool["Longitude"])
    area_term = np.abs(np.log(pool["LivingArea"] / living_area))
    bed_term = (pool["BedroomsTotal"].fillna(bedrooms) - bedrooms).abs()
    bath_term = (pool["BathroomsTotalInteger"].fillna(bathrooms) - bathrooms).abs()
    months_old = (newest - pool["CloseDate"]).dt.days / 30.44

    pool["Score"] = (pool["Distance_km"] + W_AREA * area_term + W_BED * bed_term
                     + W_BATH * bath_term + W_MONTH * months_old)
    return pool.nsmallest(n, "Score")


def render_property_details(d):
    """Three-column fact sheet. `d` is a dict of already-display-ready values."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Bedrooms:", d["bedrooms"])
        st.write("Bathrooms:", d["bathrooms"])
        st.write("Garage spaces:", d["garage_spaces"])
        st.write("Living area (sq ft):", d["living_area"])
        st.write("Lot area (sq ft):", d["lot_area"])
    with c2:
        st.write("HOA:", d["hoa"])
        st.write("Pool:", d["pool"])
        st.write("New construction:", d["new_construction"])
        st.write("Fireplace:", d["fireplace"])
        st.write("Property age:", d["age"])
    with c3:
        st.write("Elementary district:", d["elementary"])
        st.write("High school district:", d["high_school"])
        st.write("Flooring:", d["flooring"])


def _num(value, fmt="{:,.0f}"):
    return fmt.format(value) if pd.notna(value) else "—"


# ============================================================
# STATE
# ============================================================

st.session_state.setdefault("stage", 0)
st.session_state.setdefault("property_data", None)


def start_over():
    st.session_state.stage = 0
    st.session_state.property_data = None


def options_for(col):
    return sorted(history[col].dropna().astype(str).unique()) if col in history.columns else []


# ============================================================
# STAGE 0: PROPERTY INPUT FORM
# ============================================================

if st.session_state.stage == 0:

    city_options = options_for("City")
    county_options = options_for("CountyOrParish")
    elem_options = options_for("ElementarySchoolDistrict")
    high_options = options_for("HighSchoolDistrict")

    with st.form("property_inputs"):

        address = st.text_input("Address (display only)")

        a, b, c = st.columns(3)
        with a:
            city = st.selectbox("City", city_options, index=None, placeholder="Select a city")
            postal_code = st.text_input("ZIP code", max_chars=5)
        with b:
            county = st.selectbox("County or parish", county_options, index=None,
                                  placeholder="Select a county")
            latitude = st.number_input("Latitude (optional)", value=None, format="%.6f",
                                       help="Leave blank to use the ZIP code's typical location.")
        with c:
            longitude = st.number_input("Longitude (optional)", value=None, format="%.6f")

        left, middle, right = st.columns(3)

        with left:
            st.subheader("Property characteristics")
            living_area = st.number_input("Living area (sq ft)", min_value=1, value=1500, step=50)
            lot_area = st.number_input("Lot area (sq ft, 0 if unknown)", min_value=0, value=5000, step=250)
            bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)
            garage_spaces = st.number_input("Garage spaces", min_value=0, value=2, step=1)
            parking_total = st.number_input("Total parking spaces", min_value=0, value=2, step=1)
            property_age = st.number_input("Property age at estimate date (years)",
                                           min_value=0, value=20, step=1)

        with middle:
            st.subheader("Features")
            binary_values = {}
            for col in BINARY_COLUMNS:
                choice = st.selectbox(BINARY_LABELS[col], ["Yes", "No", "Not reported"], key=col)
                binary_values[col] = np.nan if choice == "Not reported" else float(choice == "Yes")

            st.subheader("Schools")
            elementary_district = st.selectbox("Elementary school district", elem_options,
                                               index=None, placeholder="Select (optional)")
            high_school_district = st.selectbox("High school district", high_options,
                                                index=None, placeholder="Select (optional)")

        with right:
            st.subheader("HOA")
            hoa_status = st.radio("HOA status", ["No HOA", "Has HOA", "Not reported"])
            fee_monthly = st.number_input("Monthly HOA fee ($)", min_value=0.0, value=0.0, step=25.0,
                                          help="Used only when HOA status is 'Has HOA'.")

            st.subheader("Flooring")
            flooring_materials = st.multiselect("Flooring materials", FLOORING_TYPES)
            flooring_remarks = st.checkbox("Flooring: 'see remarks'")
            flooring_missing = st.checkbox("Flooring not reported")

        submitted = st.form_submit_button("Predict sale price", type="primary")

    if submitted:
        if not address.strip():
            st.error("Enter an address to label this prediction.")
        elif not postal_code.isdigit() or len(postal_code) != 5:
            st.error("Enter a five-digit ZIP code.")
        elif (latitude is None) != (longitude is None):
            st.error("Enter both latitude and longitude, or leave both blank.")
        elif hoa_status == "Has HOA" and fee_monthly <= 0:
            st.error("Enter the monthly HOA fee, or change the HOA status.")
        else:
            st.session_state.property_data = {
                "address": address.strip(),
                "living_area": living_area,
                "lot_area": lot_area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "garage_spaces": garage_spaces,
                "parking_total": parking_total,
                "property_age": property_age,
                "latitude": latitude,
                "longitude": longitude,
                "city": city,
                "county": county,
                "postal_code": postal_code,
                "elementary_district": elementary_district,
                "high_school_district": high_school_district,
                "binary_values": binary_values.copy(),
                "hoa_status": hoa_status,
                "fee_monthly": fee_monthly,
                "flooring_materials": list(flooring_materials),
                "flooring_remarks": flooring_remarks,
                "flooring_missing": flooring_missing,
            }
            st.session_state.stage = 1
            st.rerun()


# ============================================================
# STAGE 1: PREDICTION + COMPARABLES
# ============================================================

if st.session_state.stage == 1:

    data = st.session_state.property_data
    if data is None:
        st.session_state.stage = 0
        st.rerun()

    binary_values = data["binary_values"]
    hoa_status = data["hoa_status"]
    flooring_materials = data["flooring_materials"]

    # Coordinates: the model's CoordinateImputer handles NaN, but the map and the
    # comparable search need a point, so fall back to the ZIP code's typical location.
    entered_coords = data["latitude"] is not None and data["longitude"] is not None
    map_point = ((data["latitude"], data["longitude"]) if entered_coords
                 else zip_centroid(history, data["postal_code"]))

    # ---- Model input row (must match the saved feature contract exactly) ----
    if hoa_status == "Has HOA":
        fee_value = data["fee_monthly"]
    elif hoa_status == "No HOA":
        fee_value = 0.0
    else:
        fee_value = np.nan

    row = {
        "LivingArea": data["living_area"],
        "LotSizeSquareFeet": data["lot_area"] if data["lot_area"] > 0 else np.nan,
        "BedroomsTotal": data["bedrooms"],
        "BathroomsTotalInteger": data["bathrooms"],
        "GarageSpaces": data["garage_spaces"],
        "ParkingTotal": data["parking_total"],
        "Latitude": data["latitude"] if entered_coords else np.nan,
        "Longitude": data["longitude"] if entered_coords else np.nan,
        "PropertyAgeAtSale": data["property_age"],
        **binary_values,
        **{f"{c}_Missing": int(pd.isna(binary_values[c])) for c in BINARY_COLUMNS},
        "City": data["city"] or "Unknown",
        "CountyOrParish": data["county"] or "Unknown",
        "PostalCode": data["postal_code"],
        "ElementarySchoolDistrict": data["elementary_district"] or "Unknown",
        "HighSchoolDistrict": data["high_school_district"] or "Unknown",
        "fee_monthly": fee_value,
        "HasHOA": int(hoa_status == "Has HOA"),
        "AssociationFee_missing": int(hoa_status == "Not reported"),
        "HasWoodFlooring": int("Wood" in flooring_materials),
        "HasTileFlooring": int("Tile" in flooring_materials),
        "HasCarpetFlooring": int("Carpet" in flooring_materials),
        "HasLaminateFlooring": int("Laminate" in flooring_materials),
        "HasVinylFlooring": int("Vinyl" in flooring_materials),
        "Flooring_SeeRemarks": int(data["flooring_remarks"]),
        "Flooring_missing": int(data["flooring_missing"] or not flooring_materials),
    }
    X = pd.DataFrame([row]).reindex(columns=MODEL_FEATURES)

    # ---- Prediction ----
    try:
        predicted_price = float(np.expm1(float(model.predict(X)[0])))
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError("Model returned an invalid sale price.")
        st.success(f"Estimated sale price: **${predicted_price:,.0f}**")
        st.caption("The trained model predicts log(ClosePrice); the app converts the result back "
                   "to dollars. Address is a label and is not looked up or geocoded.")
    except Exception as exc:
        predicted_price = None
        st.error("Prediction failed. Check model compatibility.")
        st.exception(exc)

    # ---- Subject property ----
    st.subheader(data["address"])
    subject_details = {
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "garage_spaces": data["garage_spaces"],
        "living_area": f"{data['living_area']:,}",
        "lot_area": f"{data['lot_area']:,}" if data["lot_area"] > 0 else "—",
        "hoa": hoa_status + (f" (${data['fee_monthly']:,.0f}/mo)" if hoa_status == "Has HOA" else ""),
        "pool": yes_no(binary_values["PoolPrivateYN"]),
        "new_construction": yes_no(binary_values["NewConstructionYN"]),
        "fireplace": yes_no(binary_values["FireplaceYN"]),
        "age": f"{data['property_age']} yrs",
        "elementary": data["elementary_district"] or "—",
        "high_school": data["high_school_district"] or "—",
        "flooring": ", ".join(flooring_materials) or "—",
    }
    render_property_details(subject_details)

    # ---- Comparable properties ----
    st.subheader("Comparable Properties")

    comps = pd.DataFrame()
    if map_point is None:
        st.info("No coordinates were entered and no past sales exist in this ZIP code, "
                "so comparable properties can't be located. Enter latitude and longitude to see them.")
    else:
        comps = find_comparables(history, map_point[0], map_point[1], data["living_area"],
                                 data["bedrooms"], data["bathrooms"], address=data["address"])
        if not entered_coords:
            st.caption("No coordinates entered; distances are measured from the ZIP code's typical location.")

    if not comps.empty:
        # Descriptive cross-check only. Not used by the model.
        comp_ppsf = comps["PricePerSqft"].median()
        comp_implied = comp_ppsf * data["living_area"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Median comp price / sq ft", f"${comp_ppsf:,.0f}")
        m2.metric("Comp-implied value", f"${comp_implied:,.0f}")
        if predicted_price:
            m3.metric("Model estimate vs. comps", f"{predicted_price / comp_implied - 1:+.1%}")
        st.caption(f"Most similar sales from the last {COMP_WINDOW_MONTHS} months of data, ranked by "
                   "distance, size, bed/bath count and recency. Comps are shown for reference and "
                   "are not inputs to the model.")

        tabs = st.tabs([f"Comp {i}" for i in range(1, len(comps) + 1)])
        for tab, (_, comp) in zip(tabs, comps.iterrows()):
            with tab:
                st.write(f"**{comp['UnparsedAddress']}**")
                s1, s2, s3 = st.columns(3)
                s1.metric("Sold for", f"${comp['ClosePrice']:,.0f}")
                s2.metric("Price / sq ft", f"${comp['PricePerSqft']:,.0f}")
                s3.metric("Distance", f"{comp['Distance_km']:.2f} km")
                st.write("Closed:", comp["CloseDate"].strftime("%b %d, %Y"))

                flooring_txt = comp.get("Flooring")
                comp_flooring = ", ".join(f for f in FLOORING_TYPES
                                          if comp.get(f"Has{f}Flooring", 0) == 1) or "—"
                if comp_flooring == "—" and isinstance(flooring_txt, str):
                    comp_flooring = flooring_txt
                hoa_txt = ("Has HOA" if comp.get("HasHOA", 0) == 1 else "No HOA")
                render_property_details({
                    "bedrooms": _num(comp["BedroomsTotal"]),
                    "bathrooms": _num(comp["BathroomsTotalInteger"]),
                    "garage_spaces": _num(comp["GarageSpaces"]),
                    "living_area": _num(comp["LivingArea"]),
                    "lot_area": _num(comp["LotSizeSquareFeet"]),
                    "hoa": hoa_txt,
                    "pool": yes_no(comp["PoolPrivateYN"]),
                    "new_construction": yes_no(comp["NewConstructionYN"]),
                    "fireplace": yes_no(comp["FireplaceYN"]),
                    "age": _num(comp["PropertyAgeAtSale"], "{:,.0f} yrs"),
                    "elementary": comp.get("ElementarySchoolDistrict") or "—",
                    "high_school": comp.get("HighSchoolDistrict") or "—",
                    "flooring": comp_flooring,
                })
    elif map_point is not None:
        st.info("No comparable sales found in the recent window.")

    # ---- One map: subject plus comps ----
    if map_point is not None:
        m = folium.Map(location=list(map_point), zoom_start=14)
        folium.Marker(list(map_point), popup=data["address"], tooltip="Subject property",
                      icon=folium.Icon(color="red", icon="home")).add_to(m)
        for i, (_, comp) in enumerate(comps.iterrows(), start=1):
            folium.Marker([comp["Latitude"], comp["Longitude"]],
                          popup=f"{comp['UnparsedAddress']}: ${comp['ClosePrice']:,.0f}",
                          tooltip=f"Comp {i}",
                          icon=folium.Icon(color="blue")).add_to(m)
        st_folium(m, width=700, height=380, returned_objects=[])

    st.button("Start Over", on_click=start_over)
