import streamlit as st
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="EU Wholesale Engine Vehicle Pricing Tool",
    page_icon="🚗",
    layout="wide"
)

# CSV file names
CSV_MAIN = "uk_streamlit_vehicle_main.csv"
CSV_REGION = "uk_streamlit_region_lookup.csv"
CSV_CONDITION = "uk_streamlit_condition_lookup.csv"
CSV_VIN = "uk_streamlit_vin_lookup.csv"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_csv():
    """Load all required tables from CSV files."""
    try:
        df_main = pd.read_csv(CSV_MAIN)
        df_region = pd.read_csv(CSV_REGION)
        df_condition = pd.read_csv(CSV_CONDITION)
        df_vin = pd.read_csv(CSV_VIN) if os.path.exists(CSV_VIN) else pd.DataFrame()

        # Normalize column names to lowercase for consistency
        df_main.columns = df_main.columns.str.lower()
        df_region.columns = df_region.columns.str.lower()
        df_condition.columns = df_condition.columns.str.lower()
        if len(df_vin) > 0:
            df_vin.columns = df_vin.columns.str.lower()

        if "transaction_count_6mo" not in df_main.columns and "transaction_count" in df_main.columns:
            df_main["transaction_count_6mo"] = df_main["transaction_count"]

        if "transaction_count_4wk" not in df_main.columns and "transaction_count" in df_main.columns:
            df_main["transaction_count_4wk"] = df_main["transaction_count"]

        if "forecast_date" not in df_region.columns and "forecast_start_date" in df_region.columns:
            df_region["forecast_date"] = df_region["forecast_start_date"]

        if "forecast_date" not in df_condition.columns and "forecast_start_date" in df_condition.columns:
            df_condition["forecast_date"] = df_condition["forecast_start_date"]

        return df_main, df_region, df_condition, df_vin
    except Exception as e:
        st.error(f"Error loading data from CSV files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


def load_data_with_progress():
    """Load data with progress bar display."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("📂 Loading CSV data...")
    progress_bar.progress(50)

    # Load all data (cached after first load)
    df_main, df_region, df_condition, df_vin = load_data_from_csv()

    status_text.text("✅ Data loaded successfully!")
    progress_bar.progress(100)

    # Clear progress indicators after a brief moment
    import time
    time.sleep(0.3)
    status_text.empty()
    progress_bar.empty()

    return df_main, df_region, df_condition, df_vin


def calculate_km_function(body_style_group, mileage):
    """
    Calculate mileage-based depreciation adjustment using body-style-specific polynomial curves.

    Applies the EU Stellantis wholesale engine pricing model to compute KM depreciation
    based on vehicle body style category. Each body style group uses a unique third-degree
    polynomial curve calibrated from historical market data.

    Args:
        body_style_group (str): Vehicle body style category. Valid groups:
            - Curve 1: "Hatchback", "SUV", "Sedan", "Estate"
            - Curve 2: "MPV & Van", "Derived VP", "Others"
            - Curve 3: "Pickup & Chassis", "Coupe & Sports"
        mileage (float): Vehicle mileage in kilometers. Non-negative value representing
            actual odometer reading.

    Returns:
        float: KM depreciation value in EUR. Typically negative (depreciation) but varies
            based on body style and mileage range. Returns 0 if inputs are invalid.

    Note:
        Unrecognized body style groups default to Curve 1. Invalid mileage values return 0.
    """
    if mileage is None or body_style_group is None:
        return 0

    try:
        x = float(mileage)
    except:
        return 0

    # KM curve mapping from the data preparation notebook
    curve_1_groups = ["Hatchback", "SUV", "Sedan", "Estate"]
    curve_2_groups = ["MPV & Van", "Derived VP", "Others"]
    curve_3_groups = ["Pickup & Chassis", "Coupe & Sports"]

    # Apply appropriate polynomial formula
    if body_style_group in curve_1_groups:
        # Curve 1: Hatchback, SUV, Sedan, Estate
        return -1.66e-17 * x**3 + 1.45e-11 * x**2 - 4.90e-06 * x
    elif body_style_group in curve_2_groups:
        # Curve 2: MPV & Van, Derived VP, Others
        return 1.58e-18 * x**3 + 4.70e-12 * x**2 - 3.23e-06 * x
    elif body_style_group in curve_3_groups:
        # Curve 3: Pickup & Chassis, Coupe & Sports
        return -3.55e-17 * x**3 + 2.73e-11 * x**2 - 7.02e-06 * x
    else:
        # Default to curve 1 if body style not recognized
        return -1.66e-17 * x**3 + 1.45e-11 * x**2 - 4.90e-06 * x


def main():
    st.markdown("<h1 style='text-align: center;'>EU Wholesale Engine Vehicle Pricing Tool</h1>", unsafe_allow_html=True)

    # Add custom CSS to reduce metric font sizes and style sections
    st.markdown("""
        <style>
        /* Reduce metric value font size */
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }
        /* Increase metric label font size and make bold */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] > div > div {
            font-size: 16px !important;
            font-weight: 700 !important;
        }
        /* Reduce metric delta font size */
        [data-testid="stMetricDelta"] {
            font-size: 14px !important;
        }
        /* Style section headers with background */
        [data-testid="stHeading"][class*="st-emotion-cache"] h3 {
            background: linear-gradient(90deg, #e3f2fd, #f5f5f5);
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin-bottom: 16px;
        }
        /* Bold and larger selectbox labels */
        .stSelectbox label,
        .stSelectbox label > div {
            font-weight: 700 !important;
            font-size: 16px !important;
        }
        /* Bold and larger radio button labels */
        .stRadio label,
        .stRadio label > div {
            font-weight: 700 !important;
            font-size: 16px !important;
        }
        /* Bold and larger slider labels */
        .stSlider label,
        .stSlider label > div {
            font-weight: 700 !important;
            font-size: 16px !important;
        }
        /* Bold and larger number input labels */
        .stNumberInput label,
        .stNumberInput label > div {
            font-weight: 700 !important;
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load data with progress indicators
    df_main, df_region, df_condition, df_vin = load_data_with_progress()

    st.markdown("---")

    # ========== SECTION 1: FORECAST DATE SELECTION ==========
    st.markdown("""
        <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                    padding: 12px 16px; 
                    border-radius: 8px; 
                    border-left: 4px solid #1f77b4; 
                    margin-bottom: 16px;'>
            <h3 style='margin: 0; color: #262730;'>📅 Forecast Date Selection</h3>
        </div>
    """, unsafe_allow_html=True)

    # Get unique forecast dates (used for filtering)
    forecast_dates = sorted(df_main["forecast_date"].dropna().unique())
    if not forecast_dates:
        st.error("❌ No forecast dates available in the data.")
        st.stop()

    # Demo display date (single option shown to user)
    display_forecast_date = pd.to_datetime("2026-01-16")
    selected_forecast_date = st.selectbox(
        "Select Forecast Date",
        [forecast_dates[-1]],
        format_func=lambda _: display_forecast_date.strftime("%Y-%m-%d")
    )

    # Filter data by forecast date
    df_main_filtered = df_main[df_main["forecast_date"] == selected_forecast_date].copy()

    st.markdown("---")

    # ========== SECTION 2: VEHICLE SELECTION ==========
    st.markdown("""
        <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                    padding: 12px 16px; 
                    border-radius: 8px; 
                    border-left: 4px solid #1f77b4; 
                    margin-bottom: 16px;'>
            <h3 style='margin: 0; color: #262730;'>🚘 Vehicle Selection</h3>
        </div>
    """, unsafe_allow_html=True)

    # Choose selection method
    selection_method = st.radio(
        "Choose selection method:",
        ["VIN Search", "Vehicle ID Search", "Vehicle Attribute Filters"],
        horizontal=True
    )

    selected_vehicle = None

    if selection_method == "VIN Search":
        # Option: VIN Search
        st.markdown("#### Search by VIN:")

        vin_input = st.text_input(
            "Enter VIN:",
            placeholder="Type or paste VIN here",
            help="Enter the Vehicle Identification Number"
        ).strip().upper()

        # Show example VINs after input field
        if len(df_vin) > 0:
            # Get random VINs for variety across manufacturers and body styles
            df_vin_merged = df_vin.merge(
                df_main_filtered[["vehicle_catalog_name"]],
                on="vehicle_catalog_name",
                how="inner"
            )

            if len(df_vin_merged) > 0:
                sample_size = min(5, len(df_vin_merged))
                example_vins = df_vin_merged["vin"].sample(n=sample_size, random_state=42).tolist()
            else:
                sample_size = min(5, len(df_vin))
                example_vins = df_vin["vin"].sample(n=sample_size, random_state=42).tolist()

            example_text = ", ".join(example_vins)
            st.markdown(
                f"<p style='font-size: 0.85em; font-style: italic;'>💡 Example VINs: {example_text}</p>",
                unsafe_allow_html=True
            )

        if vin_input:
            # Look up VIN in the VIN lookup table
            vin_match = df_vin[df_vin["vin"].str.upper() == vin_input]

            if len(vin_match) > 0:
                # VIN found - get the vehicle_catalog_name
                selected_vehicle = vin_match.iloc[0]["vehicle_catalog_name"]
                st.success(f"✓ Vehicle selected: {selected_vehicle}")
            else:
                st.error(
                    f"❌ VIN '{vin_input}' not found in database. Please check the VIN or try another search method."
                )
                st.stop()
        else:
            st.warning("⚠️ Please enter a VIN to continue")
            st.stop()

    elif selection_method == "Vehicle ID Search":
        # Option A: Direct selection by Vehicle ID
        vehicle_names = [""] + sorted(df_main_filtered["vehicle_catalog_name"].unique())
        selected_vehicle = st.selectbox(
            "Vehicle ID:",
            vehicle_names,
            help="Start typing to filter and narrow down the list"
        )
        st.markdown(
            "<p style='font-size: 0.85em; font-style: italic;'>💡 Tip: Click the dropdown and start typing to search for a vehicle ID</p>",
            unsafe_allow_html=True
        )

        if not selected_vehicle:
            st.warning("⚠️ Please enter or select a Vehicle ID to continue")
            st.stop()

    else:
        # Option B: Dependent filters - all visible but with cascading options
        st.markdown("#### Filter by vehicle attributes:")

        col1, col2 = st.columns(2)

        # Start with full dataset
        df_temp = df_main_filtered.copy()

        with col1:
            # 1. Model Year
            model_years = [""] + sorted(df_temp["model_year"].dropna().unique())
            selected_year = st.selectbox("Model Year", model_years)

            # Filter based on year selection
            if selected_year:
                df_temp = df_temp[df_temp["model_year"] == selected_year]

            # 2. Manufacturer
            manufacturers = [""] + sorted(df_temp["manufacturer"].dropna().unique())
            selected_manufacturer = st.selectbox("Manufacturer", manufacturers)

            # Filter based on manufacturer selection
            if selected_manufacturer:
                df_temp = df_temp[df_temp["manufacturer"] == selected_manufacturer]

            # 3. Model
            models = [""] + sorted(df_temp["model"].dropna().unique())
            selected_model = st.selectbox("Model", models)

            # Filter based on model selection
            if selected_model:
                df_temp = df_temp[df_temp["model"] == selected_model]

            # 4. Body Style Group
            body_styles = [""] + sorted(df_temp["body_style_group"].dropna().unique())
            selected_body_style = st.selectbox("Body Style", body_styles)

            # Filter based on body style selection
            if selected_body_style:
                df_temp = df_temp[df_temp["body_style_group"] == selected_body_style]

        with col2:
            # 5. Fuel Type
            if "fuel_type" in df_temp.columns:
                fuel_types = [""] + sorted(df_temp["fuel_type"].dropna().unique())
                selected_fuel = st.selectbox("Fuel Type", fuel_types)

                # Filter based on fuel type selection
                if selected_fuel:
                    df_temp = df_temp[df_temp["fuel_type"] == selected_fuel]
            else:
                selected_fuel = None

            # 6. Transmission
            if "transmission" in df_temp.columns:
                transmissions = [""] + sorted(df_temp["transmission"].dropna().unique())
                selected_transmission = st.selectbox("Transmission", transmissions)

                # Filter based on transmission selection
                if selected_transmission:
                    df_temp = df_temp[df_temp["transmission"] == selected_transmission]
            else:
                selected_transmission = None

            # 7. Derivative (if available)
            if "derivative" in df_temp.columns:
                derivatives = [""] + sorted(df_temp["derivative"].dropna().unique())
                selected_derivative = st.selectbox("Derivative", derivatives)

                # Filter based on derivative selection
                if selected_derivative:
                    df_temp = df_temp[df_temp["derivative"] == selected_derivative]
            else:
                selected_derivative = None

        # Check if all required fields are filled
        required_fields = [selected_year, selected_manufacturer, selected_model, selected_body_style]
        if "fuel_type" in df_main_filtered.columns:
            required_fields.append(selected_fuel)
        if "transmission" in df_main_filtered.columns:
            required_fields.append(selected_transmission)
        if "derivative" in df_main_filtered.columns:
            required_fields.append(selected_derivative)

        if not all(required_fields):
            st.warning("⚠️ Please select all attributes to continue.")
            st.stop()

        # Use the filtered dataframe
        df_filtered = df_temp

        # Get final vehicle
        if len(df_filtered) == 1:
            selected_vehicle = df_filtered["vehicle_catalog_name"].iloc[0]
            st.success(f"✓ Vehicle selected: {selected_vehicle}")
        elif len(df_filtered) > 1:
            selected_vehicle = st.selectbox(
                "Multiple matches found. Select one:",
                df_filtered["vehicle_catalog_name"].unique()
            )
        else:
            st.warning("No vehicles match the selected criteria.")

    # Continue only if vehicle is selected
    if selected_vehicle:
        st.markdown("---")

        # Get vehicle data
        df_vehicle_match = df_main_filtered[
            df_main_filtered["vehicle_catalog_name"] == selected_vehicle
        ]
        if df_vehicle_match.empty:
            st.error(
                "❌ Selected vehicle not found for the current forecast date."
            )
            st.stop()

        vehicle_data = df_vehicle_match.iloc[0]

        # ========== SECTION 3: VEHICLE INFORMATION DISPLAY ==========
        st.markdown("""
            <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                        padding: 12px 16px; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4; 
                        margin-bottom: 16px;'>
                <h3 style='margin: 0; color: #262730;'>📋 Vehicle Information</h3>
            </div>
        """, unsafe_allow_html=True)

        # Row 1: Forecast Price (full width)
        st.metric(
            "Typical Price",
            f"€{int(vehicle_data['raw_fst']):,}",
            help="Typical mileage, Grade 1, Base Region"
        )

        # Row 2: Typical characteristics (4 columns)
        col1, col2, col3, col4 = st.columns(4)

        # Grade labels mapping
        grade_labels = {
            1: "Excellent",
            2: "Very Good",
            3: "Good",
            4: "Fair",
            5: "Poor",
            6: "Very Poor"
        }

        with col1:
            st.metric("Typical Mileage", f"{vehicle_data['typical_km']:,.0f} km")

        with col2:
            grade_num = int(vehicle_data["typical_condition"])
            grade_label = grade_labels.get(grade_num, "Unknown")
            st.metric("Typical Condition", f"Grade {grade_num} - {grade_label}")

        with col3:
            st.metric("Typical Region", "Base Region")

        with col4:
            # Empty column for alignment
            st.write("")

        # Row 3: Transaction counts (4 columns)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "transaction_count_6mo" in vehicle_data and pd.notnull(vehicle_data["transaction_count_6mo"]):
                st.metric("Transaction Count (6 months)", f"{int(vehicle_data['transaction_count_6mo']):,}")
            else:
                st.metric("Transaction Count (6 months)", "N/A")

        with col2:
            if "transaction_count_4wk" in vehicle_data and pd.notnull(vehicle_data["transaction_count_4wk"]):
                st.metric("Transaction Count (4 weeks)", f"{int(vehicle_data['transaction_count_4wk']):,}")
            else:
                st.metric("Transaction Count (4 weeks)", "N/A")

        with col3:
            # Empty column for alignment
            st.write("")

        with col4:
            # Empty column for alignment
            st.write("")

        # Row 4: Market data (4 columns)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "median_sold_price" in vehicle_data and pd.notnull(vehicle_data["median_sold_price"]):
                st.metric("Median Sold Price", f"€{int(vehicle_data['median_sold_price']):,}")
            else:
                st.metric("Median Sold Price", "N/A")

        with col2:
            if "median_msrp" in vehicle_data and pd.notnull(vehicle_data["median_msrp"]):
                st.metric("Median MSRP", f"€{int(vehicle_data['median_msrp']):,}")
            else:
                st.metric("Median MSRP", "N/A")

        with col3:
            if "median_mileage" in vehicle_data and pd.notnull(vehicle_data["median_mileage"]):
                st.metric("Median Mileage", f"{vehicle_data['median_mileage']:,.0f} km")
            else:
                st.metric("Median Mileage", "N/A")

        with col4:
            # Empty column for alignment
            st.write("")

        st.markdown("---")

        # ========== SECTION 4: CUSTOM PRICING ADJUSTMENTS ==========
        st.markdown("""
            <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                        padding: 12px 16px; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4; 
                        margin-bottom: 16px;'>
                <h3 style='margin: 0; color: #262730;'>⚙️ Custom Pricing Adjustments</h3>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Input 1: Target Mileage
            target_mileage = st.number_input(
                "Target Mileage (km)",
                min_value=0,
                max_value=300000,
                value=int(vehicle_data["typical_km"]),
                step=1000,
                help="Enter the actual mileage of the vehicle (maximum: 300,000 km)"
            )

        with col2:
            # Input 2: Vehicle Condition
            grade_options = {
                1: "Grade 1 - Excellent (Typical Condition)",
                2: "Grade 2 - Very Good",
                3: "Grade 3 - Good",
                4: "Grade 4 - Fair",
                5: "Grade 5 - Poor",
                6: "Grade 6 - Very Poor"
            }

            selected_grade = st.selectbox(
                "Target Vehicle Condition",
                options=list(grade_options.keys()),
                format_func=lambda x: grade_options[x],
                index=0,
                help="Select the condition grade of the vehicle"
            )

        with col3:
            # Input 3: Vehicle Region
            # Normalize column names for Snowflake compatibility
            df_region_normalized = df_region.copy()
            df_region_normalized.columns = df_region_normalized.columns.str.lower()

            # Get available countries for this vehicle from region lookup (without date filter)
            region_options = df_region_normalized[
                df_region_normalized["vehicle_catalog_name"] == selected_vehicle
            ]

            # Build country list: National (default) + available countries
            country_list = ["Base Region (Typical Region)"]
            if len(region_options) > 0:
                available_countries = sorted(region_options["vehicle_country"].unique())
                country_list.extend(available_countries)

            selected_country = st.selectbox(
                "Target Vehicle Region",
                country_list,
                index=0,
                help="Select 'National' for domestic market (no regional adjustment) or choose a specific country"
            )

        st.markdown("---")

        # ========== SECTION 5: ADJUSTED PRICING SUMMARY ==========
        st.markdown("""
            <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                        padding: 12px 16px; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4; 
                        margin-bottom: 16px;'>
                <h3 style='margin: 0; color: #262730;'>💰 Adjusted Pricing Summary</h3>
            </div>
        """, unsafe_allow_html=True)

        # Get base values
        raw_fst = vehicle_data["raw_fst"]
        body_style_group = vehicle_data["body_style_group"]
        typical_km = vehicle_data["typical_km"]
        t_kmfunction = vehicle_data["t_kmfunction"]
        km_coef = vehicle_data["km_coef"]

        # 1. Calculate KM Adjustment
        target_kmfunction = calculate_km_function(body_style_group, target_mileage)
        km_adjustment = np.exp(km_coef * (target_kmfunction - t_kmfunction))

        # 2. Get Condition Adjustment
        # Normalize column names to lowercase for Snowflake compatibility
        df_condition_normalized = df_condition.copy()
        df_condition_normalized.columns = df_condition_normalized.columns.str.lower()

        # Convert forecast_date to string format for comparison if needed
        if "forecast_date" in df_condition_normalized.columns:
            df_condition_normalized["forecast_date"] = pd.to_datetime(
                df_condition_normalized["forecast_date"]
            ).dt.strftime("%Y-%m-%d")

        condition_lookup = df_condition_normalized[
            (df_condition_normalized["vehicle_catalog_name"] == selected_vehicle)
            & (df_condition_normalized["grade_num"] == selected_grade)
            & (df_condition_normalized["forecast_date"] == selected_forecast_date)
        ]

        if len(condition_lookup) > 0:
            condition_adjustment = condition_lookup["condition_adjustment"].iloc[0]
        else:
            condition_adjustment = 1.0
            if selected_grade != 1:
                st.warning(
                    f"⚠️ No condition adjustment found for Grade {selected_grade} on {selected_forecast_date}. Using 1.0"
                )

        # 3. Get Region Adjustment
        if selected_country == "Base Region (Typical Region)":
            # National = domestic market, no adjustment needed (raw_fst is already for national market)
            region_adjustment = 1.0
            display_country = "Base Region"
        else:
            display_country = selected_country
            # Normalize column names to lowercase for Snowflake compatibility
            df_region_normalized = df_region.copy()
            df_region_normalized.columns = df_region_normalized.columns.str.lower()

            # Look up adjustment for selected country
            region_lookup = df_region_normalized[
                (df_region_normalized["vehicle_catalog_name"] == selected_vehicle)
                & (df_region_normalized["vehicle_country"] == selected_country)
                & (df_region_normalized["forecast_date"] == selected_forecast_date)
            ]

            if len(region_lookup) > 0:
                region_adjustment = region_lookup["region_adjustment"].iloc[0]
            else:
                region_adjustment = 1.0
                st.warning(f"⚠️ No region adjustment found for {selected_country}. Using 1.0")

        # 4. Calculate Final Adjusted Price
        price_after_km = raw_fst * km_adjustment
        price_after_condition = price_after_km * condition_adjustment
        adjusted_price = price_after_condition * region_adjustment

        # Display adjustment factors
        st.markdown("#### Adjustment Factors")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "KM Adjustment",
                f"{km_adjustment:.4f}",
                help=f"Based on {target_mileage:,.0f} km vs typical {typical_km:,.0f} km"
            )

        with col2:
            st.metric(
                "Condition Adjustment",
                f"{condition_adjustment:.4f}",
                help=f"Based on Grade {selected_grade} vs Grade 1 (Typical Condition)"
            )

        with col3:
            st.metric(
                "Region Adjustment",
                f"{region_adjustment:.4f}",
                help=f"Based on {display_country} vs Base Region (Typical Region)"
            )

        # Display price results
        st.markdown("#### Final Price")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Typical Price",
                f"€{int(raw_fst):,}",
                help="Typical mileage, Grade 1, Base Region"
            )

        with col2:
            price_change = adjusted_price - raw_fst
            if price_change == 0:
                # No change - no arrow, black color
                st.metric(
                    "Final Price",
                    f"€{int(adjusted_price):,}",
                    delta="€0",
                    delta_color="off"
                )
            elif price_change > 0:
                # Increase - green arrow
                st.metric(
                    "Final Price",
                    f"€{int(adjusted_price):,}",
                    delta=f"€{int(price_change):,}",
                    delta_color="normal"
                )
            else:
                # Decrease - red arrow
                st.metric(
                    "Final Price",
                    f"€{int(adjusted_price):,}",
                    delta=f"-€{int(abs(price_change)):,}",
                    delta_color="normal"
                )

        with col3:
            price_diff_pct = ((adjusted_price - raw_fst) / raw_fst) * 100
            st.metric(
                "Price Difference",
                f"{price_diff_pct:+.2f}%",
                help="Percentage change from typical price"
            )

        st.markdown("---")

        # ========== VISUALIZATION SECTION ==========
        st.markdown("""
            <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                        padding: 12px 16px; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4; 
                        margin-bottom: 16px;'>
                <h3 style='margin: 0; color: #262730;'>📊 Adjustment Visualization</h3>
            </div>
        """, unsafe_allow_html=True)

        # Create a visual representation of the price adjustments
        import plotly.graph_objects as go

        # Data for waterfall chart
        x_labels = ["Typical Price", "KM<br>Adjustment", "Condition<br>Adjustment", "Region<br>Adjustment", "Final Price"]

        km_delta = (price_after_km - raw_fst)
        condition_delta = (price_after_condition - price_after_km)
        region_delta = (adjusted_price - price_after_condition)

        # Format impact text with arrows and colors
        km_arrow = "↑" if km_delta >= 0 else "↓"
        km_color = "#2ecc71" if km_delta >= 0 else "#e74c3c"

        condition_arrow = "↑" if condition_delta >= 0 else "↓"
        condition_color = "#2ecc71" if condition_delta >= 0 else "#e74c3c"

        region_arrow = "↑" if region_delta >= 0 else "↓"
        region_color = "#2ecc71" if region_delta >= 0 else "#e74c3c"

        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Price Adjustments",
            x=x_labels,
            y=[raw_fst, km_delta, condition_delta, region_delta, adjusted_price],
            measure=["absolute", "relative", "relative", "relative", "total"],
            text=[
                f"€{int(raw_fst):,}",
                f"€{km_delta:+,.0f}",
                f"€{condition_delta:+,.0f}",
                f"€{region_delta:+,.0f}",
                f"€{int(adjusted_price):,}",
            ],
            textposition=["outside", "inside", "inside", "inside", "outside"],
            textfont=dict(size=14),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
            width=[0.5, 0.5, 0.5, 0.5, 0.5]
        ))

        # Calculate appropriate y-axis range for better granularity
        all_values = [raw_fst, price_after_km, price_after_condition, adjusted_price]
        y_min = min(all_values) * 0.95
        y_max = max(all_values) * 1.05
        y_range = y_max - y_min

        # Set tick interval based on range (smaller intervals for better visibility)
        tick_interval = y_range / 20

        fig.update_layout(
            showlegend=False,
            height=550,
            yaxis=dict(
                title=dict(text="Price (€)", font=dict(size=16, color="black")),
                tickfont=dict(size=14, color="black"),
                range=[y_min, y_max],
                dtick=tick_interval,
                gridcolor="rgba(180, 180, 180, 0.4)",
                tickformat=",.0f"
            ),
            xaxis=dict(
                title="",
                tickfont=dict(size=14, color="black")
            ),
            font=dict(size=12),
            bargap=0.3,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=100, r=50, t=30, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add summary text with arrows and colors (centered)
        col1, col2, col3 = st.columns(3)

        with col1:
            if km_delta == 0:
                st.markdown(
                    "<div style='text-align: center'><b>KM Impact:</b> €{:,.0f}</div>".format(km_delta),
                    unsafe_allow_html=True
                )
            else:
                km_color_html = "green" if km_delta > 0 else "red"
                km_arrow = "↑" if km_delta > 0 else "↓"
                st.markdown(
                    f"<div style='text-align: center'><b>KM Impact:</b> <span style='color:{km_color_html}'>{km_arrow} €{abs(km_delta):,.0f}</span></div>",
                    unsafe_allow_html=True
                )

        with col2:
            if condition_delta == 0:
                st.markdown(
                    "<div style='text-align: center'><b>Condition Impact:</b> €{:,.0f}</div>".format(condition_delta),
                    unsafe_allow_html=True
                )
            else:
                condition_color_html = "green" if condition_delta > 0 else "red"
                condition_arrow = "↑" if condition_delta > 0 else "↓"
                st.markdown(
                    f"<div style='text-align: center'><b>Condition Impact:</b> <span style='color:{condition_color_html}'>{condition_arrow} €{abs(condition_delta):,.0f}</span></div>",
                    unsafe_allow_html=True
                )

        with col3:
            if region_delta == 0:
                st.markdown(
                    "<div style='text-align: center'><b>Region Impact:</b> €{:,.0f}</div>".format(region_delta),
                    unsafe_allow_html=True
                )
            else:
                region_color_html = "green" if region_delta > 0 else "red"
                region_arrow = "↑" if region_delta > 0 else "↓"
                st.markdown(
                    f"<div style='text-align: center'><b>Region Impact:</b> <span style='color:{region_color_html}'>{region_arrow} €{abs(region_delta):,.0f}</span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ========== MORE INFO SECTION ==========
        st.markdown("""
            <div style='background: linear-gradient(90deg, #e3f2fd, #f5f5f5); 
                        padding: 12px 16px; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4; 
                        margin-bottom: 16px;'>
                <h3 style='margin: 0; color: #262730;'>ℹ️ More Info</h3>
            </div>
        """, unsafe_allow_html=True)

        info_option = st.selectbox(
            "Select information to view:",
            ["-- Select to view --", "Detailed Pricing Adjustments Breakdown", "Source Data Summary"],
            index=0
        )

        if info_option == "Detailed Pricing Adjustments Breakdown":
            # Starting Point
            st.write("**📍 Starting Point:**")
            st.write(f"- Typical Price: €{int(raw_fst):,}")
            st.write(f"- Typical Mileage: {typical_km:,.0f} km")
            st.write("- Typical Condition: Grade 1 - Excellent")
            st.write("- Typical Region: Base Region")

            st.markdown("---")

            # Input Summary
            st.write("**📝 Input Summary:**")
            st.write(f"- Selected Mileage: {target_mileage:,.0f} km (vs typical {typical_km:,.0f} km)")
            st.write(
                f"- Selected Condition: Grade {selected_grade} (vs typical Grade {int(vehicle_data['typical_condition'])})"
            )
            st.write(f"- Selected Region: {display_country} (vs Typical Base Region)")

            st.markdown("---")

            # Adjustment Factors Applied
            st.write("**⚙️ Adjustment Factors Applied:**")
            st.write(f"- KM: {km_adjustment:.4f}")
            st.write(f"- Condition: {condition_adjustment:.4f}")
            st.write(f"- Region: {region_adjustment:.4f}")

            st.markdown("---")

            # Technical Details
            st.write("**🔧 Technical Details:**")
            st.write(f"- Target KM Function: {target_kmfunction:.6f}")
            st.write(f"- Typical KM Function: {t_kmfunction:.6f}")
            st.write(f"- KM Coefficient: {km_coef:.6f}")
            st.write(f"- Vehicle Selected: {selected_vehicle}")

            st.markdown("---")

            # Adjustment Steps
            st.write("**📊 Adjustment Steps:**")
            st.write(f"1. After KM Adjustment ({km_adjustment:.4f}): €{int(price_after_km):,}")
            st.write(
                f"2. After Condition Adjustment ({condition_adjustment:.4f}): €{int(price_after_condition):,}"
            )
            st.write(
                f"3. After Region Adjustment ({region_adjustment:.4f}): €{int(adjusted_price):,} → Final Price: €{int(adjusted_price):,}"
            )

        elif info_option == "Source Data Summary":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Main Vehicle Records", f"{len(df_main):,}")
            with col2:
                st.metric("Region Lookup Records", f"{len(df_region):,}")
            with col3:
                st.metric("Condition Lookup Records", f"{len(df_condition):,}")
            with col4:
                st.metric("VIN Lookup Records", f"{len(df_vin):,}")


if __name__ == "__main__":
    main()
