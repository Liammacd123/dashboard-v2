import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="BAM Analytix Marketing: Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for a Subdued Background
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #a9a9a9;  /* DarkGray */
    }
    .sidebar .sidebar-content {
        background-color: #808080;  /* Gray */
    }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------------
# Helper Function: Layered Boxplot
# -------------------------------
def layered_boxplot(df, field, y_field, tooltip_fields, width=300, height=300, box_size=80, point_color='red'):
    """
    Creates a layered Altair chart that displays a boxplot for groups with more than one observation
    and overlays a point for groups with a single occurrence.
    """
    tooltips = [alt.Tooltip(f'{col}:Q', title=title) for col, title in tooltip_fields]

    box_layer = alt.Chart(df).transform_joinaggregate(
        count='count()',
        groupby=[field]
    ).transform_filter(
        alt.datum.count > 1
    ).mark_boxplot(extent='min-max', size=box_size).encode(
        x=alt.X(f'{field}:O', title=field),
        y=alt.Y(f'{y_field}:Q', title=y_field),
        tooltip=tooltips
    ).properties(width=width, height=height)

    point_layer = alt.Chart(df).transform_joinaggregate(
        count='count()',
        groupby=[field]
    ).transform_filter(
        alt.datum.count == 1
    ).mark_point(size=100, color=point_color).encode(
        x=alt.X(f'{field}:O', title=field),
        y=alt.Y(f'{y_field}:Q', title=y_field),
        tooltip=tooltips
    ).properties(width=width, height=height)

    return box_layer + point_layer

# -------------------------------
# Data Loading with Caching (Main Data)
# -------------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df = df.sort_values(by='Month')
    df['Month'] = df['Month'].dt.strftime('%b %Y')
    return df

file_path = "All Performance Data.xlsx"  # Adjust as needed
df = load_data(file_path)
x_axis_order = df['Month'].unique()

# -------------------------------
# Data for Cost Per Impression and Impressions per Dollar
# -------------------------------
cpi_data = pd.DataFrame({
    "Campaign": [
        "Canadian Politics and Earned Media (BAM Page)",
        "BBID (Survey vs Volumetric, Brett’s Page) - NY, TX, CA",
        "BBID (Dr. Pepper; Brett’s Page)",
        "BBID (Brand Hall of Fame; BAM Page)"
    ],
    "Total Impressions": [2026150, 5124, 6472, 6156],
    "Cost per Impression": [0.000049, 0.0585, 0.0273, 0.0346]
}).sort_values(by="Cost per Impression", ascending=True)

imp_per_dollar_data = {
    "Canadian Politics and Earned Media (BAM Page)": 20408.16,
    "BBID (Survey vs Volumetric, Brett’s Page) - NY, TX, CA": 17.09,
    "BBID (Dr. Pepper; Brett’s Page)": 36.63,
    "BBID (Brand Hall of Fame; BAM Page)": 28.90
}
imp_per_dollar_df = (
    pd.DataFrame(list(imp_per_dollar_data.items()), columns=["Campaign", "Impressions per Dollar"])
    .sort_values(by="Impressions per Dollar", ascending=False)
)

# -------------------------------
# Data Loading for BAM LI Analysis (Beta View Data)
# -------------------------------
@st.cache_data
def load_bam_li_data(file_path):
    df_bam_li = pd.read_excel(file_path)
    return df_bam_li

# -------------------------------
# Lasso Model Training on BAM LI Data (without Engagement rate)
# -------------------------------
@st.cache_data
def train_lasso_model(df_model):
    """
    Trains a Lasso regression model using the provided dataframe.
    Preprocessing mimics the training procedure:
      - Converts key columns to numeric.
      - Converts "Created date" to datetime and extracts day of week.
      - Fills missing values in "Content Type" (default "Image") and applies an override.
      - Label-encodes "Content Type" into "Content Type Encoded".
      - Drops rows with missing values.
    Final features used:
      ["Likes", "Comments", "Reposts", "Clicks", "Day of Week", "Content Type Encoded"]
    Target: "Impressions"
    """
    df_model = df_model.copy()
    
    # Convert columns to numeric
    cols_numeric = ["Likes", "Comments", "Reposts", "Clicks", "Impressions"]
    df_model[cols_numeric] = df_model[cols_numeric].apply(pd.to_numeric, errors="coerce")
    
    # Convert "Created date" to datetime and extract day of week (Monday=0, Sunday=6)
    if "Created date" in df_model.columns:
        df_model["Created date"] = pd.to_datetime(df_model["Created date"], errors="coerce")
        df_model["Day of Week"] = df_model["Created date"].dt.dayofweek
    else:
        df_model["Day of Week"] = 0
    
    # Handle missing values in "Content Type"
    if "Content Type" in df_model.columns:
        df_model["Content Type"] = df_model["Content Type"].fillna("Image")
        if "Created date" in df_model.columns:
            df_model.loc[df_model["Created date"] == pd.to_datetime("2024-10-15"), "Content Type"] = "Poll"
    else:
        df_model["Content Type"] = "Image"
    
    # Label Encode "Content Type" into "Content Type Encoded"
    label_encoder = LabelEncoder()
    df_model["Content Type Encoded"] = label_encoder.fit_transform(df_model["Content Type"])
    
    # Drop rows with missing values in required columns
    required_cols = cols_numeric + ["Day of Week", "Content Type Encoded"]
    df_model = df_model.dropna(subset=required_cols)
    
    features = ["Likes", "Comments", "Reposts", "Clicks", "Day of Week", "Content Type Encoded"]
    X = df_model[features]
    y = df_model["Impressions"]
    
    model = Lasso(alpha=0.1)
    model.fit(X, y)
    
    # (Optional) Compute performance metrics:
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # st.write(f"Trained Lasso Model: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    return model

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.header("Dashboard Navigation")
view = st.sidebar.selectbox("Select View:", [
    "Summary",
    "Cost Per Impression",
    "Total Organic Impressions by Month",
    "Contribution by Channel per Month (Organic)",
    "Total Impressions (Organic + Sponsored) & Web Traffic",
    "Contribution by Channel per Month (Organic + Paid)",
    "Cumulative Impressions CTD",
    "(Beta) Linkedin Performance Predictor"
])

# -------------------------------
# View: Summary
# -------------------------------
if view == "Summary":
    st.markdown("## Key Performance Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Peak Organic Month")
        st.metric("", "Oct 2024")
    with col2:
        st.subheader("Peak Total Month (Organic + Sponsored)")
        st.metric("", "Jan 2025")
    
    st.markdown("### Top 3 Organic Posts by Channel")
    st.markdown(
        """
- **Organic LinkedIn:** Brand Hall of Fame (551), AI Brett Breaks it Down (542), Wheel of Competitive Advantage (456)
- **Organic X:** Tik Tok vs RedNote (92), Social Media as a Spectrum (64), BBID Brand Hall of Fame (55)
- **Organic Tik Tok:** BBID Mass Merchant (341), BBID Entertainment Category (230), BBID Brick and Mortar (222)
        """
    )
    st.markdown("### Top Paid Campaigns by Channel")
    st.markdown(
        """
- **Sponsored LinkedIn (BAM):** Canadian Politics and Earned Media (2,027,016)
- **Sponsored LinkedIn (Brett):** BBID Dr. Pepper (6,472)
        """
    )

# -------------------------------
# View: Cost Per Impression (Plotly)
# -------------------------------
elif view == "Cost Per Impression":
    st.markdown("## Cost Per Impression")
    st.dataframe(
        cpi_data.style.format({
            "Total Impressions": "{:,}",
            "Cost per Impression": "${:,.6f}"
        })
    )
    fig = go.Figure()
    for _, row in imp_per_dollar_df.iterrows():
        fig.add_trace(go.Bar(
            y=[row["Campaign"]],
            x=[row["Impressions per Dollar"]],
            text=f"{row['Impressions per Dollar']:.2f}",
            textposition='outside',
            orientation='h',
            marker=dict(color='green')
        ))
    fig.update_layout(
        title="Impressions per Dollar by Campaign",
        xaxis_title="Impressions per Dollar",
        yaxis=dict(categoryorder="total descending"),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# View: Total Organic Impressions by Month (Altair)
# -------------------------------
elif view == "Total Organic Impressions by Month":
    st.markdown("## Total Organic Impressions by Month")
    channels_org = ["Organic LinkedIn Impressions", "Organic X", "Organic Tik Tok"]
    df_melted_org = df[["Month"] + channels_org].melt(
        id_vars="Month",
        var_name="Channel",
        value_name="Impressions"
    )
    chart_org = (
        alt.Chart(df_melted_org)
        .mark_bar()
        .encode(
            x=alt.X("Month:N", sort=list(x_axis_order)),
            y=alt.Y("Impressions:Q", title="Total Impressions"),
            color=alt.Color("Channel:N", title="Channel"),
            tooltip=["Month", "Channel", "Impressions"]
        )
        .properties(title="Total Organic Impressions by Month", width=700, height=400)
        .configure_axisX(labelAngle=45)
    )
    st.altair_chart(chart_org, use_container_width=True)

# -------------------------------
# View: Contribution by Channel per Month (Organic, Plotly)
# -------------------------------
elif view == "Contribution by Channel per Month (Organic)":
    st.markdown("## Contribution by Channel per Month (Organic)")
    month_selected = st.sidebar.selectbox("Select Month:", sorted(df["Month"].unique()))
    channels = ["Organic LinkedIn Impressions", "Organic X", "Organic Tik Tok"]
    df_filtered = df[df["Month"] == month_selected]
    values = [df_filtered[ch].sum() for ch in channels]
    fig = go.Figure(go.Pie(
        labels=channels,
        values=values,
        hoverinfo='label+percent',
        textinfo='value'
    ))
    fig.update_layout(title=f"Contribution by Channel (Organic) - {month_selected}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# View: Total Impressions (Organic + Sponsored) & Web Traffic (Altair)
# -------------------------------
elif view == "Total Impressions (Organic + Sponsored) & Web Traffic":
    st.markdown("## Total Impressions (Organic + Sponsored) & Web Traffic (Single-Bar + Dual Axis)")
    channels_all = [
        "Organic LinkedIn Impressions",
        "Organic X",
        "Organic Tik Tok",
        "Sponsored LinkedIn Impressions",
        "Sponsored Linkedin (Brett)"
    ]
    df["Total Impressions"] = df[channels_all].sum(axis=1)
    bar_impressions = alt.Chart(df).mark_bar().encode(
        x=alt.X("Month:N", sort=list(x_axis_order)),
        y=alt.Y("Total Impressions:Q", title="Impressions (Log Scale)",
                scale=alt.Scale(type="log"), axis=alt.Axis(orient="left")),
        tooltip=["Month", "Total Impressions"]
    )
    line_traffic = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Month:N", sort=list(x_axis_order)),
        y=alt.Y("Web Traffic:Q", title="Web Traffic",
                scale=alt.Scale(type="linear"), axis=alt.Axis(orient="right")),
        color=alt.value("gray"),
        tooltip=["Month", "Web Traffic"]
    )
    combo_chart = (
        alt.layer(bar_impressions, line_traffic)
        .resolve_scale(y="independent")
        .properties(title="Total Impressions (Log Scale) & Web Traffic (Linear Scale)",
                    width=700, height=400)
        .configure_axisX(labelAngle=45)
    )
    st.altair_chart(combo_chart, use_container_width=True)

# -------------------------------
# View: Contribution by Channel per Month (Organic + Paid, Plotly)
# -------------------------------
elif view == "Contribution by Channel per Month (Organic + Paid)":
    st.markdown("## Contribution by Channel (Organic + Paid)")
    month_selected = st.sidebar.selectbox("Select Month (Organic + Paid):", sorted(df["Month"].unique()))
    df_filtered = df[df["Month"] == month_selected]
    channels = [
        "Organic LinkedIn Impressions",
        "Organic X",
        "Organic Tik Tok",
        "Sponsored LinkedIn Impressions",
        "Sponsored Linkedin (Brett)"
    ]
    values = [df_filtered[ch].sum() for ch in channels]
    fig = go.Figure(go.Pie(
        labels=channels,
        values=values,
        hoverinfo='label+percent',
        textinfo='value'
    ))
    fig.update_layout(title=f"Contribution by Channel (Organic + Paid) - {month_selected}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# View: Cumulative Impressions CTD (Altair with Log Scale)
# -------------------------------
elif view == "Cumulative Impressions CTD":
    st.markdown("## Cumulative Impressions CTD")
    columns_to_sum = [
        "Organic LinkedIn Impressions",
        "Sponsored LinkedIn Impressions",
        "Organic X",
        "Sponsored Linkedin (Brett)",
        "Organic Tik Tok"
    ]
    df["Cumulative Impressions"] = df[columns_to_sum].sum(axis=1).cumsum()
    chart_cum = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:N", sort=list(x_axis_order), title="Month"),
            y=alt.Y("Cumulative Impressions:Q", 
                    title="Cumulative Impressions",
                    scale=alt.Scale(type="log")),
            tooltip=["Month", "Cumulative Impressions"]
        )
        .properties(
            title="Campaign to Date: Total Cumulative Impressions",
            width=700,
            height=400
        )
        .configure_axisX(labelAngle=45)
    )
    st.altair_chart(chart_cum, use_container_width=True)
    st.markdown("Total Cumulative Impressions as of Feb 1, 2025: 2,054,689")

# -------------------------------
# View: (Beta) Linkedin Performance Predictor
# -------------------------------
elif view == "(Beta) Linkedin Performance Predictor":
    st.markdown("## (Beta) Linkedin Performance Predictor: Layered Altair Boxplots & Prediction")
    
    bam_li_file = "BAM LI Feb 11.xlsx"  # Adjust path if needed
    df_bam_li = load_bam_li_data(bam_li_file)
    
    # Split the screen: left for box plots, right for prediction
    left_col, right_col = st.columns([3, 1])
    
    with left_col:
        # Chart for Comments
        chart_comments = layered_boxplot(
            df_bam_li,
            field='Comments',
            y_field='Impressions',
            tooltip_fields=[
                ('comments_min', 'Min'),
                ('comments_max', 'Max'),
                ('comments_mean', 'Mean')
            ],
            width=600,
            height=600,
            box_size=100,
            point_color='blue'
        )
        st.altair_chart(chart_comments, use_container_width=True)
        
        # Chart for Likes (with thinner boxes)
        chart_likes = layered_boxplot(
            df_bam_li,
            field='Likes',
            y_field='Impressions',
            tooltip_fields=[
                ('likes_min', 'Min'),
                ('likes_max', 'Max'),
                ('likes_mean', 'Mean')
            ],
            width=600,
            height=600,
            box_size=30,  # Reduced box_size to prevent overlapping
            point_color='blue'
        )
        st.altair_chart(chart_likes, use_container_width=True)
        
        # Chart for Reposts
        chart_reposts = layered_boxplot(
            df_bam_li,
            field='Reposts',
            y_field='Impressions',
            tooltip_fields=[
                ('reposts_min', 'Min'),
                ('reposts_max', 'Max'),
                ('reposts_mean', 'Mean')
            ],
            width=600,
            height=600,
            box_size=100,
            point_color='blue'
        )
        st.altair_chart(chart_reposts, use_container_width=True)
    
    with right_col:
        st.markdown("### Prediction")
        st.markdown("**Current accuracy metrics:** R² Score: 0.8456, Mean Absolute Error (MAE): 35.77")
        
        # Train the Lasso model on the BAM LI data
        model_lasso = train_lasso_model(df_bam_li)
        
        # Display feature importance as a small bar plot
        # Build a dataframe for coefficients. The training features are in this order:
        features_list = ["Likes", "Comments", "Reposts", "Clicks", "Day of Week", "Content Type Encoded"]
        coef_df_lasso = pd.DataFrame({
            "Feature": features_list,
            "Coefficient": model_lasso.coef_
        })
        coef_df_sorted = coef_df_lasso.sort_values(by="Coefficient", ascending=False)
        coef_chart = alt.Chart(coef_df_sorted).mark_bar().encode(
            x=alt.X("Coefficient:Q", title="Coefficient"),
            y=alt.Y("Feature:N", sort='-x', title="Feature")
        ).properties(
            width=300,
            height=150,
            title="Feature Importance"
        )
        st.altair_chart(coef_chart, use_container_width=True)
        
        # Input fields for prediction (without # of Clicks)
        input_likes = st.number_input("Number of Likes", min_value=0, value=50)
        input_comments = st.number_input("Number of Comments", min_value=0, value=10)
        input_reposts = st.number_input("Number of Reposts", min_value=0, value=2)
        input_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        input_content = st.selectbox("Content Type", ["Image", "Video", "Poll"])
        
        # Use the mean number of clicks from the BAM LI data as a default value
        default_clicks = df_bam_li["Clicks"].mean() if "Clicks" in df_bam_li.columns else 0
        
        if st.button("Predict Performance"):
            # Map day of week (training uses Monday=0, ..., Sunday=6)
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                       "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            # Use a fixed label encoder for content type (order assumed from training)
            fixed_label_encoder = LabelEncoder()
            fixed_label_encoder.fit(["Image", "Poll", "Video"])
            content_encoded = fixed_label_encoder.transform([input_content])[0]
            
            # Form the feature vector in the same order as training:
            # [Likes, Comments, Reposts, Clicks, Day of Week, Content Type Encoded]
            feature_vector = [
                input_likes,
                input_comments,
                input_reposts,
                default_clicks,
                day_map[input_day],
                content_encoded
            ]
            prediction = model_lasso.predict(np.array([feature_vector]))[0]
            st.success(f"Predicted Performance (Impressions): {prediction:.2f}")
