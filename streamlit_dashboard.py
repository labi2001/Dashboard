import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


@st.cache_data
def load_data():
    df = pd.read_csv("PrimeFrontier_SolarDeploymentDataset.csv")
    return df


def create_solar_site_suitability_score(df):
    df_processed = df.copy()
    scaler = MinMaxScaler()

    # Invert features where lower is better
    df_processed["Grid_Access_inverse"] = 100 - df_processed["Grid_Access_Percent"]
    df_processed["Terrain_Ruggedness_inverse"] = (
        df_processed["Terrain_Ruggedness_Score"].max()
        - df_processed["Terrain_Ruggedness_Score"]
    )

    # Normalize features
    df_processed["Solar_Irradiance_norm"] = scaler.fit_transform(
        df_processed[["Solar_Irradiance_kWh_m2_day"]]
    )
    df_processed["Infrastructure_Index_norm"] = scaler.fit_transform(
        df_processed[["Infrastructure_Index"]]
    )
    df_processed["Electricity_Cost_norm"] = scaler.fit_transform(
        df_processed[["Electricity_Cost_USD_per_kWh"]]
    )
    df_processed["Grid_Access_inverse_norm"] = scaler.fit_transform(
        df_processed[["Grid_Access_inverse"]]
    )
    df_processed["Rural_Pop_Density_norm"] = scaler.fit_transform(
        df_processed[["Rural_Pop_Density_per_km2"]]
    )
    df_processed["Terrain_Ruggedness_inverse_norm"] = scaler.fit_transform(
        df_processed[["Terrain_Ruggedness_inverse"]]
    )

    weights = {
        "Solar_Irradiance_norm": 0.3,
        "Grid_Access_inverse_norm": 0.2,
        "Infrastructure_Index_norm": 0.15,
        "Electricity_Cost_norm": 0.15,
        "Rural_Pop_Density_norm": 0.1,
        "Terrain_Ruggedness_inverse_norm": 0.1,
    }

    df_processed["Solar_Site_Suitability_Score"] = (
        df_processed["Solar_Irradiance_norm"] * weights["Solar_Irradiance_norm"]
        + df_processed["Grid_Access_inverse_norm"] * weights["Grid_Access_inverse_norm"]
        + df_processed["Infrastructure_Index_norm"]
        * weights["Infrastructure_Index_norm"]
        + df_processed["Electricity_Cost_norm"] * weights["Electricity_Cost_norm"]
        + df_processed["Rural_Pop_Density_norm"] * weights["Rural_Pop_Density_norm"]
        + df_processed["Terrain_Ruggedness_inverse_norm"]
        * weights["Terrain_Ruggedness_inverse_norm"]
    )

    return df_processed


def main():
    df = load_data()
    st.markdown(
        "<h1 style='font-size:30px;'>Solar Site Suitability & Region Metrics Dashboard</h1>",
        unsafe_allow_html=True,
    )

    # Calculate solar site suitability scores for all regions once
    df_with_scores = create_solar_site_suitability_score(df)

    # Sidebar: region selection
    regions = df_with_scores["Region"].unique()
    selected_region = st.selectbox("Select Region", regions)

    # Filter data for selected region
    region_data = df_with_scores[df_with_scores["Region"] == selected_region]

    if not region_data.empty:
        st.subheader(f"Metrics for {selected_region}")

        # Show metrics in a table (excluding Region and the intermediate norm columns)
        display_cols = [
            "Solar_Irradiance_kWh_m2_day",
            "Rural_Pop_Density_per_km2",
            "Grid_Access_Percent",
            "Infrastructure_Index",
            "Electricity_Cost_USD_per_kWh",
            "Terrain_Ruggedness_Score",
            "Solar_Site_Suitability_Score",
        ]
        metrics_table = region_data[display_cols].T
        metrics_table.columns = ["Value"]
        metrics_table = metrics_table.reset_index()
        metrics_table = metrics_table.rename(columns={"index": "Metric"})
        st.table(metrics_table)
    else:
        st.write("No data available for selected region.")

    # Ranked list of all regions by Solar_Site_Suitability_Score (descending)
    ranked_df = (
        df_with_scores[["Region", "Solar_Site_Suitability_Score"]]
        .sort_values(by="Solar_Site_Suitability_Score", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Ranked List of Regions by Solar Site Suitability Score")
    top_10 = ranked_df.head(10)

    # Show full ranked table (all regions)
    ranked_df.index += 1
    st.table(ranked_df.rename_axis("Rank").reset_index())

    # Bar chart of top 10 scores
    st.subheader("Top 10 Regions by Solar Site Suitability Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_10["Region"], top_10["Solar_Site_Suitability_Score"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Solar Site Suitability Score")
    ax.set_title("Top 10 Solar Site Suitability Scores")
    st.pyplot(fig)

    st.subheader("Summary Recommendations")

    best_region = ranked_df.iloc[0]["Region"]
    best_score = ranked_df.iloc[0]["Solar_Site_Suitability_Score"]

    high_suitability_threshold = 0.60
    num_high_suitability = (
        ranked_df["Solar_Site_Suitability_Score"] >= high_suitability_threshold
    ).sum()

    st.markdown(
        f"""
    ### Solar Site Suitability Summary

    - The best region for solar deployment is **{best_region}** with a suitability score of **{best_score:.2f}**.
    - There are **{num_high_suitability}** regions with a solar site suitability score above **{high_suitability_threshold}**, indicating strong potential for solar projects.
    - Regions with higher suitability scores typically feature:
      - High solar irradiance,
      - Low electricity costs,
      - Robust infrastructure and grid access.

    ### Recommendations for Solar Deployment

    - **Prioritize development** in the top-ranked regions to maximize solar energy yield and return on investment.
    - **Focus initial projects** in regions scoring above the suitability threshold to ensure feasibility and impact.
    - **Customize strategies** for regions with low grid access or challenging terrain by investing in off-grid or microgrid solutions.
    - **Allocate resources strategically** using the ranked list for phased solar deployment and expansion planning.
    - **Monitor and update** the suitability scores regularly to reflect changes in infrastructure, costs, or population dynamics.

    ---
    """
    )


if __name__ == "__main__":
    main()
