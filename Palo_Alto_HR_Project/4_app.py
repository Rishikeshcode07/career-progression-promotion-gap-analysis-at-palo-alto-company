import streamlit as st
import pandas as pd
import plotly.express as px
import os

# =====================================================================
# Career Progression & Promotion Gap Analysis — Palo Alto Networks
# Streamlit Retention Intelligence Dashboard
# (Pure Streamlit layout & widgets — no custom CSS / HTML / JS.
#  Visual polish comes from: .streamlit/config.toml theme, container
#  "cards", metric deltas, consistent color maps, column_config
#  progress/number columns, tabs with icons, and a plotly style helper.)
# =====================================================================

st.set_page_config(
    page_title="Palo Alto HR Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Career Progression & Promotion Gap Analysis for Retention "
                  "Optimization - Palo Alto Networks x Unified Mentor."
    },
)

# ---------------------------------------------------------------
# CONSTANTS — consistent color language used across every chart
# ---------------------------------------------------------------
RISK_COLORS = {"Low": "#2E86AB", "Medium": "#F2A541", "High": "#C0392B"}
CLUSTER_COLORS = {
    "Promotion-Stalled / Stagnant Profiles": "#C0392B",
    "Early-Career Explorers / New Joiners": "#6A4C93",
    "Steady Performers / Fast-Trackers": "#2E86AB",
    "Long-term Contributors / Senior Staff": "#3D9970",
}
CONTINUITY_COLORS = {"Low Continuity": "#C0392B", "High Continuity": "#2E86AB"}

px.defaults.template = "plotly_white"


def style_fig(fig, height=None, legend_bottom=False):
    """Apply a consistent, polished look to every Plotly figure."""
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#1B2430"),
        title_font=dict(size=17, color="#1B2430"),
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    if height:
        fig.update_layout(height=height)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0))
    return fig


@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Processed_HR_Data.csv")
    data = pd.read_csv(file_path)

    def career_stage(years):
        if years <= 2:
            return "New Joiner (0-2 yrs)"
        elif years <= 6:
            return "Establishing (3-6 yrs)"
        elif years <= 12:
            return "Established (7-12 yrs)"
        else:
            return "Senior (13+ yrs)"

    data["Career_Stage"] = data["YearsAtCompany"].apply(career_stage)
    return data


df = load_data()
STAGE_ORDER = ["New Joiner (0-2 yrs)", "Establishing (3-6 yrs)", "Established (7-12 yrs)", "Senior (13+ yrs)"]

# Baseline (whole-company) figures, used to give KPI cards a meaningful delta
baseline_total = len(df)
baseline_attr_rate = df["Attrition"].mean() * 100
baseline_high_risk_pct = (df["Promotion_Gap_Risk"] == "High").mean() * 100
baseline_target_pct = (df["Retention_Opportunity"] == "High Priority").mean() * 100
baseline_training_pct = (df["Training_Need_Indicator"] == "Needs Development Plan").mean() * 100

# ---------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------
hero_l, hero_r = st.columns([5, 2])
with hero_l:
    st.title("Palo Alto Networks --> HR Retention Intelligence")
    st.caption("Career Progression & Promotion Gap Analysis · Retention Optimization Dashboard")
with hero_r:
    with st.container(border=True):
        st.caption("Dataset snapshot")
        st.markdown(f"**{baseline_total:,}** employees · **{df['Department'].nunique()}** departments")

st.divider()

# =====================================================================
# SIDEBAR — Control Center
# =====================================================================
st.sidebar.title("Control Center")
st.sidebar.caption("Slice the workforce dynamically - every chart below reacts instantly.")

DEFAULTS = {
    "dept_filter": sorted(df["Department"].unique().tolist()),
    "role_filter": sorted(df["JobRole"].unique().tolist()),
    "stage_filter": STAGE_ORDER,
    "cluster_filter": sorted(df["Cluster_Name"].unique().tolist()),
    "risk_filter": ["Low", "Medium", "High"],
    "gap_threshold": 0.0,
}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def reset_filters():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


with st.sidebar.expander("Workforce Filters", expanded=True):
    dept_filter = st.multiselect("Department", options=sorted(df["Department"].unique()), key="dept_filter")
    role_filter = st.multiselect("Job Role", options=sorted(df["JobRole"].unique()), key="role_filter")
    stage_filter = st.multiselect("Career Stage", options=STAGE_ORDER, key="stage_filter")

with st.sidebar.expander("Career & Risk Filters", expanded=True):
    cluster_filter = st.multiselect("Career Cluster", options=sorted(df["Cluster_Name"].unique()), key="cluster_filter")
    gap_threshold = st.slider(
        "Minimum Promotion Gap Ratio", min_value=0.0, max_value=1.0, step=0.05, key="gap_threshold",
        help="Show only employees whose Promotion_Gap_Ratio is at or above this value."
    )
    risk_filter = st.multiselect("Promotion Gap Risk Level", options=["Low", "Medium", "High"], key="risk_filter")

st.sidebar.button("Reset all filters", on_click=reset_filters, use_container_width=True)

filtered_df = df[
    (df["Department"].isin(dept_filter)) &
    (df["JobRole"].isin(role_filter)) &
    (df["Career_Stage"].isin(stage_filter)) &
    (df["Cluster_Name"].isin(cluster_filter)) &
    (df["Promotion_Gap_Risk"].isin(risk_filter)) &
    (df["Promotion_Gap_Ratio"] >= gap_threshold)
]

total_emp = len(filtered_df)
pct_of_total = (total_emp / baseline_total * 100) if baseline_total else 0

st.sidebar.divider()
st.sidebar.markdown(f"**{total_emp:,} of {baseline_total:,}** employees match your filters")
st.sidebar.progress(min(int(pct_of_total), 100))
st.sidebar.caption(f"Dataset processed via K-Means clustering (validated with Hierarchical clustering).")

# =====================================================================
# TOP-LEVEL KPI CARDS
# =====================================================================
attr_rate = (filtered_df["Attrition"].mean() * 100) if total_emp > 0 else 0
high_risk_count = len(filtered_df[filtered_df["Promotion_Gap_Risk"] == "High"])
high_risk_pct = (high_risk_count / total_emp * 100) if total_emp > 0 else 0
actionable_targets = len(filtered_df[filtered_df["Retention_Opportunity"] == "High Priority"])
target_pct = (actionable_targets / total_emp * 100) if total_emp > 0 else 0
training_needs = len(filtered_df[filtered_df["Training_Need_Indicator"] == "Needs Development Plan"])
training_pct = (training_needs / total_emp * 100) if total_emp > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    with st.container(border=True):
        st.metric("Filtered Workforce", f"{total_emp:,}", f"{pct_of_total:.0f}% of total", delta_color="off")
with k2:
    with st.container(border=True):
        st.metric("Attrition Rate", f"{attr_rate:.1f}%", f"{attr_rate - baseline_attr_rate:+.1f} pts vs. all",
                   delta_color="inverse")
with k3:
    with st.container(border=True):
        st.metric("High Promotion Risk", f"{high_risk_count:,}", f"{high_risk_pct - baseline_high_risk_pct:+.1f} pts vs. all",
                   delta_color="inverse")
with k4:
    with st.container(border=True):
        st.metric("Retention Priority", f"{actionable_targets:,}", f"{target_pct - baseline_target_pct:+.1f} pts vs. all",
                   delta_color="off")
with k5:
    with st.container(border=True):
        st.metric("Training Needs", f"{training_needs:,}", f"{training_pct - baseline_training_pct:+.1f} pts vs. all",
                   delta_color="off")

st.write("")

# =====================================================================
# TABS
# =====================================================================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Career Path Clustering",
    "Promotion Gap Monitor",
    "Retention Opportunity",
    "Managerial Insights",
    "Target Export",
])

# ============================ TAB 0: OVERVIEW =======================
with tab0:
    if total_emp == 0:
        st.warning("No employees match the current filters - try widening your selection in the sidebar.")
    else:
        st.subheader("What this dashboard tells you")
        st.markdown(
            "Traditional attrition models flag **who** might leave. This view explains **why** - by "
            "surfacing promotion stagnation, role stagnation and weak managerial continuity *before* "
            "employees disengage, so retention actions can be targeted rather than generic."
        )

        o1, o2 = st.columns(2)
        with o1:
            fig_o1 = px.pie(
                filtered_df, names="Cluster_Name", title="Career Segment Mix", hole=0.5,
                color="Cluster_Name", color_discrete_map=CLUSTER_COLORS
            )
            fig_o1.update_traces(textposition="inside", textinfo="percent+label", showlegend=False)
            st.plotly_chart(style_fig(fig_o1, height=380), use_container_width=True)
        with o2:
            fig_o2 = px.pie(
                filtered_df, names="Promotion_Gap_Risk", title="Promotion Gap Risk Mix", hole=0.5,
                color="Promotion_Gap_Risk", color_discrete_map=RISK_COLORS,
                category_orders={"Promotion_Gap_Risk": ["Low", "Medium", "High"]}
            )
            fig_o2.update_traces(textposition="inside", textinfo="percent+label", showlegend=False)
            st.plotly_chart(style_fig(fig_o2, height=380), use_container_width=True)

        dept_attr = filtered_df.groupby("Department")["Attrition"].mean().sort_values(ascending=False) * 100
        fig_o3 = px.bar(
            dept_attr.reset_index(), x="Attrition", y="Department", orientation="h",
            title="Attrition Rate by Department (%)", color="Department",
            color_discrete_sequence=px.colors.qualitative.Set2, text_auto=".1f"
        )
        fig_o3.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_o3, height=320), use_container_width=True)

        with st.expander("Quick auto-generated insights", expanded=True):
            top_gap_role = filtered_df.groupby("JobRole")["Promotion_Gap_Ratio"].mean().idxmax()
            top_gap_val = filtered_df.groupby("JobRole")["Promotion_Gap_Ratio"].mean().max()
            top_dept_attr = dept_attr.index[0] if len(dept_attr) else "N/A"
            low_cont_attr = filtered_df[filtered_df["Manager_Stability_Band"] == "Low Continuity"]["Attrition"].mean() * 100
            high_cont_attr = filtered_df[filtered_df["Manager_Stability_Band"] == "High Continuity"]["Attrition"].mean() * 100
            st.markdown(
                f"- **{top_gap_role}** shows the widest average promotion gap "
                f"(ratio ≈ **{top_gap_val:.2f}**) among filtered roles.\n"
                f"- **{top_dept_attr}** has the highest attrition rate in this selection.\n"
                f"- Employees with **low manager continuity** leave at **{low_cont_attr:.1f}%**, "
                f"vs **{high_cont_attr:.1f}%** for those with **high continuity**.\n"
                f"- **{actionable_targets:,}** employees ({target_pct:.1f}% of the filtered group) are "
                f"flagged as high-priority retention opportunities not yet gone, but showing stagnation."
            )

# ============================ TAB 1: CLUSTERING ======================
with tab1:
    st.subheader("Career Path Clustering Dashboard")
    if total_emp == 0:
        st.warning("No employees match the current filters.")
    else:
        show_pct = st.toggle("Show cluster sizes as %", value=False, key="cluster_pct_toggle")

        c1, c2 = st.columns(2)
        with c1:
            counts = filtered_df["Cluster_Name"].value_counts()
            plot_vals = (counts / counts.sum() * 100).round(1) if show_pct else counts
            fig_cluster = px.bar(
                plot_vals.reset_index(), x="Cluster_Name", y="count",
                color="Cluster_Name", color_discrete_map=CLUSTER_COLORS,
                title="Career Segment Distribution", text_auto=True,
            )
            fig_cluster.update_layout(showlegend=False, xaxis_title="", yaxis_title="% of workforce" if show_pct else "Employees")
            st.plotly_chart(style_fig(fig_cluster, height=400), use_container_width=True)

        with c2:
            fig_3d = px.scatter_3d(
                filtered_df, x="YearsAtCompany", y="Promotion_Gap_Ratio", z="Role_Stagnation_Index",
                color="Promotion_Gap_Risk", color_discrete_map=RISK_COLORS,
                title="3D Stagnation & Experience Mapping", opacity=0.75,
                category_orders={"Promotion_Gap_Risk": ["Low", "Medium", "High"]},
            )
            st.plotly_chart(style_fig(fig_3d, height=400), use_container_width=True)

        st.markdown("#### Cluster Explorer - Career Pattern Summary")
        cluster_summary = filtered_df.groupby("Cluster_Name").agg(
            Employees=("Cluster_Name", "count"),
            Avg_Promotion_Gap_Ratio=("Promotion_Gap_Ratio", "mean"),
            Avg_Role_Stagnation=("Role_Stagnation_Index", "mean"),
            Avg_Training_Intensity=("Training_Intensity_Score", "mean"),
            Avg_Manager_Stability=("Manager_Stability_Indicator", "mean"),
            Attrition_Rate_Pct=("Attrition", lambda x: round(x.mean() * 100, 1)),
        ).round(2).sort_values("Employees", ascending=False)

        st.dataframe(
            cluster_summary,
            use_container_width=True,
            column_config={
                "Employees": st.column_config.NumberColumn("Employees", format="%d"),
                "Avg_Promotion_Gap_Ratio": st.column_config.ProgressColumn("Avg Promotion Gap", min_value=0, max_value=1, format="%.2f"),
                "Avg_Role_Stagnation": st.column_config.ProgressColumn("Avg Role Stagnation", min_value=0, max_value=1, format="%.2f"),
                "Avg_Training_Intensity": st.column_config.NumberColumn("Avg Training Intensity", format="%.2f"),
                "Avg_Manager_Stability": st.column_config.ProgressColumn("Avg Manager Stability", min_value=0, max_value=1, format="%.2f"),
                "Attrition_Rate_Pct": st.column_config.ProgressColumn("Attrition Rate %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

        cluster_pick = st.selectbox("Drill into a specific cluster:", cluster_summary.index)
        row = cluster_summary.loc[cluster_pick]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric("Employees", f"{int(row['Employees']):,}")
        with m2:
            with st.container(border=True):
                st.metric("Avg. Promotion Gap", f"{row['Avg_Promotion_Gap_Ratio']:.2f}")
        with m3:
            with st.container(border=True):
                st.metric("Avg. Role Stagnation", f"{row['Avg_Role_Stagnation']:.2f}")
        with m4:
            with st.container(border=True):
                st.metric("Attrition Rate", f"{row['Attrition_Rate_Pct']:.1f}%")

# ============================ TAB 2: PROMOTION GAP ===================
with tab2:
    st.subheader("Promotion Gap Monitor")
    st.caption("Identify employees with high promotion gaps and see where stagnation concentrates.")
    if total_emp == 0:
        st.warning("No employees match the current filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            risk_counts = filtered_df["Promotion_Gap_Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig_risk = px.bar(
                risk_counts.reset_index(), x="Promotion_Gap_Risk", y="count",
                color="Promotion_Gap_Risk", color_discrete_map=RISK_COLORS,
                title="Promotion Gap Risk Distribution", text_auto=True,
            )
            fig_risk.update_layout(showlegend=False, xaxis_title="", yaxis_title="Employees")
            st.plotly_chart(style_fig(fig_risk, height=380), use_container_width=True)

        with c2:
            role_gap = filtered_df.groupby("JobRole")["Promotion_Gap_Ratio"].mean().sort_values(ascending=False).reset_index()
            fig_role_gap = px.bar(
                role_gap, x="Promotion_Gap_Ratio", y="JobRole", orientation="h",
                title="Avg. Promotion Gap Ratio by Job Role", color="Promotion_Gap_Ratio",
                color_continuous_scale="Reds",
            )
            fig_role_gap.update_layout(yaxis_title="", coloraxis_showscale=False,
                                        yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_fig(fig_role_gap, height=380), use_container_width=True)

        st.markdown("#### Department → Role → Risk Hierarchy")
        sunburst_metric = st.radio(
            "Color the hierarchy by:", ["Attrition", "Promotion_Gap_Ratio"],
            horizontal=True, key="sunburst_metric"
        )
        fig_sunburst = px.sunburst(
            filtered_df, path=["Department", "JobRole", "Promotion_Gap_Risk"],
            color=sunburst_metric, color_continuous_scale="Reds",
            title="Workforce Hierarchy Breakdown"
        )
        fig_sunburst.update_layout(height=560)
        st.plotly_chart(style_fig(fig_sunburst, height=560), use_container_width=True)

        st.markdown("#### Top 5 Highest-Risk Roles")
        top_roles = role_gap.head(5).rename(columns={"Promotion_Gap_Ratio": "Avg_Promotion_Gap_Ratio"})
        st.dataframe(
            top_roles, hide_index=True, use_container_width=True,
            column_config={
                "Avg_Promotion_Gap_Ratio": st.column_config.ProgressColumn(
                    "Avg Promotion Gap Ratio", min_value=0, max_value=1, format="%.2f"
                )
            },
        )

# ============================ TAB 3: RETENTION OPPORTUNITY ==========
with tab3:
    st.subheader("Retention Opportunity Panel")
    st.caption("Employees who are not yet flagged as attrition risks but show meaningful career "
               "stagnation the core intervention target group.")
    if total_emp == 0:
        st.warning("No employees match the current filters.")
    else:
        target_df = filtered_df[filtered_df["Retention_Opportunity"] == "High Priority"]

        with st.container(border=True):
            st.metric("High-Priority Retention Targets", f"{len(target_df):,}",
                      f"{target_pct:.1f}% of filtered workforce")

        if len(target_df) > 0:
            action_counts = target_df["Suggested_Action"].value_counts().reset_index().head(10)
            fig_actions = px.bar(
                action_counts, x="count", y="Suggested_Action", orientation="h",
                title="Most Common Suggested Actions", color="count", color_continuous_scale="Blues",
            )
            fig_actions.update_layout(yaxis_title="", coloraxis_showscale=False,
                                       yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_fig(fig_actions, height=380), use_container_width=True)
        else:
            st.info("No high-priority retention targets in the current filter selection.")

        st.markdown("#### What-If Promotion Intervention Simulator")
        with st.container(border=True):
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                promotion_budget = st.slider("% of High-Risk Employees to Promote:", 0, 100, 25, step=5)
                simulated_mitigation = int(high_risk_count * (promotion_budget / 100))
                remaining_high_risk = high_risk_count - simulated_mitigation
                st.success(
                    f"Promoting **{simulated_mitigation}** high-risk employees would reduce total "
                    f"high-risk exposure to **{remaining_high_risk}**."
                )
                st.progress(0 if high_risk_count == 0 else int(remaining_high_risk / max(high_risk_count, 1) * 100))
                st.caption("Remaining high-risk exposure after simulated intervention")
            with sim_col2:
                sim_data = pd.DataFrame({
                    "Category": ["Current High Risk", "Simulated Mitigated", "Remaining Risk"],
                    "Count": [high_risk_count, simulated_mitigation, remaining_high_risk]
                })
                fig_sim = px.bar(
                    sim_data, x="Category", y="Count", color="Category", text_auto=True,
                    title="Risk Exposure Mitigation Profile",
                    color_discrete_sequence=["#C0392B", "#2E86AB", "#F2A541"],
                )
                fig_sim.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(style_fig(fig_sim, height=320), use_container_width=True)

        with st.expander("How Retention Opportunity is scored"):
            st.markdown(
                "An employee is a **High Priority** retention opportunity when they have **not** "
                "historically left (`Attrition = 0`) but sit in the **High** promotion-gap risk band - "
                "a combination of their raw `Promotion_Gap_Ratio` and their career-cluster membership. "
                "Suggested actions then draw on training intensity, role stagnation, promotion-gap size, "
                "and manager continuity to recommend a specific next step."
            )

# ============================ TAB 4: MANAGERIAL INSIGHTS =============
with tab4:
    st.subheader("Managerial Insight Dashboard")
    st.caption("Manager tenure vs. career growth, and team-level stagnation signals.")
    if total_emp == 0:
        st.warning("No employees match the current filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_manager = px.box(
                filtered_df, x="Manager_Stability_Band", y="Role_Stagnation_Index",
                color="Manager_Stability_Band", color_discrete_map=CONTINUITY_COLORS,
                title="Manager Continuity vs. Role Stagnation"
            )
            fig_manager.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(style_fig(fig_manager, height=380), use_container_width=True)

        with c2:
            manager_attrition = filtered_df.groupby("Manager_Stability_Band")["Attrition"].mean().reset_index()
            manager_attrition["Attrition"] *= 100
            fig_mgr_attr = px.bar(
                manager_attrition, x="Manager_Stability_Band", y="Attrition",
                color="Manager_Stability_Band", color_discrete_map=CONTINUITY_COLORS,
                title="Attrition Rate (%) by Manager Continuity", text_auto=".1f"
            )
            fig_mgr_attr.update_layout(showlegend=False, xaxis_title="", yaxis_title="Attrition Rate (%)")
            st.plotly_chart(style_fig(fig_mgr_attr, height=380), use_container_width=True)

        st.markdown("#### Team-Level Stagnation Signals (by Department)")
        team_signals = filtered_df.groupby("Department").agg(
            Avg_Role_Stagnation=("Role_Stagnation_Index", "mean"),
            Avg_Promotion_Gap=("Promotion_Gap_Ratio", "mean"),
            Low_Manager_Continuity_Pct=("Manager_Stability_Band", lambda x: round((x == "Low Continuity").mean() * 100, 1)),
            Attrition_Rate_Pct=("Attrition", lambda x: round(x.mean() * 100, 1)),
        ).round(2)

        st.dataframe(
            team_signals, use_container_width=True,
            column_config={
                "Avg_Role_Stagnation": st.column_config.ProgressColumn("Avg Role Stagnation", min_value=0, max_value=1, format="%.2f"),
                "Avg_Promotion_Gap": st.column_config.ProgressColumn("Avg Promotion Gap", min_value=0, max_value=1, format="%.2f"),
                "Low_Manager_Continuity_Pct": st.column_config.ProgressColumn("Low Continuity %", min_value=0, max_value=100, format="%.1f%%"),
                "Attrition_Rate_Pct": st.column_config.ProgressColumn("Attrition Rate %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

# ============================ TAB 5: TARGET EXPORT ====================
with tab5:
    st.subheader("Actionable Target Roster & Export")

    target_df = filtered_df[filtered_df["Retention_Opportunity"] == "High Priority"]

    if len(target_df) == 0:
        st.info("No high-priority retention targets to export for the current filter selection.")
    else:
        sort_col, sort_dir = st.columns([2, 1])
        with sort_col:
            sort_by = st.selectbox(
                "Sort roster by:",
                ["Promotion_Gap_Ratio", "MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion"],
                index=0,
            )
        with sort_dir:
            ascending = st.toggle("Ascending order", value=False)

        r1, r2, r3 = st.columns(3)
        with r1:
            with st.container(border=True):
                st.metric("Roster Size", f"{len(target_df):,}")
        with r2:
            with st.container(border=True):
                st.metric("Avg. Monthly Income", f"${target_df['MonthlyIncome'].mean():,.0f}")
        with r3:
            with st.container(border=True):
                st.metric("Avg. Years Since Promotion", f"{target_df['YearsSinceLastPromotion'].mean():.1f}")

        cols_to_show = [
            "Age", "Department", "JobRole", "MonthlyIncome",
            "YearsAtCompany", "YearsSinceLastPromotion", "Promotion_Gap_Ratio",
            "Cluster_Name", "Training_Need_Indicator", "Manager_Stability_Band", "Suggested_Action"
        ]

        st.dataframe(
            target_df[cols_to_show].sort_values(by=sort_by, ascending=ascending),
            use_container_width=True,
            hide_index=True,
            column_config={
                "MonthlyIncome": st.column_config.NumberColumn("Monthly Income", format="$%d"),
                "Promotion_Gap_Ratio": st.column_config.ProgressColumn("Promotion Gap Ratio", min_value=0, max_value=1, format="%.2f"),
                "Suggested_Action": st.column_config.TextColumn("Suggested Action", width="large"),
            },
        )

        csv_data = target_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export High-Priority Retention Roster (CSV)",
            data=csv_data,
            file_name="Palo_Alto_High_Priority_Retention_Targets.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Export includes every column of the fully processed dataset for the filtered, high-priority group.")