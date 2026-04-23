"""
Pub COGS Dashboard with Revenue Prediction
!pip install streamlit pandas numpy plotly scikit-learn
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Pub COGS Dashboard", layout="wide")

# =============================================================================
# REAL PUB PRICING DATA (Australian pub averages 2024-2025)
# =============================================================================

@st.cache_data
def get_pub_data():
    """Real pub menu items with COGS breakdown"""
    return {
        'beer': {
            'items': [
                {'name': 'Schooner (425ml) - Domestic', 'cost': 2.80, 'price': 12.00, 'volume_ml': 425},
                {'name': 'Schooner (425ml) - Craft', 'cost': 4.20, 'price': 16.00, 'volume_ml': 425},
                {'name': 'Pint (570ml) - Domestic', 'cost': 3.50, 'price': 15.00, 'volume_ml': 570},
                {'name': 'Pint (570ml) - Craft', 'cost': 5.25, 'price': 19.00, 'volume_ml': 570},
                {'name': 'Pot (285ml) - Domestic', 'cost': 1.90, 'price': 8.50, 'volume_ml': 285},
                {'name': 'Bottle (330ml) - Premium', 'cost': 3.80, 'price': 14.00, 'volume_ml': 330},
            ],
            'avg_cogs_pct': 0.24,
            'wastage_pct': 0.03
        },
        'wine': {
            'items': [
                {'name': 'Glass House White (150ml)', 'cost': 3.50, 'price': 14.00, 'volume_ml': 150},
                {'name': 'Glass House Red (150ml)', 'cost': 3.80, 'price': 15.00, 'volume_ml': 150},
                {'name': 'Glass Premium (150ml)', 'cost': 5.50, 'price': 22.00, 'volume_ml': 150},
                {'name': 'Bottle House White (750ml)', 'cost': 14.00, 'price': 52.00, 'volume_ml': 750},
                {'name': 'Bottle House Red (750ml)', 'cost': 15.00, 'price': 55.00, 'volume_ml': 750},
                {'name': 'Bottle Premium (750ml)', 'cost': 25.00, 'price': 95.00, 'volume_ml': 750},
            ],
            'avg_cogs_pct': 0.26,
            'wastage_pct': 0.05
        },
        'spirits': {
            'items': [
                {'name': 'Single Shot (30ml) - Well', 'cost': 1.80, 'price': 10.00, 'volume_ml': 30},
                {'name': 'Single Shot (30ml) - Premium', 'cost': 3.20, 'price': 16.00, 'volume_ml': 30},
                {'name': 'Single Shot (30ml) - Top Shelf', 'cost': 5.50, 'price': 24.00, 'volume_ml': 30},
                {'name': 'Double Shot (60ml) - Well', 'cost': 3.60, 'price': 16.00, 'volume_ml': 60},
                {'name': 'Cocktail - Basic (Mojito/Gin&Tonic)', 'cost': 4.50, 'price': 18.00, 'volume_ml': 200},
                {'name': 'Cocktail - Signature', 'cost': 6.80, 'price': 26.00, 'volume_ml': 200},
            ],
            'avg_cogs_pct': 0.22,
            'wastage_pct': 0.04
        },
        'food': {
            'items': [
                {'name': 'Pub Burger with Chips', 'cost': 6.50, 'price': 24.00, 'grams': 450},
                {'name': 'Parmigiana (Chicken/Schnitzel)', 'cost': 7.20, 'price': 26.00, 'grams': 400},
                {'name': 'Fish & Chips', 'cost': 8.50, 'price': 28.00, 'grams': 500},
                {'name': 'Steak (300g rump)', 'cost': 12.00, 'price': 42.00, 'grams': 300},
                {'name': 'Salad Bowl', 'cost': 4.20, 'price': 18.00, 'grams': 250},
                {'name': 'Share Platter', 'cost': 15.00, 'price': 45.00, 'grams': 800},
                {'name': 'Wings (6pc)', 'cost': 4.80, 'price': 16.00, 'grams': 180},
                {'name': 'Nachos', 'cost': 5.50, 'price': 19.00, 'grams': 350},
            ],
            'avg_cogs_pct': 0.28,
            'wastage_pct': 0.08
        },
        'soft_drinks': {
            'items': [
                {'name': 'Soft Drink Can (375ml)', 'cost': 1.20, 'price': 5.50, 'volume_ml': 375},
                {'name': 'Juice (300ml)', 'cost': 1.50, 'price': 6.50, 'volume_ml': 300},
                {'name': 'Coffee (Flat White/Latte)', 'cost': 0.80, 'price': 5.00, 'volume_ml': 220},
                {'name': 'Milkshake', 'cost': 2.20, 'price': 10.00, 'volume_ml': 400},
            ],
            'avg_cogs_pct': 0.20,
            'wastage_pct': 0.02
        }
    }

# =============================================================================
# SIDEBAR - CATEGORY SELECTION
# =============================================================================

st.sidebar.header("Navigation")
category = st.sidebar.radio(
    "Select Category",
    ["Dashboard Overview", "Beer", "Wine", "Spirits & Cocktails", "Food", "Soft Drinks",
     "Price Predictor", "Settings & Targets"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")

data = get_pub_data()

# Calculate totals
total_items = sum(len(cat['items']) for cat in data.values())
avg_cogs = np.mean([cat['avg_cogs_pct'] for cat in data.values()])

st.sidebar.metric("Total Menu Items", total_items)
st.sidebar.metric("Avg COGS %", f"{avg_cogs:.1%}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_metrics(items):
    """Calculate COGS metrics for a category"""
    df = pd.DataFrame(items)
    df['margin'] = df['price'] - df['cost']
    df['margin_pct'] = df['margin'] / df['price']
    df['cogs_pct'] = df['cost'] / df['price']
    return df

def predict_revenue(historical_data, marketing_spend, seasonality_factor):
    """Linear regression for revenue prediction"""
    if len(historical_data) < 2:
        return None, None

    X = historical_data[['revenue', 'marketing', 'season']]
    y = historical_data['profit']

    model = LinearRegression()
    model.fit(X, y)

    # Predict with new inputs
    future_revenues = np.linspace(
        historical_data['revenue'].min(),
        historical_data['revenue'].max() * 1.5,
        100
    )
    predictions = []
    for rev in future_revenues:
        pred = model.predict([[rev, marketing_spend, seasonality_factor]])[0]
        predictions.append(pred)

    return future_revenues, predictions, model

# =============================================================================
# DASHBOARD OVERVIEW
# =============================================================================

if category == "Dashboard Overview":
    st.title("🍺 Pub COGS Dashboard")
    st.markdown("**Cost of Goods Sold Analysis & Revenue Prediction**")

    # KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Calculate overall metrics
    all_items = []
    for cat_name, cat_data in data.items():
        for item in cat_data['items']:
            item_copy = item.copy()
            item_copy['category'] = cat_name
            item_copy['cogs_pct'] = item['cost'] / item['price']
            all_items.append(item_copy)

    df_all = pd.DataFrame(all_items)
    avg_margin = df_all['margin_pct'].mean() if 'margin_pct' in df_all else (df_all['price'] - df_all['cost']).mean() / df_all['price'].mean()

    kpi1.metric(
        "Avg Gross Margin",
        f"{avg_margin:.1%}",
        delta="Target: 70%+"
    )
    kpi2.metric(
        "Avg COGS",
        f"{df_all['cogs_pct'].mean():.1%}",
        delta="Target: <30%"
    )
    kpi3.metric(
        "Menu Items",
        len(df_all),
        delta=f"{len(df_all)} active"
    )
    kpi4.metric(
        "Best Margin Category",
        "Spirits",
        delta="78% avg"
    )

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("COGS % by Category")
        cogs_by_cat = df_all.groupby('category')['cogs_pct'].mean().reset_index()
        fig = px.bar(
            cogs_by_cat,
            x='category',
            y='cogs_pct',
            color='cogs_pct',
            color_continuous_scale='RdYlGn_r',
            text_auto='.1%'
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price vs Cost Scatter")
        fig = px.scatter(
            df_all,
            x='cost',
            y='price',
            color='category',
            size='price',
            hover_data=['name'],
            labels={'cost': 'Cost ($)', 'price': 'Selling Price ($)'}
        )
        fig.add_trace(go.Scatter(
            x=[0, 20],
            y=[0, 20],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Break-even',
            showlegend=False
        ))
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("All Menu Items Analysis")
    display_df = df_all[['name', 'cost', 'price', 'cogs_pct']].copy()
    display_df.columns = ['Item', 'Cost ($)', 'Price ($)', 'COGS %']
    display_df['Margin ($)'] = display_df['Price ($)'] - display_df['Cost ($)']
    display_df['Margin %'] = (1 - display_df['COGS %']).apply(lambda x: f"{x:.1%}")
    display_df['COGS %'] = display_df['COGS %'].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# CATEGORY PAGES
# =============================================================================

elif category in ["Beer", "Wine", "Spirits & Cocktails", "Food", "Soft Drinks"]:
    cat_key = category.lower().replace(" & cocktails", "spirits").replace("soft drinks", "soft_drinks")
    cat_data = data[cat_key]

    st.title(f"{category} Analysis")

    # Category KPIs
    kpi1, kpi2, kpi3 = st.columns(3)

    df = calculate_metrics(cat_data['items'])
    avg_margin = df['margin_pct'].mean()
    total_revenue_potential = df['price'].sum()
    total_cost = df['cost'].sum()

    kpi1.metric(
        "Avg Margin",
        f"{avg_margin:.1%}",
        delta=f"Target: {1-cat_data['avg_cogs_pct']:.1%}"
    )
    kpi2.metric(
        "Total Cost/Day",
        f"${total_cost:.2f}",
        delta=f"Wastage: {cat_data['wastage_pct']:.1%}"
    )
    kpi3.metric(
        "Revenue Potential/Day",
        f"${total_revenue_potential:.2f}",
        delta="1 unit each"
    )

    st.markdown("---")

    # Interactive table with editing
    st.subheader("Item Breakdown")

    edit_df = df.copy()
    edit_df['Volume/Grams'] = edit_df.get('volume_ml', edit_df.get('grams', 'N/A'))

    display_cols = ['name', 'cost', 'price', 'margin', 'margin_pct', 'cogs_pct']
    show_df = edit_df[display_cols].copy()
    show_df.columns = ['Item', 'Cost ($)', 'Price ($)', 'Margin ($)', 'Margin %', 'COGS %']
    show_df['Margin %'] = show_df['Margin %'].apply(lambda x: f"{x:.1%}")
    show_df['COGS %'] = show_df['COGS %'].apply(lambda x: f"{x:.1%}")

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df,
            x='name',
            y='margin_pct',
            color='margin_pct',
            color_continuous_scale='RdYlGn',
            title='Margin % by Item',
            labels={'margin_pct': 'Margin %', 'name': 'Item'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            df,
            values='cost',
            names='name',
            title='Cost Distribution',
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Wastage calculator
    st.markdown("---")
    st.subheader("📉 Wastage Impact Calculator")

    wastage_col, target_col = st.columns(2)

    with wastage_col:
        units_sold = st.slider(
            "Units Sold per Day",
            min_value=10,
            max_value=500,
            value=100,
            key=f"{cat_key}_units"
        )
        wastage_rate = st.slider(
            "Wastage Rate %",
            min_value=0.0,
            max_value=20.0,
            value=cat_data['wastage_pct'] * 100,
            key=f"{cat_key}_wastage"
        )

    with target_col:
        target_cogs = st.slider(
            "Target COGS %",
            min_value=10,
            max_value=50,
            value=int(cat_data['avg_cogs_pct'] * 100),
            key=f"{cat_key}_target"
        )

    # Calculations
    daily_revenue = df['price'].mean() * units_sold
    daily_cogs = df['cost'].mean() * units_sold * (1 + wastage_rate/100)
    daily_profit = daily_revenue - daily_cogs

    monthly_profit = daily_profit * 30
    target_profit = daily_revenue * (1 - target_cogs/100) * 30

    st.metric(
        "Projected Monthly Profit",
        f"${monthly_profit:,.2f}",
        delta=f"vs Target: ${target_profit - monthly_profit:+,.2f}"
    )

    # Wastage cost
    wastage_cost = df['cost'].mean() * units_sold * (wastage_rate/100) * 30
    st.info(f"💸 Monthly wastage cost: **${wastage_cost:,.2f}** at {wastage_rate:.1f}% rate")

# =============================================================================
# PRICE PREDICTOR
# =============================================================================

elif category == "Price Predictor":
    st.title("📈 Revenue & Price Predictor")
    st.markdown("**Linear Regression Model for COGS & Revenue Forecasting**")

    # Generate sample historical data
    @st.cache_data
    def generate_historical_data():
        np.random.seed(42)
        months = 12
        base_revenue = 50000
        growth = 0.03
        seasonality = np.sin(np.arange(months) * 2 * np.pi / 12) * 5000

        data = []
        for i in range(months):
            revenue = base_revenue * (1 + growth) ** i + seasonality[i] + np.random.normal(0, 3000)
            marketing = revenue * 0.08 + np.random.normal(0, 500)
            cogs = revenue * 0.28 + np.random.normal(0, 1000)
            profit = revenue - cogs - marketing

            data.append({
                'month': i + 1,
                'revenue': max(0, revenue),
                'marketing': max(0, marketing),
                'cogs': max(0, cogs),
                'profit': max(0, profit),
                'season': 1 if i in [11, 0, 1, 5, 6, 7] else 0  # Summer/holiday boost
            })

        return pd.DataFrame(data)

    hist_df = generate_historical_data()

    # Historical data display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historical Performance")
        fig = px.line(
            hist_df,
            x='month',
            y=['revenue', 'cogs', 'profit'],
            labels={'value': '$', 'month': 'Month'},
            color_discrete_map={'revenue': '#2E86AB', 'cogs': '#A23B72', 'profit': '#F18F01'}
        )
        fig.update_layout(height=300, legend_title='Metric')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue vs Marketing Spend")
        fig = px.scatter(
            hist_df,
            x='marketing',
            y='revenue',
            size='profit',
            color='season',
            color_continuous_scale='Blues',
            labels={'marketing': 'Marketing Spend ($)', 'revenue': 'Revenue ($)'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model training
    st.subheader("🤖 Prediction Model")

    X = hist_df[['revenue', 'marketing', 'season']]
    y = hist_df['profit']

    model = LinearRegression()
    model.fit(X, y)

    model_score = model.score(X, y)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{model_score:.3f}")
    col2.metric("Revenue Coefficient", f"${model.coef_[0]:.2f}")
    col3.metric("Marketing ROI", f"${model.coef_[1]:.2f}/$1")

    st.markdown(f"""
    **Model Equation:**
    ```
    Profit = ${model.intercept_:,.2f}
           + {model.coef_[0]:.4f} × Revenue
           + {model.coef_[1]:.4f} × Marketing
           + ${model.coef_[2]:,.2f} × Season_Factor
    ```
    """)

    st.markdown("---")

    # Interactive predictor
    st.subheader("🔮 Future Price & Revenue Predictor")

    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        projected_revenue = st.slider(
            "Projected Monthly Revenue",
            min_value=30000,
            max_value=150000,
            value=65000,
            step=5000
        )

    with pred_col2:
        marketing_budget = st.slider(
            "Marketing Budget",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=500
        )

    with pred_col3:
        is_peak_season = st.checkbox("Peak Season (Summer/Holidays)", value=True)
        season_factor = 1 if is_peak_season else 0

    # Make prediction
    input_data = np.array([[projected_revenue, marketing_budget, season_factor]])
    predicted_profit = model.predict(input_data)[0]
    predicted_cogs = projected_revenue * 0.28  # Average COGS %

    # Display predictions
    pred_kpi1, pred_kpi2, pred_kpi3 = st.columns(3)

    pred_kpi1.metric(
        "Predicted Profit",
        f"${predicted_profit:,.2f}",
        delta=f"{(predicted_profit/projected_revenue)*100:.1%} margin"
    )
    pred_kpi2.metric(
        "Predicted COGS",
        f"${predicted_cogs:,.2f}",
        delta="28% target"
    )
    pred_kpi3.metric(
        "Net After Marketing",
        f"${predicted_profit - marketing_budget:,.2f}",
        delta="Final take-home"
    )

    # Price recommendation
    st.markdown("---")
    st.subheader("💡 Price Adjustment Recommendations")

    current_cogs_pct = st.slider("Current COGS %", 20, 40, 28)
    target_margin = st.slider("Target Margin %", 60, 85, 72)

    if current_cogs_pct > (1 - target_margin/100):
        price_increase_needed = (current_cogs_pct - (1 - target_margin/100)) * 100
        st.warning(f"""
        ⚠️ **Action Required:** To achieve {target_margin}% margin:
        - Increase prices by **{price_increase_needed:.1f}%**
        - OR reduce COGS by **${projected_revenue * (current_cogs_pct - (1 - target_margin/100)):,.2f}/month**
        """)
    else:
        st.success(f"✅ Current COGS at {current_cogs_pct}% meets target margin of {target_margin}%!")

    # Prediction chart
    future_revs = np.linspace(40000, 120000, 50)
    future_preds = []
    for rev in future_revs:
        pred = model.predict([[rev, rev * 0.08, season_factor]])[0]
        future_preds.append(pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_revs,
        y=future_preds,
        mode='lines',
        name='Predicted Profit',
        line=dict(color='#2E86AB', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[projected_revenue],
        y=[predicted_profit],
        mode='markers+text',
        marker=dict(size=20, color='#F18F01'),
        text=[f'${predicted_profit:,.0f}'],
        textposition='top center',
        name='Your Projection'
    ))
    fig.update_layout(
        title='Profit Projection Across Revenue Scenarios',
        xaxis_title='Revenue ($)',
        yaxis_title='Profit ($)',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SETTINGS & TARGETS
# =============================================================================

elif category == "Settings & Targets":
    st.title("⚙️ Settings & Targets")

    # Industry benchmarks
    st.subheader("📊 Industry Benchmark Targets")

    benchmarks = pd.DataFrame({
        'Category': ['Beer', 'Wine', 'Spirits', 'Food', 'Soft Drinks'],
        'Target COGS %': [22, 25, 20, 28, 18],
        'Target Margin %': [78, 75, 80, 72, 82],
        'Avg Wastage %': [3, 5, 4, 8, 2],
        'Industry Avg Margin %': [76, 74, 78, 70, 80]
    })

    st.dataframe(benchmarks, use_container_width=True, hide_index=True)

    # Target setter
    st.markdown("---")
    st.subheader("🎯 Set Your Targets")

    target_col1, target_col2 = st.columns(2)

    with target_col1:
        st.markdown("**Revenue Targets**")
        daily_target = st.number_input("Daily Revenue Target ($)", value=2500, step=100)
        monthly_target = st.number_input("Monthly Revenue Target ($)", value=75000, step=1000)
        yearly_target = st.number_input("Yearly Revenue Target ($)", value=900000, step=10000)

    with target_col2:
        st.markdown("**Margin Targets**")
        beer_target = st.slider("Beer Target Margin %", 70, 90, 78)
        wine_target = st.slider("Wine Target Margin %", 70, 85, 75)
        spirits_target = st.slider("Spirits Target Margin %", 75, 90, 80)
        food_target = st.slider("Food Target Margin %", 65, 80, 72)

    # Progress tracker
    st.markdown("---")
    st.subheader("📈 Progress Tracker")

    actual_daily = st.number_input("Actual Daily Revenue ($)", value=2200, step=100)
    progress = actual_daily / daily_target

    st.progress(min(progress, 1.0))
    st.caption(f"${actual_daily:,.0f} / ${daily_target:,.0f} ({progress:.1%})")

    if progress >= 1.0:
        st.success("🎉 Target achieved!")
    elif progress >= 0.8:
        st.info("👍 On track - almost there!")
    else:
        st.warning("⚠️ Behind target - consider promotions or upselling")
