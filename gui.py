import matplotlib
matplotlib.use("Agg")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Titanic Passenger Analysis",
    page_icon="🚢",
    layout="wide"
)

# =====================================
# Load Dataset
# =====================================

df = sns.load_dataset("titanic")

# Create Family Size Column
df["Family_Size"] = df["sibsp"] + df["parch"]

# =====================================
# Sidebar
# =====================================

st.sidebar.title("🚢 Titanic Dashboard")

page = st.sidebar.radio(

    "Choose Section",

    [

        "🏠 Home",

        "📋 Dataset",

        "📊 Statistics",

        "🔍 Filtering",

        "📈 Charts",

        "ℹ️ About"

    ]

)

# =====================================
# HOME
# =====================================

if page == "🏠 Home":

    st.title("🚢 Titanic Passenger Analysis Dashboard")

    st.markdown("---")

    st.write("""
Welcome to the **Titanic Passenger Analysis Dashboard**.

This dashboard allows you to explore passenger information,
analyze survival statistics,
compare passenger classes,
filter passengers,
and visualize Titanic insights.
""")

    st.image(
        "https://images.unsplash.com/photo-1518546305927-5a555bb7020d?w=1200",
    )

    st.markdown("---")

    st.subheader("📊 Quick Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(

            "Passengers",

            len(df)

        )

    with col2:

        st.metric(

            "Survived",

            int(df["survived"].sum())

        )

    with col3:

        st.metric(

            "Average Age",

            f"{df['age'].mean():.1f}"

        )

    with col4:

        st.metric(

            "Highest Fare",

            f"${df['fare'].max():.2f}"

        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(

        df.head(),

        use_container_width=True

    )

# =====================================
# DATASET
# =====================================

elif page == "📋 Dataset":

    st.title("📋 Titanic Dataset")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Show First 10 Rows"):

            st.dataframe(

                df.head(10),

                use_container_width=True

            )

    with col2:

        if st.button("Show Last 10 Rows"):

            st.dataframe(

                df.tail(10),

                use_container_width=True

            )

    st.markdown("---")

    if st.button("Show Entire Dataset"):

        st.dataframe(

            df,

            use_container_width=True

        )

    st.markdown("---")

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Rows",

            df.shape[0]

        )

    with col2:

        st.metric(

            "Columns",

            df.shape[1]

        )

    st.markdown("---")

    if st.checkbox("Show Columns"):

        st.write(df.columns.tolist())

    if st.checkbox("Show Data Types"):

        st.dataframe(

            df.dtypes.astype(str),

            use_container_width=True

        )

    if st.checkbox("Show Statistical Summary"):

        st.dataframe(

            df.describe(),

            use_container_width=True

        )

# =====================================
# STATISTICS
# =====================================

elif page == "📊 Statistics":

    st.title("📊 Titanic Statistics")

    st.subheader("Fare Statistics")

    fare = df["fare"].to_numpy()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Maximum Fare",
            f"${np.max(fare):.2f}"
        )

        st.metric(
            "Mean Fare",
            f"${np.mean(fare):.2f}"
        )

    with col2:

        st.metric(
            "Minimum Fare",
            f"${np.min(fare):.2f}"
        )

        st.metric(
            "Total Fare",
            f"${np.sum(fare):,.2f}"
        )

    st.markdown("---")

    st.subheader("GroupBy Analysis")

    option = st.selectbox(

        "Choose Analysis",

        [

            "Average Age by Passenger Class",

            "Average Fare by Passenger Class",

            "Maximum Fare by Passenger Class",

            "Minimum Age by Passenger Class",

            "Passenger Count by Class"

        ]

    )

    if option == "Average Age by Passenger Class":

        result = df.groupby(
            "pclass"
        )["age"].mean()

        st.dataframe(result)

    elif option == "Average Fare by Passenger Class":

        result = df.groupby(
            "pclass"
        )["fare"].mean()

        st.dataframe(result)

    elif option == "Maximum Fare by Passenger Class":

        result = df.groupby(
            "pclass"
        )["fare"].max()

        st.dataframe(result)

    elif option == "Minimum Age by Passenger Class":

        result = df.groupby(
            "pclass"
        )["age"].min()

        st.dataframe(result)

    elif option == "Passenger Count by Class":

        result = df.groupby(
            "pclass"
        )["survived"].count()

        st.dataframe(result)

# =====================================
# FILTERING
# =====================================

elif page == "🔍 Filtering":

    st.title("🔍 Filter Passengers")

    passenger_class = st.selectbox(

        "Passenger Class",

        ["All"] + sorted(df["pclass"].dropna().unique())

    )

    gender = st.selectbox(

        "Gender",

        ["All"] + sorted(df["sex"].dropna().unique())

    )

    survived = st.selectbox(

        "Survived",

        ["All", "Yes", "No"]

    )

    age = st.slider(

        "Minimum Age",

        0,

        int(df["age"].max()),

        0

    )

    fare = st.slider(

        "Minimum Fare",

        float(df["fare"].min()),

        float(df["fare"].max()),

        float(df["fare"].min())

    )

    filtered = df.copy()

    if passenger_class != "All":

        filtered = filtered[
            filtered["pclass"] == passenger_class
        ]

    if gender != "All":

        filtered = filtered[
            filtered["sex"] == gender
        ]

    if survived == "Yes":

        filtered = filtered[
            filtered["survived"] == 1
        ]

    elif survived == "No":

        filtered = filtered[
            filtered["survived"] == 0
        ]

    filtered = filtered[
        filtered["age"].fillna(0) >= age
    ]

    filtered = filtered[
        filtered["fare"] >= fare
    ]

    st.success(

        f"Number of Passengers: {len(filtered)}"

    )

    st.dataframe(

        filtered,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Quick Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Average Fare",

            f"${filtered['fare'].mean():.2f}"

            if len(filtered) > 0 else "$0"

        )

    with col2:

        st.metric(

            "Average Age",

            f"{filtered['age'].mean():.1f}"

            if len(filtered) > 0 else "0"

        )

    with col3:

        st.metric(

            "Survival Rate",

            f"{filtered['survived'].mean()*100:.1f}%"

            if len(filtered) > 0 else "0%"

        )

# =====================================
# CHARTS
# =====================================

elif page == "📈 Charts":

    st.title("📈 Titanic Charts")

    chart = st.selectbox(

        "Choose Chart",

        [

            "Average Fare by Passenger Class",

            "Age Distribution",

            "Age vs Fare",

            "Average Fare (Seaborn)",

            "Age Histogram with KDE"

        ]

    )

    # --------------------------------

    if chart == "Average Fare by Passenger Class":

        fig, ax = plt.subplots(figsize=(9,5))

        avg_fare = df.groupby("pclass")["fare"].mean()

        ax.bar(avg_fare.index.astype(str), avg_fare.values)

        ax.set_title("Average Fare by Passenger Class")

        ax.set_xlabel("Passenger Class")

        ax.set_ylabel("Average Fare")

        st.pyplot(fig)

    # --------------------------------

    elif chart == "Age Distribution":

        fig, ax = plt.subplots(figsize=(9,5))

        ax.hist(

            df["age"].dropna(),

            bins=20,

            edgecolor="black"

        )

        ax.set_title("Age Distribution")

        ax.set_xlabel("Age")

        ax.set_ylabel("Count")

        st.pyplot(fig)

    # --------------------------------

    elif chart == "Age vs Fare":

        fig, ax = plt.subplots(figsize=(9,5))

        sns.scatterplot(

            data=df,

            x="age",

            y="fare",

            ax=ax

        )

        ax.set_title("Age vs Fare")

        st.pyplot(fig)

    # --------------------------------

    elif chart == "Average Fare (Seaborn)":

        fig, ax = plt.subplots(figsize=(9,5))

        sns.barplot(

            data=df,

            x="pclass",

            y="fare",

            ax=ax

        )

        ax.set_title("Average Fare by Passenger Class")

        st.pyplot(fig)

    # --------------------------------

    elif chart == "Age Histogram with KDE":

        fig, ax = plt.subplots(figsize=(9,5))

        sns.histplot(

            data=df,

            x="age",

            kde=True,

            ax=ax

        )

        ax.set_title("Age Distribution with KDE")

        plt.savefig(

            "titanic_histogram.png",

            dpi=300,

            bbox_inches="tight"

        )

        st.pyplot(fig)

        with open(

            "titanic_histogram.png",

            "rb"

        ) as file:

            st.download_button(

                "⬇ Download Histogram",

                file,

                file_name="titanic_histogram.png",

                mime="image/png"

            )

# =====================================
# ABOUT
# =====================================

elif page == "ℹ️ About":

    st.title("🚢 Titanic Passenger Analysis")

    with st.container(border=True):

        st.subheader("About This Project")

        st.markdown("### 🛠 Technologies Used")

        st.markdown("""
✅ Python

✅ NumPy

✅ Pandas

✅ Matplotlib

✅ Seaborn

✅ Streamlit
""")

        st.markdown("---")

        st.markdown("### 📊 Dataset")

        st.write("Titanic Dataset from Seaborn")

        st.markdown("---")

        st.markdown("### ✨ Features")

        st.markdown("""
✔ Dataset Preview

✔ Statistical Analysis

✔ NumPy Analysis

✔ GroupBy Analysis

✔ Passenger Filtering

✔ Interactive Charts

✔ Download Histogram
""")

        st.markdown("---")

        st.markdown(
            """
            <div style="text-align:center;">
                <h4>Developed by Coding Hub</h4>
                <p>© 2026 All Rights Reserved</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success("Thank you for using Titanic Passenger Analysis Dashboard 💙")
  
