import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.write("Segment customers using demographic and behavioral characteristics.")

@st.cache_data
def load_data():
    return pd.read_csv("customers.csv")

df = load_data()

features = ["age","income","purchase_frequency","avg_order_value","recency_days","engagement_score"]

st.sidebar.header("Segmentation Settings")
n_clusters = st.sidebar.slider("Number of customer segments", 2, 8, 4)

X_scaled = StandardScaler().fit_transform(df[features])
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["segment"] = model.fit_predict(X_scaled)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(df))
c2.metric("Segments", n_clusters)
c3.metric("Average Income", f"₹{df['income'].mean():,.0f}")
c4.metric("Average Order Value", f"₹{df['avg_order_value'].mean():,.0f}")

tab1, tab2, tab3 = st.tabs(["Customer Data", "Segment Analysis", "Visualization"])

with tab1:
    st.subheader("Customer Dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Segmented CSV", df.to_csv(index=False).encode("utf-8"),
                       "customers_segmented.csv", "text/csv")

with tab2:
    st.subheader("Segment Profiles")
    profile = df.groupby("segment")[features].mean().round(2)
    st.dataframe(profile, use_container_width=True)
    st.subheader("Customers per Segment")
    st.bar_chart(df["segment"].value_counts().sort_index())

with tab3:
    st.subheader("Customer Segments")
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_pca[:,0], X_pca[:,1], c=df["segment"], s=90, alpha=0.8)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("K-Means Customer Segmentation")
    st.pyplot(fig)

st.caption("Customer segmentation using K-Means clustering, StandardScaler and PCA.")
