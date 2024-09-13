import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load Sample Customer Data
df = pd.read_csv('/Users/aksharsakhi/Documents/customer_data.csv')

# Step 2: Preprocessing
# Fill missing values (if any) and scale the RFM data
df.fillna(0, inplace=True)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Step 3: K-means Clustering (Customer Segmentation)
# Using the Elbow method to find the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# Plotting the elbow graph to visualize the optimal clusters
plt.plot(range(1, 11), sse)
plt.xlabel('Number of clusters')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method for Optimal K')
plt.show()

# Choose the optimal number of clusters (e.g., 4) based on the Elbow method
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze the resulting customer segments
print(df.groupby('Cluster').mean())

# Step 4: Assign Marketing Campaigns Based on Segments
def marketing_strategy(cluster):
    if cluster == 0:
        return "Loyalty rewards and exclusive discounts"
    elif cluster == 1:
        return "Welcome offers for new customers"
    elif cluster == 2:
        return "Upsell and cross-sell opportunities"
    else:
        return "Surveys and feedback to improve engagement"

df['Marketing_Campaign'] = df['Cluster'].apply(marketing_strategy)

# Step 5: Predictive Analytics for Customer Behavior (Next Purchase)
X = rfm_scaled
y = df['NextPurchase']  # 0 = No, 1 = Yes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Product Recommendations Based on Segments
def recommend_products(cluster):
    if cluster == 0:
        return "Product A, Product B"
    elif cluster == 1:
        return "Product C, Product D"
    elif cluster == 2:
        return "Product E, Product F"
    else:
        return "Product G, Product H"

df['Product_Recommendations'] = df['Cluster'].apply(recommend_products)

# Final Output: Customer ID, Segment, Campaign, Product Recommendations
print(df[['CustomerID', 'Cluster', 'Marketing_Campaign', 'Product_Recommendations']])