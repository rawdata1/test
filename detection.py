import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
def load_logs(filepath):
    print("Loading Zeek logs...")
    data = pd.read_csv(filepath, delimiter='\t')
    
    # Select relevant features
    features = ['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 
                'proto', 'orig_bytes', 'resp_bytes', 'duration', 'orig_pkts', 'resp_pkts']
    data = data[features]

    # Handle missing or invalid data
    data = data.fillna(0)  # Replace NaNs with 0
    data.replace('-', 0, inplace=True)  # Replace '-' with 0
    
    # Ensure numeric conversion for relevant columns
    numeric_columns = ['orig_bytes', 'resp_bytes', 'duration', 
                       'orig_pkts', 'resp_pkts', 'id.resp_p']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)  # Convert to numeric, replace invalid with 0

    # One-hot encode 'proto' (protocol)
    proto_encoded = pd.get_dummies(data['proto'], prefix='proto')

    # Derived features
    data['total_bytes'] = data['orig_bytes'] + data['resp_bytes']
    data['ratio'] = data['orig_bytes'] / (data['resp_bytes'] + 1)  # Avoid division by zero
    data['byte_rate'] = data['total_bytes'] / (data['duration'] + 1)
    data['orig_pkt_size'] = data['orig_bytes'] / (data['orig_pkts'] + 1)
    data['resp_pkt_size'] = data['resp_bytes'] / (data['resp_pkts'] + 1)

    # Concatenate proto one-hot encoding
    data = pd.concat([data, proto_encoded], axis=1)
    data = data.drop(columns=['proto'])  # Drop the original 'proto' column

    print(f"Processed data shape: {data.shape}")
    return data


# K-Means clustering
def kmeans_detection(data, max_clusters=10):
    print("Running K-Means clustering...")
    scaler = MinMaxScaler()
    features = ['orig_bytes', 'resp_bytes', 'total_bytes', 'ratio', 'duration', 
                'orig_pkts', 'resp_pkts', 'id.resp_p'] + [col for col in data.columns if 'proto_' in col]
    scaled_data = scaler.fit_transform(data[features])
    
    # Elbow Method
    inertia = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    # Plot Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), inertia, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.grid(True)
    plt.show()

    # Choose the optimal number of clusters
    optimal_k = 4  # Replace with elbow method observation or automated selection
    print(f"Using optimal K={optimal_k}")
    
    # K-Means with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_data)
    
    return data

# Analyze Isolation Forest anomalies
def analyze_isolation_forest_anomalies(data, iso_forest, scaled_data):
    print("\nAnalyzing Anomalies Identified by Isolation Forest...")
    
    # Get anomaly scores
    anomaly_scores = iso_forest.score_samples(scaled_data)
    data['anomaly_score'] = anomaly_scores  # Higher negative score = stronger anomaly
    
    # Extract examples of anomalies
    anomalies = data[data['anomaly'] == 1]
    print(f"Total anomalies detected: {len(anomalies)}")
    print("Top 5 anomalies:")
    print(anomalies[['orig_bytes', 'resp_bytes', 'total_bytes', 'ratio', 'id.resp_p', 'anomaly_score']].head())
    
    # Visualize anomalies vs. normal data
    plt.figure(figsize=(8, 6))
    plt.hist(data['anomaly_score'], bins=50, alpha=0.7, label='All Data')
    plt.hist(anomalies['anomaly_score'], bins=50, alpha=0.7, label='Anomalies', color='red')
    plt.title("Anomaly Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    return anomalies

# Explain anomalies using SHAP
def explain_isolation_forest(data, iso_forest, scaled_data):
    print("\nExplaining Anomalies with SHAP...")
    
    # Define feature names explicitly to ensure alignment
    feature_names = ['orig_bytes', 'resp_bytes', 'total_bytes', 'ratio', 'duration', 'orig_pkts', 
                     'resp_pkts', 'id.resp_p'] + [col for col in data.columns if 'proto_' in col]
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(scaled_data)
    
    # Visualize global feature importance
    shap.summary_plot(shap_values, scaled_data, feature_names=feature_names)
    
    # Ensure indices are aligned
    scaled_data = pd.DataFrame(scaled_data, columns=feature_names).reset_index(drop=True)
    data = data.reset_index(drop=True)
    anomaly_index = data[data['anomaly'] == 1].index[0]
    
    # Analyze a single anomaly
    shap.force_plot(
        explainer.expected_value, 
        shap_values[anomaly_index], 
        scaled_data.iloc[anomaly_index], 
        feature_names=feature_names
    )

# Isolation Forest detection
def isolation_forest_detection(data):
    print("Running Isolation Forest...")
    scaler = MinMaxScaler()
    features = ['orig_bytes', 'resp_bytes', 'total_bytes', 'ratio', 'duration', 
                'orig_pkts', 'resp_pkts', 'id.resp_p'] + [col for col in data.columns if 'proto_' in col]
    scaled_data = scaler.fit_transform(data[features])
    
    # Split into training and validation sets
    train_data, val_data = train_test_split(scaled_data, test_size=0.2, random_state=42)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(train_data)
    
    # Predict anomalies
    data['anomaly'] = iso_forest.predict(scaled_data)
    data['anomaly'] = data['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    # Analyze anomalies
    analyze_isolation_forest_anomalies(data, iso_forest, scaled_data)
    explain_isolation_forest(data, iso_forest, scaled_data)
    
    return data

# LSTM for temporal anomaly detection
def lstm_detection(data):
    print("Running LSTM temporal analysis...")
    
    # Ensure 'ts' is numeric and sort by timestamp
    data['ts'] = pd.to_numeric(data['ts'], errors='coerce')
    data = data.dropna(subset=['ts']).sort_values(by='ts')
    
    # Scale data
    scaler = MinMaxScaler()
    features = ['orig_bytes', 'resp_bytes', 'total_bytes', 'ratio', 'duration', 
                'orig_pkts', 'resp_pkts', 'id.resp_p'] + [col for col in data.columns if 'proto_' in col]
    scaled_data = scaler.fit_transform(data[features])
    
    # Create sequences
    seq_length = 10
    sequences, labels = [], []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i+seq_length])
        labels.append(data['anomaly'].iloc[i+seq_length])

    X = np.array(sequences)
    y = np.array(labels)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, X.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model with validation data
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Predict anomalies
    predictions = model.predict(X)
    data['lstm_anomaly'] = 0
    data.loc[seq_length:, 'lstm_anomaly'] = (predictions > 0.5).astype(int).flatten()
    
    return data

# Main function
def main():
    filepath = "conn.log"
    data = load_logs(filepath)

    # Step 1: K-Means
    data = kmeans_detection(data)

    # Step 2: Isolation Forest
    data = isolation_forest_detection(data)

    # Step 3: LSTM
    data = lstm_detection(data)

    # Save results
    data.to_csv("detection_results.csv", index=False)
    print("Results saved to detection_results.csv")

if __name__ == "__main__":
    main()