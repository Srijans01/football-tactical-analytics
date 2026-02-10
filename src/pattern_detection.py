from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_possessions(features_df, k=4):
    X = features_df[['num_passes', 'duration', 'avg_x', 'avg_y']]
    X = StandardScaler().fit_transform(X)
    
    model = KMeans(n_clusters=k, random_state=42)
    features_df['pattern_cluster'] = model.fit_predict(X)
    
    return features_df
