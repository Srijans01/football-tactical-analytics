import pandas as pd

def extract_possession_features(events_df):
    features = events_df.groupby('possession_id').agg(
        team=('team', 'first'),
        duration=('timestamp', 'max'),
        num_passes=('type', lambda x: (x == 'Pass').sum()),
        num_shots=('type', lambda x: (x == 'Shot').sum()),
        avg_x=('location_x', 'mean'),
        avg_y=('location_y', 'mean')
    ).reset_index()
    
    return features
