import pandas as pd

def build_possessions(events_df):
    events_df = events_df.sort_values(['match_id', 'timestamp'])
    events_df['possession_id'] = (
        (events_df['team'] != events_df['team'].shift()).cumsum()
    )
    return events_df
