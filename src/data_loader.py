import json
import pandas as pd

def load_events(path):
    with open(path, 'r') as f:
        data = json.load(f)
    events = pd.json_normalize(data['events'])
    return events
