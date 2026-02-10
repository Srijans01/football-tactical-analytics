import os
import json
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.possession_builder import build_possessions
from src.feature_engineering import extract_possession_features
from src.pattern_detection import cluster_possessions

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_events_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.json'):
        data = json.load(file)
        # Handle StatsBomb format (list) or simple format ({"events": [...]})
        if isinstance(data, list):
            return parse_statsbomb_events(data)
        else:
            return pd.json_normalize(data['events'])
    elif filename.endswith('.csv'):
        return pd.read_csv(file)
    else:
        raise ValueError('Unsupported file format. Use JSON or CSV.')


def parse_statsbomb_events(data):
    """Parse StatsBomb open-data format into our standard format."""
    events = []
    for e in data:
        event = {
            'match_id': e.get('match_id', 1),
            'timestamp': parse_timestamp(e.get('timestamp', '00:00:00')),
            'team': e.get('team', {}).get('name', 'Unknown'),
            'type': e.get('type', {}).get('name', 'Unknown'),
            'player': e.get('player', {}).get('name', None) if e.get('player') else None,
            'location_x': e.get('location', [0, 0])[0] if e.get('location') else 0,
            'location_y': e.get('location', [0, 0])[1] if e.get('location') else 0,
        }
        events.append(event)
    return pd.DataFrame(events)


def parse_timestamp(ts):
    """Convert HH:MM:SS.mmm to seconds."""
    try:
        parts = ts.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except:
        return 0


def label_clusters(cluster_stats):
    """Generate human-readable labels for clusters based on their characteristics."""
    labels = {}
    for idx, row in cluster_stats.iterrows():
        passes = row['num_passes']
        shots = row['num_shots']
        avg_x = row['avg_x']
        
        if passes >= 5 and shots > 0:
            label = "🎯 Build-up Attack"
            desc = "Patient possession with multiple passes leading to shots."
        elif passes >= 5 and shots == 0:
            label = "🔄 Possession Play"
            desc = "Extended ball retention without creating shots."
        elif passes < 3 and avg_x > 70:
            label = "⚡ Quick Counter"
            desc = "Fast transitions into attacking areas."
        elif passes < 3 and avg_x < 40:
            label = "🛡️ Defensive Recovery"
            desc = "Short possessions in defensive areas."
        elif shots > 0:
            label = "🚀 Direct Attack"
            desc = "Quick attacks leading to shooting opportunities."
        else:
            label = "↔️ Transition Play"
            desc = "Mid-field possessions, transitional phases."
        
        labels[idx] = {'label': label, 'description': desc}
    return labels


def get_team_stats(events):
    """Get per-team statistics."""
    team_stats = events.groupby('team').agg(
        total_events=('type', 'count'),
        passes=('type', lambda x: (x == 'Pass').sum()),
        shots=('type', lambda x: (x == 'Shot').sum()),
        pressures=('type', lambda x: (x == 'Pressure').sum()),
    ).reset_index()
    team_stats['pass_accuracy'] = (team_stats['passes'] / team_stats['total_events'] * 100).round(1)
    return team_stats.to_dict('records')


def get_player_stats(events):
    """Get top players by activity."""
    if 'player' not in events.columns or events['player'].isna().all():
        return []
    
    player_stats = events[events['player'].notna()].groupby(['player', 'team']).agg(
        events=('type', 'count'),
        passes=('type', lambda x: (x == 'Pass').sum()),
        shots=('type', lambda x: (x == 'Shot').sum()),
    ).reset_index().sort_values('events', ascending=False).head(10)
    return player_stats.to_dict('records')


def create_team_heatmaps(events):
    """Create side-by-side heatmaps showing where each team played."""
    teams = events['team'].unique()[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, team in enumerate(teams):
        team_events = events[events['team'] == team]
        ax = axes[idx]
        
        # Draw pitch outline
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 80)
        ax.set_facecolor('#2e8b2e')
        ax.axvline(60, color='white', linewidth=1, alpha=0.5)
        ax.add_patch(plt.Rectangle((0, 18), 18, 44, fill=False, color='white', linewidth=1))
        ax.add_patch(plt.Rectangle((102, 18), 18, 44, fill=False, color='white', linewidth=1))
        
        # Heatmap
        hb = ax.hexbin(team_events['location_x'], team_events['location_y'], 
                       gridsize=15, cmap='YlOrRd', mincnt=1, alpha=0.8)
        ax.set_title(f"{team}", fontsize=14, fontweight='bold', color='#333')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_team_comparison_plot(events):
    """Create bar chart comparing team stats."""
    teams = events['team'].unique()[:2]
    
    stats = []
    for team in teams:
        t = events[events['team'] == team]
        stats.append({
            'team': team,
            'Passes': (t['type'] == 'Pass').sum(),
            'Shots': (t['type'] == 'Shot').sum() * 10,  # Scale for visibility
            'Pressures': (t['type'] == 'Pressure').sum()
        })
    
    df = pd.DataFrame(stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(teams))
    width = 0.25
    
    ax.bar([i - width for i in x], df['Passes'], width, label='Passes', color='#4CAF50')
    ax.bar(x, df['Shots'], width, label='Shots (x10)', color='#f44336')
    ax.bar([i + width for i in x], df['Pressures'], width, label='Pressures', color='#2196F3')
    
    ax.set_xticks(x)
    ax.set_xticklabels(teams, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Team Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        events = load_events_from_file(file)
        events = build_possessions(events)
        features = extract_possession_features(events)
        
        n_possessions = len(features)
        k = min(4, n_possessions)
        
        if k < 2:
            return jsonify({'error': 'Not enough data. Need at least 2 possessions.'}), 400
        
        features = cluster_possessions(features, k=k)
        
        heatmap_plot = create_team_heatmaps(events)
        stats_plot = create_team_comparison_plot(events)
        
        cluster_stats = features.groupby('pattern_cluster').agg({
            'num_passes': 'mean',
            'num_shots': 'mean',
            'duration': 'mean',
            'avg_x': 'mean'
        }).round(2)
        
        # Add human-readable labels
        cluster_labels = label_clusters(cluster_stats)
        
        summary = {
            'total_events': len(events),
            'total_possessions': n_possessions,
            'clusters': k,
            'teams': events['team'].unique().tolist(),
            'cluster_summary': {
                str(idx): {
                    **row.to_dict(),
                    'label': cluster_labels[idx]['label'],
                    'description': cluster_labels[idx]['description']
                }
                for idx, row in cluster_stats.iterrows()
            }
        }
        
        return jsonify({
            'success': True,
            'heatmap_plot': heatmap_plot,
            'stats_plot': stats_plot,
            'summary': summary,
            'team_stats': get_team_stats(events),
            'player_stats': get_player_stats(events)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
