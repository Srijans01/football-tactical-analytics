# ⚽ Football Match Tactical Analysis

A web application that analyzes football match event data to provide tactical insights, team comparisons, and player statistics.

## 🎯 Features

- **Team Possession Analysis** - See which team dominated the ball
- **Activity Heatmaps** - Visual pitch maps showing where each team played
- **Team Comparison** - Side-by-side stats (passes, shots, pressures)
- **Top Players** - Most active players with detailed stats
- **Match Insights** - AI-generated tactical observations

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd football-event-tactical-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open Browser
Go to **http://localhost:5000**

### 4. Get Match Data
Download any JSON file from [StatsBomb Open Data Events](https://github.com/statsbomb/open-data/tree/master/data/events)

Example: `15946.json`, `9880.json`, etc.

### 5. Upload & Analyze
Drag and drop the JSON file into the web app to see the analysis!

## 📁 Project Structure

```
football-event-tactical-analysis/
├── app.py                 # Flask web application
├── templates/
│   └── index.html         # Web UI
├── src/
│   ├── data_loader.py     # Load event data
│   ├── possession_builder.py
│   ├── feature_engineering.py
│   ├── pattern_detection.py
│   └── visualization.py
├── data/
│   └── raw/               # Sample data
├── requirements.txt
└── README.md
```

## 📊 Supported Data Formats

| Format | Source | Structure |
|--------|--------|-----------|
| StatsBomb JSON | [open-data/events](https://github.com/statsbomb/open-data/tree/master/data/events) | Array of event objects |
| Simple JSON | Custom | `{"events": [...]}` |
| CSV | Custom | Columns: match_id, timestamp, team, type, location_x, location_y |

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **Frontend**: Bootstrap 5, Font Awesome

## 📈 Analysis Outputs

| Metric | Description |
|--------|-------------|
| Possession % | Ball control based on pass counts |
| Shots | Total shooting attempts per team |
| Pressures | Defensive pressing actions |
| Heatmap | Spatial distribution of team activity |
| Top Players | Ranked by total events, passes, shots |
