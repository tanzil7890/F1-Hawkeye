# F1_Hawkeye

```
 _____ _   _   _   _                _
|  ___/ | | | | | | | __ ___      _| | _____ _   _  ___
| |_  | | | |_| |_| |/ _` \ \ /\ / / |/ / _ \ | | |/ _ \
|  _| | | |  _  _  | (_| |\ V  V /|   <  __/ |_| |  __/
|_|   |_| |_| |_| |_|\__,_| \_/\_/ |_|\_\___|\__, |\___|
                                             |___/
```

### Professional F1 Telemetry Analysis Platform

*Real-time telemetry | ML-powered predictions | Advanced analytics | Strategy optimization*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/Qt-PySide6-green.svg)](https://www.qt.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Architecture](#architecture)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**F1_Hawkeye** is a professional-grade telemetry analysis platform for F1 2025 and previous F1 games. It combines real-time data visualization, machine learning predictions, and advanced analytics to provide race engineers, sim racers, and F1 enthusiasts with deep insights into race performance.

### What Makes F1_Hawkeye Special?

- **Real-Time Telemetry**: Live UDP data streaming from F1 games (20-60 Hz)
- **AI-Powered Predictions**: ML models for tyre wear, lap times, pit stops, and race outcomes
- **Advanced Analytics**: Historical analysis, pace evolution, sector performance
- **Modern UI**: Dark F1-inspired theme with 12 interactive dashboards
- **Database Integration**: SQLite persistence with export to CSV/Parquet
- **REST API**: External access to telemetry and predictions
- **Cloud Ready**: AWS S3 and Google Cloud Storage integration
- **Anomaly Detection**: AI-powered detection of flatspots, damage, and performance issues

### Key Statistics

- **50,000+** lines of code
- **100+** Python files
- **50+** features implemented
- **4** ML prediction models
- **8** database tables
- **12** interactive dashboards

---

## Features

### Core Telemetry Display

#### Real-Time Dashboards (9 Original Tabs)

1. **Main** - Race positions, lap times, tyre compounds, ERS, DRS status
2. **Damage** - Car component damage percentages with visual indicators
3. **Laps** - Detailed lap times, sector splits, validity status
4. **Temperatures** - Tyre surface/inner temps, brake temps
5. **Map** - Live track map with car positions and marshal zones
6. **ERS & Fuel** - Energy recovery metrics, fuel consumption, mix settings
7. **Weather** - Track temperature, air temperature, rain forecast
8. **Packet Reception** - Network statistics and data reception rates
9. **Race Director** - Event log (penalties, fastest laps, incidents)

### Machine Learning & Analytics (Phase 2-5)

#### Advanced Dashboards (3 New Tabs)

10. **Analytics** - Historical tyre wear, pace evolution, sector heatmaps
11. **Predictions** - ML-powered forecasts for tyres, lap times, pit windows
12. **Strategy** - Pit stop optimization, strategy comparison, race simulation

#### AI Features

-  **Tyre Wear Prediction** - XGBoost models with 15-lap forecasting
-  **Lap Time Forecasting** - Degradation-aware lap time prediction
-  **Pit Stop Optimization** - Monte Carlo simulation for optimal pit timing
- [x] **Race Outcome Prediction** - Win/podium/points probabilities
-  **Anomaly Detection** - Isolation Forest for flatspots, damage, asymmetric wear
-  **Track-Specific Learning** - Models trained per circuit for improved accuracy
-  **Live Strategy Alerts** - Real-time recommendations for undercuts, tyre wear

### Data Management

- **Database**: SQLite with SQLAlchemy ORM (8+ tables)
- **Export Formats**: CSV (human-readable) and Parquet (ML-optimized)
- **Import**: Historical CSV data with session recreation
- **Batch Operations**: Export all sessions with one click
- **Statistics**: Session counts, lap data, telemetry snapshots

### REST API (Optional)

FastAPI-based REST API for external applications:

```bash
# Start API server
uvicorn src.api.main:app --reload --port 8000

# Access at http://localhost:8000/docs
```

**Endpoints:**
- `GET /sessions` - List all recorded sessions
- `GET /sessions/{id}` - Session details with statistics
- `POST /predictions` - Get ML predictions for current state
- `GET /anomalies/{session_id}/{driver}` - Detect anomalies
- `GET /health` - Health check

### Cloud Integration (Optional)

Upload telemetry and models to cloud storage:

```python
from src.cloud import CloudStorage

# AWS S3
storage = CloudStorage(provider='aws', credentials={
    'key': 'YOUR_KEY',
    'secret': 'YOUR_SECRET'
})
storage.upload_session(session_id=1, bucket='f1-telemetry')

# Google Cloud Storage
storage = CloudStorage(provider='gcp', credentials={
    'project': 'YOUR_PROJECT'
})
storage.upload_session(session_id=1, bucket='f1-telemetry')
```

---

## Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **4GB RAM** minimum (8GB+ for ML training)
- **2GB disk space** (for database and models)
- **UDP port 20777** available

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/f1-hawkeye.git
cd f1-hawkeye

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Installation Options

#### Option 1: Full Installation (All Features)

```bash
pip install -r requirements.txt
```

**Includes:**
- Real-time telemetry
- Database persistence
- ML predictions
- Advanced analytics
- REST API
- Cloud storage (optional)

#### Option 2: Minimal Installation (Display Only)

```bash
pip install PySide6 ttkbootstrap
```

**Includes:**
- Real-time telemetry display only
- No database, ML, or analytics

#### Option 3: API Server Only

```bash
pip install fastapi pydantic uvicorn[standard]
```

Perfect for headless servers.

---

## Quick Start

### 1. Launch the Application

```bash
python f1_hawkeye_app.py
```

### 2. Choose Your Mode

**Startup Dialog** will appear with two options:

#### Real-Time Mode (LIVE)

Connect to F1 game for live telemetry:

1. Launch F1 2025 (or F1 22/23/24)
2. Enable UDP telemetry in game settings:
   - Settings → Telemetry Settings
   - UDP Telemetry: **ON**
   - UDP Port: **20777**
   - UDP Format: **2025** (or your game year)
3. Click **"Real-time Mode"** in F1_Hawkeye
4. Start racing! Data updates live.

#### Manual Data Mode (PLAYBACK)

Replay pre-recorded telemetry files:

1. Click **"Manual Data Mode"**
2. Select a `.bin` telemetry file
3. Data plays back in loop
4. Perfect for analysis without running the game

### 3. Explore the Features

#### Basic Navigation

- **Sidebar Menu**: Click tabs to switch views
- **Top Menu Bar**: Export, Database, Advanced features
- **Keyboard Shortcuts**:
  - `Ctrl+1-9`: Switch tabs
  - `Ctrl+E`: Export menu
  - `F5`: Refresh data
  - `F11`: Fullscreen

#### Try These Features

1. **View Real-Time Data**: Main tab shows live positions
2. **Check Tyre Wear**: Damage tab → tyre percentages
3. **Analyze Laps**: Laps tab → sector times
4. **Track Position**: Map tab → live car positions
5. **Run Analytics**: Analytics tab → load session
6. **Get Predictions**: Predictions tab → load ML models
7. **Compare Strategies**: Strategy tab → analyze pit stops

---


## Screenshots

### Main Dashboard

Real-time race positions with tyre compounds, ERS, and lap times:

```
+-------------------------------------------------------------+
| Monaco - Qualifying - Lap 5/12                              |
+-------------------------------------------------------------+
| Pos | Driver    | Tyre | Lap Time | S1     | S2     | S3   |
|  1  | VER       | SOFT | 1:10.504 | 23.104 | 28.774 | ...  |
|  2  | LEC       | SOFT | 1:10.782 | 23.245 | 28.892 | ...  |
|  3  | HAM       | SOFT | 1:10.891 | 23.301 | 28.945 | ...  |
+-------------------------------------------------------------+
```

### Analytics Dashboard

Historical tyre wear with ML predictions:

```
Tyre Wear Evolution
100% |                                            /
 80% |                                   /---/
 60% |                       /---/
 40% |           /---/
 20% |   /---/          <- SOFT Compound
  0% +--------------------------------------------
     0   5   10  15  20  25  30  (Lap)
         ^- Actual    ^- ML Prediction
```

### Strategy Comparison

1-stop vs 2-stop analysis:

```
+--------+---------+----------+--------+--------+
| Strat  | Pit Lap | Expected | Podium | Risk   |
+--------+---------+----------+--------+--------+
| 1-STOP | Lap 35  |   P4     |  45%   | Medium |
| 2-STOP | Lap 25  |   P5     |  38%   | Low    |
| 0-STOP |    -    |   P12    |   0%   | High   |
+--------+---------+----------+--------+--------+
```

### Live Track Map

Real-time car positions with marshal zones:

```
        Monaco Circuit
     +-------------------+
     |   (Finish)        |  <- Start/Finish
  +--+        [1]        |  <- VER (P1)
  |           [2]        |  <- LEC (P2)
  |                      |
  |     [!] Marshal      |  <- Yellow flag
  +----------+           |
             |  [3]      |  <- HAM (P3)
             +----------+
```

---

## Architecture

### Project Structure

```
f1-hawkeye/
├── src/
│   ├── windows/              # UI windows and dialogs
│   │   ├── main_window.py    # Main application window
│   │   ├── analytics_window.py
│   │   ├── prediction_window.py
│   │   └── strategy_window.py
│   ├── table_models/         # Data table models
│   ├── visualization/        # Chart components
│   │   ├── tyre_wear_chart.py
│   │   ├── pace_evolution_chart.py
│   │   └── strategy_simulator.py
│   ├── analysis/             # Analytics modules
│   │   ├── tyre_analytics.py
│   │   ├── pace_analytics.py
│   │   └── fuel_analytics.py
│   ├── ml/                   # Machine learning
│   │   ├── models/           # ML model classes
│   │   ├── features/         # Feature engineering
│   │   ├── training.py       # Model training
│   │   └── inference.py      # Real-time predictions
│   ├── advanced/             # Advanced features
│   │   ├── anomaly_detection.py
│   │   └── multi_session_learning.py
│   ├── database/             # Database layer
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── db_manager.py     # Database operations
│   │   └── repositories/     # Data access layer
│   ├── export/               # Data export
│   │   ├── csv_exporter.py
│   │   └── parquet_exporter.py
│   ├── importers/            # Data import
│   ├── api/                  # REST API
│   │   └── main.py           # FastAPI app
│   ├── cloud/                # Cloud integration
│   │   └── storage.py        # S3/GCS upload
│   ├── alerts/               # Strategy alerts
│   ├── parsers/              # Packet parsers
│   └── packet_processing/    # UDP packet handling
├── models/                   # Trained ML models (gitignored)
├── data/                     # Sample telemetry files
├── extra/                    # Documentation
├── Telemetry.py             # Main entry point
├── requirements.txt          # Dependencies
├── style.css                # Dark theme stylesheet
└── README.md                # This file
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **GUI** | PySide6 (Qt6) | Modern cross-platform interface |
| **Database** | SQLite + SQLAlchemy | Local data persistence |
| **ML** | scikit-learn, XGBoost | Predictive models |
| **Analytics** | pandas, numpy | Data processing |
| **Visualization** | matplotlib, plotly | Charts and graphs |
| **API** | FastAPI | REST API server |
| **Cloud** | boto3, GCS | Cloud storage |
| **Parsing** | ctypes, struct | UDP packet parsing |

### Data Flow

```
F1 Game (UDP:20777)
      |
      v
SocketThread (Background)
      |
      v
Packet Parser (ctypes)
      |
      v
Data Writer (Queue -> SQLite)
      |
      v
+-----------------------------+
|  In-Memory (Real-time)      |  ->  UI Tables (60 FPS)
|  + Database (Historical)    |  ->  Analytics Dashboard
+-----------------------------+  ->  ML Predictions
                                 ->  REST API
                                 ->  Cloud Storage
```

---

## Usage Examples

### Example 1: Real-Time Race Monitoring

```bash
# 1. Start the application
python Telemetry.py

# 2. Select "Real-time Mode"

# 3. Start F1 game race

# 4. Watch live telemetry update

# 5. After race, export data:
#    Menu -> Export -> Export to CSV
```

### Example 2: Historical Analysis

```python
# Analyze a previous session
from src.analysis import TyreAnalytics

# Load session
analytics = TyreAnalytics(session_id=1)

# Get wear statistics
wear_stats = analytics.calculate_degradation_rate(driver_index=0)
print(f"Avg degradation: {wear_stats['avg_degradation_per_lap']:.2f}%/lap")

# Predict remaining laps
remaining = analytics.predict_remaining_laps(
    current_wear=65.0,
    avg_degradation=2.3
)
print(f"Tyres can last {remaining} more laps")
```

### Example 3: Train Custom ML Model

```python
from src.ml.training import train_all_models

# Train models on your data
results = train_all_models(
    session_ids=[1, 2, 3, 4, 5],  # Your session IDs
    output_dir='models/custom/'
)

# View results
for model_type, metrics in results.items():
    print(f"{model_type}: R² = {metrics['test_r2']:.3f}")
```

### Example 4: Get Live Predictions

```python
from src.ml.inference import RealTimePredictor

# Load trained models
predictor = RealTimePredictor()
predictor.load_models('models/')

# Get predictions
predictions = predictor.predict({
    'current_lap': 25,
    'current_position': 5,
    'tyre_age': 15,
    'avg_wear': 52.0,
    'total_laps': 50
})

print(f"Expected finish position: P{predictions['final_position']}")
print(f"Recommended pit lap: {predictions['optimal_pit_lap']}")
print(f"Podium probability: {predictions['podium_probability']:.1%}")
```

### Example 5: Detect Anomalies

```python
from src.advanced import AnomalyDetector

detector = AnomalyDetector()

# Detect all anomalies
anomalies = detector.detect_all_anomalies(
    session_id=1,
    driver_index=0
)

# Generate report
report = detector.generate_report(anomalies)
print(report)

# Output:
# ============================================================
# ANOMALY DETECTION REPORT
# ============================================================
# Total Anomalies Detected: 3
#
# TYRE ANOMALIES (2):
# [!] Lap 15: FLATSPOT
#    Sudden wear spike: +12.5% in one lap
#    Severity: CRITICAL
```

---

## Roadmap

### Completed (v1.0)

- [x] Real-time telemetry display (9 tabs)
- [x] Database persistence (SQLite)
- [x] CSV/Parquet export
- [x] Advanced analytics (Phase 2)
- [x] ML predictions (Phase 3)
- [x] Interactive charts (Phase 4)
- [x] Anomaly detection (Phase 5)
- [x] REST API
- [x] Cloud storage integration


---

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add: Real-time sector comparison feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Setup

```bash
# Install with development dependencies
pip install -r requirements.txt
# Uncomment dev tools in requirements.txt first

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions/classes
- Include type hints where possible
- Write unit tests for new features
- Update documentation
- Keep commits atomic and well-described

---

## Troubleshooting

### Common Issues

#### Issue: "Advanced windows not available"

**Solution:**
```bash
pip install --upgrade matplotlib plotly pandas scikit-learn
```

#### Issue: "Database not available"

**Solution:**
```bash
pip install --upgrade SQLAlchemy alembic
```

#### Issue: "No telemetry received"

**Solutions:**
1. Check F1 game UDP settings (port 20777)
2. Disable firewall for port 20777
3. Verify game is running
4. Try manual data mode with sample file

#### Issue: "FastAPI not available"

**Solution:**
```bash
pip install fastapi pydantic uvicorn[standard]
```

#### Issue: "Models not loading"

**Solution:** Train models first:
```python
from src.ml.training import train_all_models
train_all_models(session_ids=[1, 2, 3], output_dir='models/')
```

### Performance Issues

- **High CPU usage**: Reduce packet reception rate in game
- **Memory leaks**: Restart application after long sessions
- **Slow ML predictions**: Use smaller models or reduce features
- **Database growing large**: Export old sessions and delete

### Getting Help

- **Email**: support@f1-hawkeye.com
- **Discord**: Join our community server
- **Issues**: [GitHub Issues](https://github.com/yourusername/f1-hawkeye/issues)
- **Docs**: See documentation files in `extra/`

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 F1_Hawkeye Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

### Built With

- **[PySide6](https://wiki.qt.io/Qt_for_Python)** - Qt for Python framework
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting framework
- **[pandas](https://pandas.pydata.org/)** - Data analysis library
- **[matplotlib](https://matplotlib.org/)** - Visualization library
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - SQL toolkit and ORM

### Inspiration

- **Codemasters/EA F1 Games** - For the telemetry API
- **F1 Teams** - Real race engineering workflows
- **Community** - Feedback and feature requests

### Special Thanks

- All contributors who helped build F1_Hawkeye
- F1 game modding community for documentation
- Open source projects that made this possible

---

## Project Stats

- **Lines of Code**: 50,000+
- **Python Files**: 100+
- **Features**: 50+
- **ML Models**: 4 (Tyre Wear, Lap Time, Pit Stop, Race Outcome)
- **Database Tables**: 8
- **API Endpoints**: 6
- **Chart Types**: 5
- **Development Time**: 6+ months

---

## Links

- **GitHub**: [https://github.com/tanzil7890/f1-hawkeye](https://github.com/tanzil7890/f1-hawkeye)


---

<div align="center">

### Star this project if you find it useful!

Made with passion by the F1_Hawkeye team

**[Back to top](#f1_hawkeye)**

</div>
