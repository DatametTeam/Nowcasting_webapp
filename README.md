# Weather Nowcasting Web Application

A Streamlit-based web application for visualizing short-term weather radar precipitation forecasts using deep learning models.

## Features

- **Real-time Predictions**: View precipitation forecasts up to 60 minutes ahead with animated timeline
- **Animated Visualization**: Auto-playing timeline with 19 frames (7 ground truth + 12 predictions)
- **Multiple Models**: Support for ConvLSTM, ED_ConvLSTM, DynamicUnet, pystep, and more
- **Interactive Maps**: Leaflet.js-based maps with play/pause controls and speed adjustment
- **Historical Analysis**: Query predictions by date and time
- **Performance Metrics**: CSI scores and other evaluation metrics
- **HPC Integration**: Seamless integration with PBS job scheduler on HPC clusters
- **Local Development**: Automatic mock data generation for development without HPC
- **Status Panel**: Real-time job status tracking per model (Computing, Ready, Not computed)
- **Colored Logging**: Comprehensive logging with colored output for better debugging

## Quick Start

### Prerequisites

- Python 3.10 or higher
- For HPC: PBS/Torque job scheduler
- For local development: Nothing special needed!

### Installation

```bash
# Clone the repository
cd Nowcasting_webapp

# Install dependencies
pip install -e .

# For local development: Generate mock data
python setup_mock_data.py
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run src/nwc_webapp/hello.py
```

The application automatically detects your environment:
- **On HPC**: Uses real PBS job submission and radar data
- **Locally**: Uses synthetic data and simulated predictions

## Project Structure

```
Nowcasting_webapp/
│
├── src/nwc_webapp/
│   ├── services/                   # Business logic layer
│   │   ├── prediction_service.py   # Job submission and management
│   │   └── data_service.py         # Data loading and processing
│   │
│   ├── ui/                         # UI components (WIP)
│   │
│   ├── sou_py/                     # Radar processing library
│   │   ├── dpg/                    # Data processing and geometry
│   │   ├── dpb/                    # Data block handling
│   │   ├── products/               # Radar products (CAPPI, VIL, etc.)
│   │   └── preprocessing/          # Data preprocessing
│   │
│   ├── cfg/
│   │   └── cfg.yaml                # Centralized configuration
│   │
│   ├── hello.py                    # Main Streamlit application
│   ├── config.py                   # Configuration management
│   ├── logging_config.py           # Logging setup
│   ├── environment.py              # Environment detection (HPC vs local)
│   ├── mock.py                     # Mock PBS for local development
│   ├── mock_data_generator.py      # Synthetic data generation
│   ├── pbs.py                      # PBS job management
│   ├── utils.py                    # Utility functions
│   ├── graphics.py                 # Visualization utilities
│   ├── layouts.py                  # Streamlit layouts
│   └── parallel_code.py            # Parallel processing
│
├── data/                           # Data directory (auto-created)
│   ├── mock_sri/                   # Mock SRI files (local dev)
│   ├── predictions/                # Prediction outputs
│   └── nodes/                      # Node data
│
├── logs/                           # Application logs
│
├── setup_mock_data.py              # Mock data setup script
├── pyproject.toml                  # Project dependencies
└── CLAUDE.md                       # Development guide

```

## Configuration

All settings are centralized in `src/nwc_webapp/cfg/cfg.yaml`:

- **Models**: Available prediction models
- **Paths**: Data locations (separate for HPC and local)
- **Visualization**: Map settings, colormaps, GIF parameters
- **Prediction**: Timesteps, forecast horizons
- **PBS/HPC**: Queue settings, GPU targets, conda environments
- **Auto-refresh**: Update intervals
- **Logging**: Log levels and destinations

## Development

### Environment Detection

The application uses automatic environment detection:

```python
from nwc_webapp.environment import is_hpc, is_local
from nwc_webapp.config import get_config

config = get_config()
sri_folder = config.sri_folder  # Automatically uses HPC or local path
```

### Logging

Centralized colored logging system with timestamps:

```python
from nwc_webapp.logging_config import setup_logger

logger = setup_logger(__name__)
logger.info("Processing data...")
logger.warning("This is a warning")
logger.error("Something went wrong!", exc_info=True)
```

**Features**:
- ERROR messages: Entire message colored RED
- WARNING messages: Entire message colored YELLOW
- DEBUG messages: "DEBUG" prefix colored BLUE
- Timestamp format: `YYYY-MM-DD HH:MM:SS` (to the second)

### Adding a New Model

Adding a new model to the webapp is straightforward and requires creating configuration files and ensuring your inference script is compatible. Follow these steps:

#### 1. Add Model to Main Configuration

Add your model name to the models list in `src/nwc_webapp/resources/cfg/cfg.yaml`:

```yaml
models:
  - ConvLSTM
  - ED_ConvLSTM
  - IAM4VP
  - PredFormer
  - SPROG
  - YourNewModel  # ← Add here
```

#### 2. Create Real-Time Prediction Config

Create `src/nwc_webapp/resources/cfg/real_time_prediction_cfg/{YourNewModel}.yaml`:

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

trainings_artifacts:
  training_cfg_path: '/path/to/your/model/config_used.yaml'
  checkpoint_path: '/path/to/your/model/best_checkpoint.ckpt'

transforms:
  0:
    name: "LogarithmicTransform"
    args:
      threshold: 0.2
    adjust: ["threshold"]
  1:
    name: "MinMaxNormalize"
    args:
      excluded_value: -9999
      input_min: 0.0
      input_max: 300.0
      output_min: 0.0
      output_max: 1.0
    adjust: ["input_min", "input_max", "excluded_value"]

windows_extraction_args:
  hdf_key: /dataset1/data1/data
  lead_time: 12
  wlen: 12
  stride: 1
  interval_minutes: 5

dataframe_strategy:
  input:
    name: from_hdf_folder
    args:
      data_folder: /davinci-1/work/protezionecivile/data
      datetime_format: '%d-%m-%Y-%H-%M'
      fallback_folder: /davinci-1/work/protezionecivile/data1/SRI_adj
      use_only_fallback: true  # Use only fallback folder for real-time (faster)

  slicers:
    - name: last_n_timestamps  # Get most recent data
      args:
        n_timestamps: 12
        interval_minutes: 5

    - name: continuity  # Handle gaps in data
      args:
        freq: 5min
        fill: true
        max_gap_frames: 3

save_path: /davinci-1/work/protezionecivile/nwc_webapp/real_time_results/YourNewModel
save_npz: False
save_npy: True
```

#### 3. Create Date-Range Prediction Config

Create `src/nwc_webapp/resources/cfg/start_end_prediction_cfg/{YourNewModel}.yaml`:

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

trainings_artifacts:
  training_cfg_path: /path/to/your/model/config_used.yaml
  checkpoint_path: /path/to/your/model/best_checkpoint.ckpt

transforms:
  0:
    name: LogarithmicTransform
    args:
      threshold: 0.2
    adjust: [threshold]
  1:
    name: MinMaxNormalize
    args:
      excluded_value: -9999
      input_min: 0.0
      input_max: 300.0
      output_min: 0.0
      output_max: 1.0
    adjust: [input_min, input_max, excluded_value]

windows_extraction_args:
  hdf_key: /dataset1/data1/data
  lead_time: 12
  wlen: 12
  stride: 1
  interval_minutes: 5

dataframe_strategy:
  input:
    name: from_hdf_folder
    args:
      data_folder: /davinci-1/work/protezionecivile/data
      datetime_format: '%d-%m-%Y-%H-%M'
      fallback_folder: /davinci-1/work/protezionecivile/data1/SRI_adj

  slicers:
    - name: start_end  # Process specific date range
      args:
        start_dt: "2025-11-22 20:00:00"  # Placeholder - replaced at runtime
        end_dt: "2025-11-22 20:50:00"    # Placeholder - replaced at runtime

    - name: continuity  # Handle gaps in data
      args:
        freq: 5min
        fill: true
        max_gap_frames: 3

save_path: /davinci-1/work/protezionecivile/nwc_webapp/real_time_results/YourNewModel
save_npz: false
save_npy: true
```

**Key Differences:**
- Real-time uses `last_n_timestamps` slicer (gets most recent N frames) with `use_only_fallback: true` for faster loading
- Start-end uses `start_end` slicer (processes specific date range with `start_dt`/`end_dt`)
- Both use `continuity` slicer to handle gaps in radar data (allows up to 15 minutes of missing data)

#### 4. (Optional) Configure Conda Environment

If your model requires a specific conda environment (different from the default `nowcasting3.12`), add it to `cfg.yaml`:

```yaml
pbs:
  queue: "fast"
  walltime: "12:00:00"

  environments:
    ED_ConvLSTM: "protezionecivile"
    YourNewModel: "your_custom_env"  # ← Add here if needed
    default: "nowcasting3.12"
```

If not specified, the model will use the `default` environment.

#### 5. Ensure Inference Script Compatibility

Your inference script (in the `spatiotemporal-nowcast` repository) must be compatible with the webapp interface:

**Required:**
- ✅ Accept `--cfg_path` command-line argument
- ✅ Read configuration from the provided YAML file
- ✅ Output predictions as `.npy` files
- ✅ Prediction shape: `(12, 1400, 1200)` — 12 timesteps at 5-minute intervals
- ✅ File naming format: `DD-MM-YYYY-HH-MM.npy` (e.g., `22-11-2025-20-00.npy`)
- ✅ Save to the directory specified in config's `save_path`

#### 6. Verify and Test

Once configured, the model will automatically appear in:
- Real-time Prediction tab (dropdown menu)
- Nowcasting tab (date range predictions)
- Metrics Analysis tab (for evaluation)

**No code changes needed in the webapp!** The UI automatically updates based on `cfg.yaml`.

#### Summary Checklist

- [ ] Add model name to `cfg.yaml` → `models:` list
- [ ] Create `real_time_prediction_cfg/{model}.yaml`
- [ ] Create `start_end_prediction_cfg/{model}.yaml`
- [ ] (Optional) Add custom conda environment to `cfg.yaml` → `pbs.environments:`
- [ ] Verify inference script accepts `--cfg_path` and outputs correct format
- [ ] Test on both real-time and date-range prediction workflows

### Mock Data for Testing

The mock data system generates realistic precipitation fields:

```bash
# Regenerate mock data
python setup_mock_data.py
```

Mock data includes:
- 20 SRI files with 5-minute intervals
- Prediction outputs for all models
- Realistic temporal evolution

## Architecture

### Data Flow

1. **Input**: Radar SRI data (HDF5 files) with 5-minute intervals
2. **Processing**: Models generate 60-minute forecasts
3. **Visualization**: Maps and GIFs show precipitation evolution
4. **Output**: NPY files with prediction arrays

### Coordinate System

- **Projection**: Transverse Mercator
