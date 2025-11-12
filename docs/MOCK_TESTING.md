# Mock Data Testing Guide

This guide explains how to test the real-time prediction tab locally using the mock data system.

## Overview

The mock system simulates the entire real-time prediction workflow without requiring HPC access:

1. **Mock SRI Data Generator**: Drops new radar data files every 30 seconds (configurable)
2. **Mock PBS Submission**: Simulates job submission with 5-10 second processing time
3. **Mock Predictions**: Auto-generates realistic precipitation predictions

## Quick Start

### 1. Start the Application

```bash
# Activate environment
conda activate nwc_webapp

# Run Streamlit app
streamlit run src/nwc_webapp/hello.py
```

The mock system will **automatically start** when running locally (not on HPC).

### 2. What Happens Automatically

When you start the app locally:

1. ✅ **Environment Detection**: Detects you're running locally
2. 📁 **Folder Creation**: Creates `data/mock_sri/` folder
3. 📜 **Historical Data**: Generates 12 historical radar files (last hour)
4. 🔄 **Background Service**: Starts generating new files every 30 seconds
5. 🎭 **Mock PBS**: Uses mock job submission instead of real PBS

### 3. Testing the Real-Time Tab

1. Open the app in your browser (usually `http://localhost:8501`)
2. Go to the **"Real Time Prediction"** tab (first tab)
3. Select a model and prediction time
4. Watch the process:
   - New radar data files appear every 30 seconds
   - When new data is detected, mock jobs are submitted
   - Predictions are generated in 5-10 seconds
   - Results appear on the map

## Configuration

### Change Data Generation Interval

Edit `src/nwc_webapp/hello.py`:

```python
# For testing (30 seconds)
start_mock_service(interval_seconds=30, generate_history=True)

# For realistic simulation (5 minutes)
start_mock_service(interval_seconds=300, generate_history=True)
```

### Manually Generate Mock Data

```bash
# From project root
python src/nwc_webapp/mock_data_generator.py
```

### Test Mock PBS System

```bash
# Test mock job submission
python src/nwc_webapp/mock.py
```

### Test Real-Time Service

```bash
# Test continuous data generation
python src/nwc_webapp/services/mock_realtime_service.py
```

## Data Locations

### Local Mode

- **SRI Data**: `data/mock_sri/`
- **Predictions**: `data/predictions/{model_name}/`
- **Real-time**: `data/predictions/real_time_pred/`

### HPC Mode

- **SRI Data**: `/davinci-1/work/protezionecivile/data1/SRI_adj`
- **Predictions**: `/davinci-1/work/protezionecivile/sole24/pred_teo/{model_name}/`
- **Real-time**: `/davinci-1/work/protezionecivile/sole24/pred_teo/real_time_pred/`

## File Naming Convention

Mock files follow the same naming as real data:

```
dd-mm-yyyy-HH-MM.hdf
```

Examples:
- `11-11-2025-14-35.hdf`
- `11-11-2025-14-40.hdf`
- `11-11-2025-14-45.hdf`

## Troubleshooting

### No New Files Appearing

Check console output for:
```
🎭 Mock realtime service started (local mode)
✅ Generated mock SRI file: 11-11-2025-14-35.hdf
```

### Jobs Not Submitting

Check that mock imports are working:
```python
from nwc_webapp.environment import is_local
print(f"Running locally: {is_local()}")
```

### Predictions Not Loading

1. Check `data/predictions/` folder exists
2. Verify `.npy` files are being created
3. Check console for errors

## Development Tips

### Fast Testing

Use 10-second intervals for rapid testing:

```python
start_mock_service(interval_seconds=10, generate_history=True)
```

### Skip Historical Data

If you already have data:

```python
start_mock_service(interval_seconds=30, generate_history=False)
```

### Clean Up Mock Data

```bash
# Remove all mock data
rm -rf data/mock_sri/*
rm -rf data/predictions/*
```

## Next Steps

Once the mock system is working:

1. ✅ Verify map displays predictions correctly
2. ✅ Test with multiple models
3. ✅ Test error scenarios (missing files, failed jobs)
4. ✅ Optimize performance and loading times
5. ✅ Add proper status tracking UI

## Notes

- Mock system only runs when `is_local()` returns True
- On HPC, real PBS and real data are used automatically
- Mock predictions use realistic random data
- Processing times are simulated (5-10 seconds per model)