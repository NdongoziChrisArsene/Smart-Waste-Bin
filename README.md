# Smart Waste-Bin System: time until the bin is full

Multiple Linear Regression, trained on a dataset (no Raspberry Pi or circuit needed), deployed as a Streamlit app.

## Run it
1. `pip install -r requirements.txt`
2. Download `smart_city_sensor_data.csv` from Kaggle (Synthetic Smart City Infrastructure Sensor Data) and put it next to the notebook.
3. Run `Smart_Waste_Bin_Training.ipynb` (Jupyter or Google Colab). It creates the `artifacts/` folder (model, SQLite database, metrics, plots).
4. Start the app: `streamlit run app.py`

The `artifacts/` folder included here was trained on the real Kaggle CSV. Re-running step 3 recreates it on your machine. The app's "Model and data" tab shows which data source was used.

## How it works
- Cleaning: duplicate rows removed, sensor spikes removed, short gaps interpolated.
- Collection events are detected where smoothed fill % drops by more than 20 points (this bin is emptied about weekly).
- "Full" level is set automatically to 90% of the bin's typical peak fill (about 46% in this dataset, which never fills beyond ~50%). Set `FULL_THRESHOLD = 80` in the notebook to use a fixed value.
- Target `hours_to_full` = time until smoothed fill first reaches that level within the same cycle.
- Features: fill %, weight, hours since collection, average fill rate, hour of day, weekend flag.
- Weight is **simulated** (the dataset has no weight). In the app, users enter real weight.
- Split by time (oldest 80% train, newest 20% test). Metrics: MAE, RMSE, R2.

## App inputs
Fill %, weight (kg), last collection date and time, and the current time (automatic or manual).
