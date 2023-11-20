import os
import pandas as pd
import hopsworks
import joblib
import datetime
from datetime import datetime
import dataframe_image as dfi
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import requests
import numpy as np

def process_x_data(data):
    # white -> 0, red -> 1
    type_patterns = [(data['type'] == 'white', 0), (data['type'] == 'red', 1)]
    criteria, values = zip(*type_patterns)
    data['type'] = np.select(criteria, values, 'other')

def inference_wine():
    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    process_x_data(batch_data)

    y_pred = np.round(model.predict(batch_data)).astype(int)

    offset = 1
    pred_qualities = y_pred[y_pred.size - offset:]
    dataset_api = project.get_dataset_api()

    wine_fg = fs.get_feature_group(name='wine', version=1)
    df = wine_fg.read()
    labels = df.iloc[-offset:]['quality']

    monitor_fg = fs.get_or_create_feature_group(
        name='wine_predictions',
        version=1,
        primary_key=["datetime"],
        description="Wine Quality Prediction/Outcome Monitoring"
    )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    data = {
        'prediction': [pred_qualities[-1]],
        'label': [labels.values[-1]],
        'datetime': [now],
    }

    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    history_df = monitor_fg.read(read_options={"use_hive": True})

    history_df = pd.concat([history_df, monitor_df])

    history_df['date_created'] = pd.to_datetime(df['datetime'], dayfirst=True)
    history_df = history_df.sort_values(by=['date_created'], ascending=False)
    
    df_recent = history_df.head(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    results = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(results, list(np.arange(3, 10)), list(np.arange(3, 10)))

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix.png")
    dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    inference_wine()
