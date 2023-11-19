import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("iris_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/iris_model.pkl")
print("Model downloaded")


def wine(type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]], 
                      columns=['type', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=wine,
    title="Wine Predictor",
    description="Input Known Wine Features to Deterine the Type",
    allow_flagging="never",
    inputs=[
        gr.Dropdown(["red", "white"], label="Type", info=""),
        gr.inputs.Number(default=1.0, label="fixed_acidity"),
        gr.inputs.Number(default=1.0, label="volatile_acidity"),
        gr.inputs.Number(default=1.0, label="citric_acid"),
        gr.inputs.Number(default=1.0, label="residual_sugar"),
        gr.inputs.Number(default=1.0, label="chlorides"),
        gr.inputs.Number(default=1.0, label="free_sulfur_dioxide"),
        gr.inputs.Number(default=1.0, label="total_sulfur_dioxide"),
        gr.inputs.Number(default=1.0, label="density"),
        gr.inputs.Number(default=1.0, label="pH"),
        gr.inputs.Number(default=1.0, label="sulphates"),
        gr.inputs.Number(default=17.0, label="alcohol"),
        ],
    outputs=gr.Number(default=1, label="quality"))

demo.launch(debug=True)

