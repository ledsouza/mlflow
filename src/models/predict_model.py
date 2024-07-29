import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8080")
logged_model = 'runs:/b5507f80842d4bc6975b1071ba36e1bc/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
df = pd.read_csv("data/processed/casas.csv")
data = df.drop(columns=["preco"])

predicted = loaded_model.predict(data)
data["predicted"] = predicted

data.to_csv("precos_previstos.csv")
