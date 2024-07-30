import mlflow

import argparse

import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description="House Prices ML")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="Taxa de aprendizado para atualizar o tamanho do passo durante o boosting"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profundidade máxima das árvores de decisão"
    )
    return parser.parse_args()


SEED = 42

processed_data_path = "data/processed/"
filename = "casas.csv"
full_path = processed_data_path + filename

df = pd.read_csv(full_path)

features = df.drop(columns=["preco"])
target = df["preco"]
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=SEED)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "house-prices-script"
mlflow.set_experiment(experiment_name)


def main():

    args = parse_args()
    xgb_params = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "seed": SEED
    }

    with mlflow.start_run():

        mlflow.xgboost.autolog()

        dtrain = xgboost.DMatrix(features_train, label=target_train)
        dtest = xgboost.DMatrix(features_test, label=target_test)

        xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, "train")])

        predicted = xgb.predict(dtest)

        r2 = r2_score(target_test, predicted)
        rmse = root_mean_squared_error(target_test, predicted)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic XGBoost model for houses data")

        # Log metrics
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    main()
