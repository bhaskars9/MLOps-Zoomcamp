import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from datetime import datetime as dt
from datetime import timedelta as td
import pickle
import mlflow

def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical,date):
    with mlflow.start_run(): 
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values

        with open(f"models/dv-{date}.b", "wb+") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(f"models/dv-{date}.b", artifact_path="preprocessor")

        print(f"The shape of X_train is {X_train.shape}")
        print(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        print(f"The MSE of training is: {mse}")
        
        with open(f'models/model-{date}.bin', 'wb') as f_out:
            pickle.dump(lr, f_out)

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

        return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


def get_paths(date):
    traindate = (date-td(60)).strftime("%Y-%m")
    valdate = (date-td(30)).strftime("%Y-%m")
    train_path = f'../data/fhv_tripdata_{traindate}.parquet'
    val_path = f'../data/fhv_tripdata_{valdate}.parquet'
    print(train_path)
    print(val_path)
    return train_path,val_path

@flow
def main(date = "None"):
    if(date=="None"):
        train_path, val_path = get_paths(dt.now())
    else:
        date_new = dt.strptime(date, "%Y-%m-%d")
        train_path, val_path = get_paths(date_new)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr)
    
main(date="2021-08-15")

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *", timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)