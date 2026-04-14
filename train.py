import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import mlflow
import mlflow.sklearn
import dagshub

# Initialize DAGsHub if environment variables are present
if os.envirdagon.get("DAGSHUB_USER_TOKEN") or (os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD")):
    dagshub.init(repo_owner='finebartok', repo_name='mlops-ci-pipeline_5', mlflow=True)
else:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save Run ID
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc}")