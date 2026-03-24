import mlflow
import os
import sys

import dagshub

# Initialize DAGsHub MLflow tracking
dagshub.init(repo_owner='finebartok', repo_name='mlops-ci-pipeline_5', mlflow=True)

# Read Run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics["accuracy"]

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Accuracy below threshold. Failing pipeline.")
    sys.exit(1)
else:
    print("✅ Accuracy meets threshold.")