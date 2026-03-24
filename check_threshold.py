import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

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