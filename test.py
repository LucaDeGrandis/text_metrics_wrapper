import os

# Get the value of the environment variable
text_metrics_wrapper_dir = os.environ.get("TEXT_METRICS_WRAPPER_DIR")
text_metrics_wrapper_dir = os.path.abspath(text_metrics_wrapper_dir)

# Print the value
print(text_metrics_wrapper_dir)
