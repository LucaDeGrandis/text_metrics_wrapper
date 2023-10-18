from typing import Dict, Any
import os
import logging


def load_environment_variables(
    path: str,
) -> None:
    environment_variables = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line.startswith("export"):
                os.environ[line.split("=")[0].split(" ")[1]] = line.split("=")[1].replace("\n", "").replace("'", "")


def set_logger(log_file_path):
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger
