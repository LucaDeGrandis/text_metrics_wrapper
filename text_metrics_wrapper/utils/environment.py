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
    f_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    return logger
