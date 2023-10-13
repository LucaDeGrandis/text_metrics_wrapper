from typing import Dict, Any
import os


def load_environment_variables(
    path: str,
) -> None:
    environment_variables = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line.startswith("export"):
                os.environ[line.split("=")[0].split(" ")[1]] = line.split("=")[1].replace("\n", "")
