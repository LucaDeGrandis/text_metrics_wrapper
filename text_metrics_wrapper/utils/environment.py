from typing import Dict, Any


def load_environment_variables(
    path: str,
) -> None:
    environment_variables = []
    with open(path, "r") as file:
        for line in file.readlines():
            if line.startswith("export"):
                os.environ[line.split("=")[0].split(" ")] = line.split("=")[1]
