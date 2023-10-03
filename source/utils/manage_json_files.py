import json
from typing import Any, Dict, List, Union
import os


def load_json_file(path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], overwrite: bool = False) -> None:
    if overwrite:
        try:
            os.remove(path)
        except:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
