import json
from typing import Any, Dict, List, Union
import os


def load_jsonl_file(
    path: str
) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


def write_jsonl_file(
    path: str,
    data: Union[List[Dict[str, Any]], List[str]],
    overwrite: bool = False
) -> list:
    if overwrite:
        try:
            os.remove(path)
        except:
            pass
    with open(path, 'a+', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
