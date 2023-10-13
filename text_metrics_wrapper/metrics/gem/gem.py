from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from bleurt import score
import shutil
import os
import subprocess
from pathlib import Path
from text_metrics_wrapper.utils.manage_json_files import write_json_file, load_json_file
from text_metrics_wrapper.utils.environment import load_environment_variables


def Gem(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
) -> Union[Tuple[float, float, float], Tuple[Dict[str, float]]]:
    if isinstance(references[0], str):
        references_gem = [[ref] for ref in references]
    else:
        references_gem = references

    load_environment_variables("/etc/environment")

    # Create a temporary directory and save temporary files
    base_dir = os.environ["TEXT_METRICS_WRAPPER_DIR"]
    temp_dir = os.path.join(base_dir, ".temp")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    hyp_path = os.path.join(temp_dir, "hypothesis.txt")
    ref_path = os.path.join(temp_dir, "references.txt")
    write_json_file(hyp_path, hypothesis, overwrite=True)
    write_json_file(ref_path, references_gem, overwrite=True)

    # Compute metrics
    out_path = os.path.join(temp_dir, "scores.json")
    GEM_dir = os.path.join(base_dir, "GEM-metrics")
    code = ""
    code += f'cd "{GEM_dir}"' + "\n"
    code += "&& ./run_metrics.py" + " "
    code += f"{hyp_path}" + " "
    code += f"-r {ref_path}" + " "
    code += f"-o {out_path}" + " "
    code += f"--metric-list {kwargs['metrics_list']}"
    subprocess.run(code, shell=True, check=True)

    # Reload the scores
    scores = load_json_file(out_path)

    # Remove the hidden directory
    shutil.rmtree(temp_dir)

    return scores
