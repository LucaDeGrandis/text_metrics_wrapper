from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from bleurt import score
import shutil
import os
import subprocess
from pathlib import Path
from text_metrics_wrapper.utils.manage_json_files import write_json_file, load_json_file
from text_metrics_wrapper.utils.environment import load_environment_variables, set_logger
import logging


logger = logging.getLogger()


def Gem_single_metric(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    metric: str,
    **kwargs,
) -> Union[Tuple[float, float, float], Tuple[Dict[str, float]]]:
    """Computes the GEM (Generic Evaluation Metric) score for a given hypothesis and references.

    Args:
        hypothesis: A list of strings representing the hypothesis.
        references: A list of strings or a list of lists of strings representing the references.
        metrics_list: str: a string representing a single metric in the Gem benchmark.

    Returns:
        A dictionary of metrics representing precision, recall, and F1 score.

    Raises:
        AssertionError: If the length of the hypothesis and references lists do not match.
        subprocess.CalledProcessError: If the command to compute the metrics fails.
    """
    logger = set_logger(kwargs["log_file_path"])
    logger.info(f"Computing Gem-{metric} score...")

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
    assert len(hypothesis) == len(references_gem), f"{len(hypothesis)} != {len(references_gem)}"
    write_json_file(hyp_path, hypothesis, overwrite=True)
    write_json_file(ref_path, references_gem, overwrite=True)

    # Compute metrics
    out_path = os.path.join(temp_dir, "scores.json")
    GEM_dir = os.path.join(base_dir, "GEM-metrics")
    code = ""
    code += f'cd "{GEM_dir}"' + " "
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

    logger.info(f"Computing Gem-{metric} score... FINISHED!")

    return scores


def Gem(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
) -> Union[Tuple[float, float, float], Tuple[Dict[str, float]]]:
    """Computes the GEM (Generic Evaluation Metric) score for a given hypothesis and references.

    Args:
        hypothesis: A list of strings representing the hypothesis.
        references: A list of strings or a list of lists of strings representing the references.
        **kwargs: Additional keyword arguments.
            metrics_list: A string representing the list of metrics to compute.

    Kwargs:
        metrics_list: str: a string representing the metrics to compute. Metrics are all lowercase and
            separated by a single space.

    Returns:
        A dictionary of metrics representing precision, recall, and F1 score.

    Raises:
        AssertionError: If the length of the hypothesis and references lists do not match.
        subprocess.CalledProcessError: If the command to compute the metrics fails.
    """
    logger.info(f'Computing Gem ({kwargs["metrics_list"]}) metrics...')

    if isinstance(references[0], str):
        references_gem = [[ref] for ref in references]
    else:
        references_gem = references

    load_environment_variables("/etc/environment")

    metrics = kwargs["metrics_list"].split(" ")

    # Compute metrics
    results = {}
    for metric in metrics:
        results[metric] = Gem_single_metric(hypothesis, references_gem, metric, kwargs)

    logger.info(f'Computing Gem ({kwargs["metrics_list"]}) metrics... FINISHED!')

    return results
