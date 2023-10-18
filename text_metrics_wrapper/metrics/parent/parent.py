from text_metrics_wrapper.metrics.parent.parent_utilities import (
    _text_reader_reference,
    _text_reader_candidate,
    _table_reader,
    parent,
    overlap_probability,
)
from text_metrics_wrapper.utils.manage_jsonl_files import load_jsonl_file
from typing import List, Union, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Union
import logging
from text_metrics_wrapper.utils.environment import set_logger


logger = logging.getLogger()


def Parent(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
) -> Tuple[float, float, float]:
    """Computes the Parent metric for a given set of hypothesis and references.

    Args:
        hypothesis: A list of strings representing the hypothesis.
        references: A list of strings or a list of lists of strings representing the references.
        **kwargs: Additional keyword arguments.

    Kwargs:
        tables: List[List[Dict[str, Any]]]: A list of dictionaries representing the tables.

    Returns:
        A tuple of floats representing the precision, recall, and F1 score of the Parent metric.
    """
    logger = set_logger(kwargs["log_file_path"])
    logger.info("Computing Parent score...")

    # Compute the parent metric
    Fs = []
    Ps = []
    Rs = []
    for _index, (_des, _hyp, _tab) in enumerate(zip(references, hypothesis, kwargs["tables"])):
        parent_references = _text_reader_reference(_des)
        parent_candidates = _text_reader_candidate([_hyp])
        parent_tables = [_table_reader([_tab])]
        parent_references = list(parent_references)
        parent_candidates = list(parent_candidates)
        parent_tables = list(parent_tables)

        temp_scores = []
        for ref in parent_references:
            temp_scores.append(
                parent(
                    parent_candidates,
                    [ref],
                    parent_tables,
                    lambda_weight=0.5,
                    smoothing=1e-5,
                    entailment_fn=overlap_probability,
                )
            )
        score_index = np.argmax([x[3] for x in temp_scores])

        Fs.append(temp_scores[score_index][2])
        Ps.append(temp_scores[score_index][0])
        Rs.append(temp_scores[score_index][1])

        if _index % 100 == 0:
            log_sentence = f"Parent progress: {_index} / {len(hypothesis)}"
            logger.info(log_sentence)

    logger.info(f"Parent progress: {len(hypothesis)} / {len(hypothesis)}")
    logger.info("Computing Parent score... FINISHED!")

    return {"precision": np.mean(Ps), "recall": np.mean(Rs), "F1score": np.mean(Fs)}
