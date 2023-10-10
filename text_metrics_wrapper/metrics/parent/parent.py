from text_metrics_wrapper.metrics.parent.parent_utilities import (
    _text_reader_reference,
    _text_reader_candidate,
    _table_reader,
    parent,
    overlap_probability,
)
from typing import List, Union, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np


def Parent(
    hypothesis: List[str], references: Union[List[str], List[List[str]]], tables: List[List[Dict[str, Any]]]
) -> Tuple[float, float, float]:
    print("Computing Parent metric...")

    # Compute the parent metric
    Fs = []
    Ps = []
    Rs = []
    for _des, _hyp, _tab in tqdm(zip(references, hypothesis, tables)):
        parent_references = _text_reader_reference(_des)
        parent_candidates = _text_reader_candidate(_hyp)
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

    return np.mean(Ps), np.mean(Rs), np.mean(Fs)
