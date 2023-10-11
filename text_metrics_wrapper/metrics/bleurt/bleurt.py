from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from bleurt import score


from typing import List, Tuple, Union
from bleurt import score
from tqdm import tqdm


def Bleurt(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
) -> Union[Tuple[float, float, float], Tuple[Dict[str, float]]]:
    """Computes the BLEURT score for a given hypothesis and reference(s).

    Args:
        hypothesis: A list of strings representing the hypothesis text.
        references: A list of strings or a list of list of strings representing the reference text(s).

    Kwargs:
        checkpoint: str = "/content/bleurt/BLEURT-20": A string representing the path to the BLEURT checkpoint.
        method: str = "max": A string representing the aggregation method to use. Must be either 'max' or 'avg'.
        return_all_scores: bool = False: A boolean indicating whether to return all scores or just the average.

    Returns:
        A tuple containing the BLEURT score(s).

    Raises:
        AssertionError: If the method is not 'max' or 'avg'.

    """
    scorer = score.BleurtScorer(kwargs["checkpoint"])

    assert kwargs["method"] in ["max", "avg"], "method must be either 'max' or 'avg'"

    scores = []
    for hyp, refs in tqdm(zip(hypothesis, references)):
        hyp_scores = []
        for ref in refs:
            hyp_scores.append(scorer.score(references=[ref], candidates=[hyp])[0])
        if kwargs["method"] == "max":
            scores.append(max(hyp_scores))
        else:
            scores.append(sum(hyp_scores) / len(hyp_scores))

    if kwargs["return_all_scores"]:
        return {"scores": scores, "bleurt": sum(scores) / len(scores)}
    else:
        return {"bleurt": sum(scores) / len(scores)}
