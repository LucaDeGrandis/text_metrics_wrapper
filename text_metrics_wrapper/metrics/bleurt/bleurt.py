from typing import List, Union, Tuple
from tqdm import tqdm
from bleurt import score


def Bleurt(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    checkpoint: str = "/content/bleurt/BLEURT-20",
    method: str = "max",
    return_all_scores: bool = False,
) -> Tuple[float, float, float]:
    scorer = score.BleurtScorer(checkpoint)

    assert method in ["max", "avg"], "method must be either 'max' or 'avg'"

    scores = []
    for hyp, refs in tqdm(zip(hypothesis, references)):
        hyp_scores = []
        for ref in refs:
            hyp_scores.append(scorer.score(references=[ref], candidates=[hyp])[0])
        if method == "max":
            scores.append(max(hyp_scores))
        else:
            scores.append(sum(hyp_scores) / len(hyp_scores))

    if return_all_scores:
        return {"scores": scores, "bleurt": sum(scores) / len(scores)}
    else:
        return {"bleurt": sum(scores) / len(scores)}
