from typing import List
from nltk.translate.meteor_score import meteor_score
from text_metrics_wrapper.metrics.nltk.nltk_utils import nltk_tokenizer


def Meteor_nltk(
    hypothesis: List[str],
    references: List[List[str]],
    **kwargs,
):
    """Calculates the Meteor score using the NLTK library.

    Args:
        hypothesis: A list of strings representing the predicted text.
        references: A list of lists of strings representing the reference text.
        **kwargs: Additional keyword arguments to pass to the function.

    Kwrgs:
        return_all_scores: bool: whether to return all scores or just the average.
            True for all scores + average, False for average only.

    Returns:
        A dictionary containing the Meteor score.
    """
    assert len(hypothesis) == len(references), f"{len(hypothesis)} != {len(references)}"

    # Tokenize the sentences
    hypothesis_tok = [nltk_tokenizer(x) for x in hypothesis]
    references_tok = [[nltk_tokenizer(x) for x in ref_list] for ref_list in references]

    meteor_scores = []
    for _hyp, _ref in zip(hypothesis_tok, references_tok):
        meteor_scores.append(meteor_score([_ref], _hyp))
    result = meteor_score(references, hypothesis)

    if kwargs["return_all_scores"]:
        return {"scores": meteor_scores, "meteor": sum(meteor_scores) / len(meteor_scores)}
    else:
        return {"meteor": sum(meteor_scores) / len(meteor_scores)}
