from typing import List
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from altilia_text_metrics.metrics.nltk.nltk_utils import nltk_tokenizer
from altilia_text_metrics.utils.environment import set_logger
import nltk
import logging


logger = logging.getLogger()


nltk.download("wordnet")


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
    logger = set_logger(kwargs["log_file_path"])
    logger.info("Computing METEOR score...")

    assert len(hypothesis) == len(references), f"{len(hypothesis)} != {len(references)}"

    # Tokenize the sentences
    hypothesis_tok = [nltk_tokenizer(x) for x in hypothesis]
    references_tok = [[nltk_tokenizer(x) for x in ref_list] for ref_list in references]

    meteor_scores = []
    for _hyp, _ref in zip(hypothesis_tok, references_tok):
        if kwargs["method"] == "max":
            meteor_scores.append(max([single_meteor_score(x, _hyp) for x in _ref]))
        elif kwargs["method"] == "avg":
            temp_scores = [single_meteor_score(x, _hyp) for x in _ref]
            meteor_scores.append(sum(temp_scores) / len(temp_scores))

    logger.info("Computing METEOR score... FINISHED!")

    if kwargs["return_all_scores"]:
        return {"scores": meteor_scores, "meteor": sum(meteor_scores) / len(meteor_scores)}
    else:
        return {"meteor": sum(meteor_scores) / len(meteor_scores)}
