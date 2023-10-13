from typing import List
from nltk.translate.bleu_score import corpus_bleu
from text_metrics_wrapper.metrics.nltk.nltk_utils import prepare_weights


def Bleu_nltk(
    hypothesis: List[str],
    references: List[List[str]],
    **kwargs,
):
    """Calculates the BLEU score using the NLTK library.

    Args:
        hypothesis: A list of strings representing the predicted text.
        references: A list of lists of strings representing the reference text.
        **kwargs: Additional keyword arguments to pass to the function.

    Kwargs:
        n_weights: int: The n-gram order for BLEU.

    Returns:
        A dictionary containing the BLEU score.
    """
    weights = prepare_weights(kwargs["bleu_n"])
    result = corpus_bleu(references_post, hypothesis)

    return {"bleu": result.score}
