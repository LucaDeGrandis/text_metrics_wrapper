from typing import List
from nltk.translate.bleu_score import corpus_bleu
from text_metrics_wrapper.metrics.nltk.nltk_utils import prepare_weights
from text_metrics_wrapper.metrics.nltk.nltk_utils import nltk_tokenizer


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
    assert len(hypothesis) == len(references), f"{len(hypothesis)} != {len(references)}"
    weights = prepare_weights(kwargs["bleu_n"])
    hypothesis_tok = [nltk_tokenizer(x) for x in hypothesis]
    references_tok = [[nltk_tokenizer(x) for x in ref_list] for ref_list in references]
    result = corpus_bleu(references_tok, hypothesis_tok)

    return {"bleu": result}
