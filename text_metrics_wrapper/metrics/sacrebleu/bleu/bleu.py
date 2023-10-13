from typing import List, Union
from sacrebleu.metrics import BLEU
from text_metrics_wrapper.metrics.bleu_sacrebleu.sacrebleu_utils import organize_references_in_lists

bleu = BLEU(tokenize="none")


def Bleu_sacrebleu(hypothesis: List[str], references: Union[List[str], List[List[str]]], **kwargs) -> dict:
    """
    Calculates the BLEU score using the SacreBLEU library.

    Args:
        hypothesis: A list of strings representing the predicted sentences.
        references: A list of strings or a list of lists of strings representing the reference sentences.
        **kwargs: Additional arguments to be passed to the SacreBLEU library.

    Returns:
        A dictionary containing the BLEU score.

    Raises:
        TypeError: If the input arguments are not of the expected type.

    Example:
        >>> hypothesis = ["The cat is on the mat", "There is a cat on the mat"]
        >>> references = [["The cat is on the mat", "The cat is sleeping on the mat"], ["There is a cat on the mat"]]
        >>> Bleu_sacrebleu(hypothesis, references)
        {'bleu': 0.8408964276313782}
    """
    references_post = organize_references_in_lists(references)
    result = bleu.corpus_score(hypothesis, references_post)

    return {"bleu": result.score}
