from typing import List, Union
from sacrebleu.metrics import BLEU
from text_metrics_wrapper.metrics.sacrebleu.sacrebleu_utils import organize_references_in_lists
import logging


logger = logging.getLogger()


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
    """
    logger.info("Computing BLEU score...")

    references_post = organize_references_in_lists(references)
    result = bleu.corpus_score(hypothesis, references_post)

    logger.info("Computing BLEU score... FINISHED!")

    return {"bleu": result.score}
