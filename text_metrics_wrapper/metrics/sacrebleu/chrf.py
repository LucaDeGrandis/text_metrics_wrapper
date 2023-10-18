from typing import List, Union
from sacrebleu.metrics import CHRF
from text_metrics_wrapper.metrics.sacrebleu.sacrebleu_utils import organize_references_in_lists
from text_metrics_wrapper.utils.environment import set_logger
import logging


logger = logging.getLogger()


chrf = CHRF()


def Chrf_sacrebleu(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
) -> dict:
    """
    Calculates the CHRF score using the SacreBLEU library.

    Args:
        hypothesis: A list of strings representing the predicted sentences.
        references: A list of strings or a list of lists of strings representing the reference sentences.
        **kwargs: Additional arguments to be passed to the SacreBLEU library.

    Returns:
        A dictionary containing the CHRF score.

    Raises:
        TypeError: If the input arguments are not of the expected type.
    """
    logger = set_logger(kwargs["log_file_path"])
    logger.info("Computing CHRF score...")

    references_post = organize_references_in_lists(references)
    result = chrf.corpus_score(hypothesis, references_post)

    logger.info("Computing CHRF score... FINISHED!")

    return {"chrf": result.score}
