from sacrebleu.metrics import BLEU
from text_metrics_wrapper.metrics.bleu_sacrebleu.sacrebleu_utils import organize_references_in_lists


def Bleu_sacrebleu(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
):
    references_post = organize_references_in_lists(references)
