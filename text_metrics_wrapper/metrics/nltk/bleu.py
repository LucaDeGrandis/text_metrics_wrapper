from typing import List, Union
from nltk.translate.bleu_score import corpus_bleu
from text_metrics_wrapper.metrics.nltk.nltk_utils import prepare_weights


def Bleu_nltk(
    hypothesis: List[str],
    references: List[List[str],
    **kwargs,
):
    weights = prepare_weights(kwargs["bleu_n"])
    result = corpus_bleu(references_post, hypothesis)

    return {"bleu": result.score}
