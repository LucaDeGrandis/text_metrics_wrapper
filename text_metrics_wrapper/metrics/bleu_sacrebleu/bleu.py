from sacrebleu.metrics import BLEU


def Bleu_sacrebleu(
    hypothesis: List[str],
    references: Union[List[str], List[List[str]]],
    **kwargs,
):
    desc_list = [[]]
    hypo_list = []
