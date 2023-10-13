import sacrebleu


def compute_bleu(reference, hypothesis):
    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    return bleu.score


# Example usage
reference = ["The cat is on the mat."]
hypothesis = ["The cat is on the mat.", "It's sleeping."]
bleu_score = compute_bleu(reference, hypothesis)
print("BLEU score:", bleu_score)
