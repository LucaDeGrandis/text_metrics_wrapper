from typing import List, Tuple, Any, Dict, Union
from nltk.translate.meteor_score import _generate_enums, _enum_allign_words, _count_chunks
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet


def prepare_weights(
    n_weights: int = 4,
) -> List[Tuple[float, ...]]:
    """Prepares a list of tuples representing weights for a moving average.

    Args:
        n_weights: An integer representing the number of weights to prepare.
            Defaults to 4.

    Returns:
        A list of tuples, where each tuple represents a set of weights for a
        moving average. Each tuple contains n elements, where the i-th element
        represents the weight for the i-th most recent value in the moving
        average. The sum of the elements in each tuple is always 1.

    Raises:
        ValueError: If n_weights is not a positive integer.
    """
    if not isinstance(n_weights, int) or n_weights <= 0:
        raise ValueError("n_weights must be a positive integer")
    weights = []
    for n_weight in range(n_weights):
        denom = n_weight + 1
        weights.append(tuple([1 / denom for _ in range(denom)]))
    return weights


def nltk_tokenizer(
    sentence: str,
) -> List[str]:
    """
    Tokenizes a sentence for usage with the nltk library.

    Args:
        sentence (str): The sentence to tokenize.

    Returns:
        list: A list of tokens.
    """
    return sentence.split()


def single_meteor_score(
    reference: str,
    hypothesis: str,
    preprocess: callable = str.lower,
    stemmer: Any = PorterStemmer(),
    wordnet: Any = wordnet,
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> Tuple[float, float, float, float, float, float]:
    """
    Computes the Single METEOR score between a reference and a hypothesis sentence.

    Args:
        reference: A string representing the reference sentence.
        hypothesis: A string representing the hypothesis sentence.
        preprocess: A function that takes a string and returns a preprocessed string.
        stemmer: A stemmer object from the nltk library.
        wordnet: A wordnet object from the nltk library.
        alpha: A float representing the precision-recall balance.
        beta: A float representing the chunk penalty factor.
        gamma: A float representing the fragmentation penalty factor.

    Returns:
        A tuple containing the precision, recall, fmean, chunk count, fragment count, and penalty.
    """
    enum_hypothesis, enum_reference = _generate_enums(hypothesis, reference, preprocess=preprocess)
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference)
    matches_count = len(matches)

    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
        penalty = gamma * frag_frac**beta
    except ZeroDivisionError:
        precision = 0
        recall = 0
        fmean = 0
        chunk_count = 0
        frag_frac = 0
        penalty = 0

    return precision, recall, fmean, chunk_count, frag_frac, penalty


from typing import List
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet


def corpus_meteor(
    references: List[List[str]],
    hypothesis: List[str],
    preprocess: callable = str.lower,
    stemmer: Any = PorterStemmer(),
    wordnet: Any = wordnet,
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> float:
    """
    Computes the METEOR score for a corpus of references and a hypothesis.

    Args:
        references: A list of lists of strings, where each inner list contains the reference sentences.
        hypothesis: A list of strings, where each string is the hypothesis sentence.
        preprocess: A function to preprocess the sentences before computing the score.
        stemmer: A stemmer object from the NLTK library.
        wordnet: A WordNet object from the NLTK library.
        alpha: The weight given to precision in the F-measure calculation.
        beta: The weight given to recall in the F-measure calculation.
        gamma: The penalty factor for unaligned words.

    Returns:
        The METEOR score for the corpus of references and hypothesis.
    """
    # Obtain all metrics for each reference
    all_precisions = []
    all_recalls = []
    all_penalties = []
    for _ref_list, _hyp in zip(references, hypothesis):
        temp_fs = []
        temp_scores = []
        for _ref in _ref_list:
            precision, recall, fmean, chunk_count, frag_frac, penalty = single_meteor_score(
                _ref,
                _hyp,
                stemmer=stemmer,
                wordnet=wordnet,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            temp_fs.append(fmean)
            temp_scores.append([precision, recall, fmean, chunk_count, frag_frac, penalty])
        precision, recall, fmean, chunk_count, frag_frac, penalty = temp_scores[np.argmax(temp_fs)]
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_penalties.append(penalty)

    # Compute corpus scores
    corpus_precision = np.mean(all_precisions)
    corpus_recall = np.mean(all_recalls)
    corpus_penalty = np.mean(all_penalties)
    corpus_fmean = (corpus_precision * corpus_recall) / (alpha * corpus_precision + (1 - alpha) * corpus_recall)

    # Compute corpus meteor
    corpus_meteor = corpus_fmean * (1 - corpus_penalty)

    return corpus_meteor
