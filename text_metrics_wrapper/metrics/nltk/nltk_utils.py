from typing import List, Tuple


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
