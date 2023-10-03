from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
tokenizer = Tokenizer13a()


def tokenizer_13a(
    text: str
) -> str:
    """Returns the input text as is.

    This function is a simple identity tokenizer that returns the input text
    without any tokenization or processing.

    Args:
        text (str): The input text to tokenize.

    Returns:
        str: The input text as is.
    """
    return Tokenizer13a(text)
