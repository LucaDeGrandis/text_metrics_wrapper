from typing import List, Union
from text_metrics_wrapper.tokenizers.tokenizer import tokenizer


def preprocess_text_blocks(text_blocs: Union[List[str], List[List[str]]], tokenizer_name: str, **kwargs: dict) -> None:
    """
    Preprocesses the given text blocks using the specified tokenizer.

    Args:
        text_blocks (List[text_block]): The list of text blocks to preprocess.
        tokenizer_name (str): The name of the tokenizer to use.
        **kwargs (dict): Additional keyword arguments to pass to the tokenizer.

    Returns:
        List[str]: The list of preprocessed text.
    """
    selected_tokenizer = tokenizer(tokenizer_name)

    text_list = []
    for text_block in text_blocs:
        if isinstance(text_block, str):
            text_list.append(text_block.replace("\n", " ").replace("  ", " "))
        else:
            text_list.append(text_block.text.replace("\n", " ").replace("  ", " "))

    preprocessed_text = selected_tokenizer.execute(text, **kwargs)

    return preprocessed_text
