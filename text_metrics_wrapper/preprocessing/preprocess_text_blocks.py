from typing import List, Union
from text_metrics_wrapper.tokenizers.tokenizer import tokenizer


def custom_text_preprocessing(text: str) -> str:
    """
    Custom text preprocessing function.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    return text.replace("\n", " ").replace("  ", " ")


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
    if isinstance(text_blocs[0], list):
        for text_block_list in text_blocs:
            temp_list = []
            for text_block in text_block_list:
                if isinstance(text_block, str):
                    temp_list.append(custom_text_preprocessing(text_block))
                else:
                    temp_list.append(custom_text_preprocessing(text_block.text))
            text_list.append(temp_list)
    else:
        for text_block in text_blocs:
            if isinstance(text_block, str):
                text_list.append(custom_text_preprocessing(text_block))
            else:
                text_list.append(custom_text_preprocessing(text_block.text))

    preprocessed_text = selected_tokenizer.execute(text_list, **kwargs)

    return preprocessed_text
