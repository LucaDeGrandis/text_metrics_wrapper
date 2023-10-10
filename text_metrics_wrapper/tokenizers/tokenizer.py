from typing import Optional, List
from text_metrics_wrapper.tokenizers.identity_tokenizer import identity_tokenizer
from text_metrics_wrapper.tokenizers.tokenizer_13a import tokenizer_13a
from text_metrics_wrapper.blocks.text_block import text_block


tokenizer_map = {"identity": identity_tokenizer, "tokenizer_13a": tokenizer_13a}


class tokenizer:
    def __init__(self, tokenizer_name: Optional[str]) -> None:
        """
        Initializes a Tokenizer object.

        Args:
            tokenizer_name (Optional[str]): The name of the tokenizer to use. Defaults to 'identity' if None.
        """
        self.tokenizer_name = "identity" if tokenizer_name is None else tokenizer_name
        self.tokenizer = self.create_tokenizer()

    def create_tokenizer(self) -> None:
        """
        Creates a tokenizer based on the specified tokenizer name.
        """
        self.tokenizer = tokenizer_map[self.tokenizer_name]

    def execute(self, text_blocks: List[text_block], **kwargs: dict) -> List[str]:
        """
        Tokenizes the given text blocks using the specified tokenizer.

        Args:
            text_blocks (List[text_block]): The list of text blocks to tokenize.

        Returns:
            List[str]: The list of tokenized sentences.
        """
        tokenized_sentences = []
        for block in text_blocks:
            tokenized_sentences.append(self.tokenizer(block.text))

        return tokenized_sentences
