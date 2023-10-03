

class text_block():
    """Represents a block of text.

    Attributes:
        text (str): The text content of the block.
    """

    def __init__(
        self,
        text: str
    ) -> None:
        """Initializes a new instance of the text_block class.

        Args:
            text (str): The text content of the block.
        """
        self.text = text
