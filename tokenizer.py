"""Byte-level tokenizer for MicroMixer-1 language model."""


class ByteTokenizer:
    """Byte-level tokenizer with 256 vocabulary size.

    Reserves byte values 0, 1, 2 for special tokens:
    - 0: pad_token
    - 1: bos_token (beginning of sequence)
    - 2: eos_token (end of sequence)

    All other byte values (3-255) represent raw UTF-8 bytes.
    """

    def __init__(self):
        self.vocab_size = 256
        # Special tokens (using byte values that are rare in text)
        self.pad_token_id = 0
        self.bos_token_id = 1  # Beginning of sequence
        self.eos_token_id = 2  # End of sequence

    def encode(self, text: str) -> list[int]:
        """Encode string to list of byte IDs.

        Args:
            text: Input string to encode

        Returns:
            List of byte IDs with BOS token at start and EOS token at end
        """
        if not text:
            # Empty string: just return BOS + EOS
            return [self.bos_token_id, self.eos_token_id]

        # Convert text to UTF-8 bytes
        text_bytes = text.encode("utf-8")

        # Map each byte to its integer value (0-255)
        byte_ids = [b for b in text_bytes]

        # Add BOS at start, EOS at end
        return [self.bos_token_id] + byte_ids + [self.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        """Decode list of byte IDs back to string.

        Args:
            ids: List of byte IDs

        Returns:
            Decoded string
        """
        if not ids:
            return ""

        # Filter out special tokens (pad, bos, eos)
        byte_values = [
            b for b in ids
            if b not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]

        if not byte_values:
            return ""

        # Convert byte values back to bytes object
        byte_data = bytes(byte_values)

        # Decode UTF-8 bytes to string, handle errors gracefully
        return byte_data.decode("utf-8", errors="replace")

    def encode_batch(
        self, texts: list[str], max_length: int = None, padding: bool = True
    ) -> dict:
        """Encode a batch of texts with optional padding.

        Args:
            texts: List of strings to encode
            max_length: Maximum sequence length (truncation if specified)
            padding: Whether to pad sequences to longest in batch

        Returns:
            Dict with 'input_ids' (list of lists) and 'attention_mask' (list of lists)
        """
        # Encode each text
        encoded = [self.encode(text) for text in texts]

        # Get sequence lengths before padding/truncation
        lengths = [len(ids) for ids in encoded]

        # Truncate if max_length specified
        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        # Pad to longest sequence if padding=True
        if padding and encoded:
            max_seq_len = max(len(ids) for ids in encoded)
            pad_token_id = self.pad_token_id

            padded = []
            attention_masks = []
            for ids in encoded:
                pad_len = max_seq_len - len(ids)
                padded.append(ids + [pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            return {
                "input_ids": padded,
                "attention_mask": attention_masks,
            }

        # No padding
        attention_masks = [[1] * len(ids) for ids in encoded]
        return {
            "input_ids": encoded,
            "attention_mask": attention_masks,
        }