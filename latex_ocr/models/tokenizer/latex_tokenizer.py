from pathlib import Path
import sentencepiece as spm
from typing import List, Optional
import json

import click

padding_idx = 3


def load_user_defined_symbols(f: Path, top_k: int = 300) -> list[str]:
    if not f.exists():
        raise FileNotFoundError(f"User defined symbols file not found: {f}")
    l = json.loads(f.read_text(encoding="utf-8"))
    if len(l) > top_k:
        l = l[:top_k]
    return l


class Tokenizer:
    def __init__(
        self,
    ):
        """Initialize the Tokenizer with no loaded SentencePiece model."""
        self.sp_ctrl: Optional[spm.SentencePieceProcessor] = None

    def train(
        self,
        corpus_file: str,
        vocab_size: int = 5000,
        user_defined_symbols_limit: int = 300,
    ):
        """Train a SentencePiece model on the given corpus.

        Args:
            corpus_file: Path to the corpus file(s), comma-separated if multiple.
            model_prefix: Prefix for the model files.
            vocab_size: Size of the vocabulary.
        """
        user_defined_symbols = load_user_defined_symbols(
            Path(__file__).parent.parent.parent
            / "datasets"
            / "user_defined_symbols.json",
            top_k=user_defined_symbols_limit,
        )
        spm.SentencePieceTrainer.Train(
            input=corpus_file,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_prefix="latex_tokenizer",
            # normalization_rule_name="nmt_nfkc_cf",
            user_defined_symbols=user_defined_symbols,
            pad_id=padding_idx,
            pad_piece="<pad>",
            model_type="unigram",
        )

    def load(self, model_path: Path | None = None):
        """Load a SentencePiece model from the given path.

        Args:
            model_path: Path to the model file, default to latex_tokenizer.model under the same dir of this script.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if not model_path:
            model_path = Path(__file__).parent / "latex_tokenizer.model"
        print(f"Loading SentencePiece model from {model_path}...")
        self.sp_ctrl = spm.SentencePieceProcessor()
        ok = self.sp_ctrl.Load(model_path.absolute().__str__())
        if not ok:
            raise RuntimeError(f"Failed to load SentencePiece model from {model_path}")

    def _ensure_loaded(self):
        """Ensure that the SentencePiece model is loaded.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.sp_ctrl is None:
            raise RuntimeError(
                "SentencePiece tokenizer is not loaded. Call `load` or `train` first."
            )

    # HuggingFace-compatible properties
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size.

        Returns:
            The vocabulary size.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.vocab_size()

    @property
    def bos_token_id(self) -> int:
        """Get the BOS (beginning of sentence) token ID.

        Returns:
            The BOS token ID.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.bos_id()

    @property
    def eos_token_id(self) -> int:
        """Get the EOS (end of sentence) token ID.

        Returns:
            The EOS token ID.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.eos_id()

    @property
    def pad_token_id(self) -> int:
        """Get the PAD (padding) token ID.

        Returns:
            The PAD token ID.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.pad_id()

    @property
    def unk_token_id(self) -> int:
        """Get the UNK (unknown) token ID.

        Returns:
            The UNK token ID.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.unk_id()

    @property
    def bos_token(self) -> str:
        """Get the BOS (beginning of sentence) token string.

        Returns:
            The BOS token string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.IdToPiece(self.bos_token_id)

    @property
    def eos_token(self) -> str:
        """Get the EOS (end of sentence) token string.

        Returns:
            The EOS token string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.IdToPiece(self.eos_token_id)

    @property
    def pad_token(self) -> str:
        """Get the PAD (padding) token string.

        Returns:
            The PAD token string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.IdToPiece(self.pad_token_id)

    @property
    def unk_token(self) -> str:
        """Get the UNK (unknown) token string.

        Returns:
            The UNK token string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return self.sp_ctrl.IdToPiece(self.unk_token_id)

    # HuggingFace-compatible encode/decode methods
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs (HuggingFace-compatible).

        Args:
            text: The input text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        text = text.strip()
        ids = self.sp_ctrl.EncodeAsIds(text)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text (HuggingFace-compatible).

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to remove special tokens (BOS, EOS, PAD, UNK).

        Returns:
            Decoded text string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"

        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.unk_token_id,
            }
            token_ids = [id for id in token_ids if id not in special_ids]

        return self.sp_ctrl.DecodeIds(token_ids)

    def batch_encode(
        self, texts: List[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        """Encode a batch of texts to token IDs (HuggingFace-compatible).

        Args:
            texts: List of input texts to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of lists of token IDs.
        """
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

    def batch_decode(
        self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token ID sequences to texts (HuggingFace-compatible).

        Args:
            batch_token_ids: List of token ID sequences to decode.
            skip_special_tokens: Whether to remove special tokens.

        Returns:
            List of decoded text strings.
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs (HuggingFace-compatible).

        Args:
            tokens: List of token strings.

        Returns:
            List of token IDs.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return [self.sp_ctrl.PieceToId(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens (HuggingFace-compatible).

        Args:
            ids: List of token IDs.

        Returns:
            List of token strings.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        return [self.sp_ctrl.IdToPiece(id) for id in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to a string (HuggingFace-compatible).

        Args:
            tokens: List of token strings.

        Returns:
            Concatenated string.
        """
        self._ensure_loaded()
        assert self.sp_ctrl is not None, "SentencePiece processor not loaded"
        # Convert tokens to IDs and then decode
        ids = self.convert_tokens_to_ids(tokens)
        return self.sp_ctrl.DecodeIds(ids)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--corpus-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the corpus file for training the tokenizer.",
)
@click.option(
    "--vocab-size",
    type=int,
    default=5000,
    show_default=True,
    help="Size of the vocabulary.",
)
@click.option(
    "--user-defined-symbols-limit",
    type=int,
    default=300,
    show_default=True,
    help="Maximum number of user-defined symbols to include from user_defined_symbols.json.",
)
def train_cli(
    corpus_file: Path,
    vocab_size: int,
    user_defined_symbols_limit: int,
):
    r"""
    Train a SentencePiece tokenizer on a LaTeX formula corpus.
    
    The trained model will be saved as 'latex_tokenizer.model' and 'latex_tokenizer.vocab'
    in the current working directory.
    
    Example usage:
    
        PYTHONPATH=. uv run python latex_ocr/models/tokenizer/latex_tokenizer.py \
            --corpus-file latex_ocr/datasets/formula_corpus.txt \
            --vocab-size 1122 \
            --user-defined-symbols-limit 300
    """
    click.echo(f"Training tokenizer with:")
    click.echo(f"  Corpus file: {corpus_file}")
    click.echo(f"  Vocab size: {vocab_size}")
    click.echo(f"  User-defined symbols limit: {user_defined_symbols_limit}")

    tokenizer = Tokenizer()
    tokenizer.train(
        corpus_file=str(corpus_file),
        vocab_size=vocab_size,
        user_defined_symbols_limit=user_defined_symbols_limit,
    )

    click.echo("✅ Training completed successfully!")
    click.echo(f"📦 Model files saved:")
    click.echo(f"  - latex_tokenizer.model")
    click.echo(f"  - latex_tokenizer.vocab")


@cli.command()
def demo():
    """
    Demo tokenizer functionality with HuggingFace-compatible properties and methods.

    PYTHONPATH=. uv run python latex_ocr/models/tokenizer/latex_tokenizer.py demo
    """
    tokenizer = Tokenizer()
    tokenizer.load(None)
    sample_text = r"\frac{a}{b} + \sqrt{c}"

    print("=" * 70)
    print("Tokenizer Demo - HuggingFace-Compatible API")
    print("=" * 70)

    # HuggingFace-style encoding/decoding
    print("\n" + "=" * 70)
    print("HuggingFace-compatible Encode/Decode Methods")
    print("=" * 70)
    print(f"Sample text: {sample_text}")

    # Encode without special tokens
    encoded_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"\nencode(text, add_special_tokens=False):")
    print(f"  IDs: {encoded_ids}")

    # Encode with special tokens
    encoded_ids_with_special = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"\nencode(text, add_special_tokens=True):")
    print(f"  IDs: {encoded_ids_with_special}")

    # Decode
    decoded_text = tokenizer.decode(encoded_ids_with_special, skip_special_tokens=True)
    print(f"\ndecode(ids, skip_special_tokens=True):")
    print(f"  Text: {decoded_text}")

    decoded_text_with_special = tokenizer.decode(
        encoded_ids_with_special, skip_special_tokens=False
    )
    print(f"\ndecode(ids, skip_special_tokens=False):")
    print(f"  Text: {decoded_text_with_special}")

    # Batch encoding/decoding
    print("\n" + "=" * 70)
    print("Batch Encode/Decode")
    print("=" * 70)
    batch_texts = [r"\frac{a}{b}", r"\sqrt{x}", r"y = mx + b"]
    print(f"Batch texts: {batch_texts}")

    batch_encoded = tokenizer.batch_encode(batch_texts, add_special_tokens=False)
    print(f"\nbatch_encode(texts):")
    for i, ids in enumerate(batch_encoded):
        print(f"  [{i}] {ids}")

    batch_decoded = tokenizer.batch_decode(batch_encoded, skip_special_tokens=True)
    print(f"\nbatch_decode(ids):")
    for i, text in enumerate(batch_decoded):
        print(f"  [{i}] {text}")

    # Token conversion
    print("\n" + "=" * 70)
    print("Token/ID Conversion")
    print("=" * 70)
    tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
    print(f"convert_ids_to_tokens({encoded_ids}):")
    print(f"  Tokens: {tokens}")

    ids_from_tokens = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nconvert_tokens_to_ids({tokens}):")
    print(f"  IDs: {ids_from_tokens}")

    text_from_tokens = tokenizer.convert_tokens_to_string(tokens)
    print(f"\nconvert_tokens_to_string({tokens}):")
    print(f"  Text: {text_from_tokens}")

    print("\n" + "=" * 70)
    print("HuggingFace-compatible Properties")
    print("=" * 70)
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"unk_token_id: {tokenizer.unk_token_id}")
    print(f"bos_token: '{tokenizer.bos_token}'")
    print(f"eos_token: '{tokenizer.eos_token}'")
    print(f"pad_token: '{tokenizer.pad_token}'")
    print(f"unk_token: '{tokenizer.unk_token}'")


if __name__ == "__main__":
    cli()
