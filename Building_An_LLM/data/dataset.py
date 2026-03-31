#!/usr/bin/env python3
"""
PyTorch Dataset for literature text data.

Provides efficient loading and batching of text data for LLM training.
Supports variable-length sequences with configurable parameters.

Usage:
    from dataset import LiteratureDataset
    dataset = LiteratureDataset('processed/train.txt', seq_length=512)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple


class LiteratureDataset(Dataset):
    """
    Dataset for loading and chunking literature text.

    Args:
        file_path: Path to text file
        seq_length: Length of each sequence chunk
        stride: How many tokens to move between chunks (for overlapping windows)
        tokenizer: Optional tokenizer instance (if None, uses character-level)
    """

    def __init__(
        self,
        file_path: str,
        seq_length: int = 512,
        stride: Optional[int] = None,
        tokenizer=None
    ):
        self.file_path = Path(file_path)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        self.tokenizer = tokenizer

        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load and optionally tokenize text
        self.text = self._load_text()
        self.tokens = self._tokenize(self.text)

        # Calculate number of chunks
        self.num_chunks = max(1, (len(self.tokens) - seq_length) // self.stride + 1)

        print(f"Loaded {file_path}: {len(self.tokens):,} tokens, {self.num_chunks:,} chunks")

    def _load_text(self) -> str:
        """Load text from file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        If no tokenizer provided, uses simple character-level encoding.
        """
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            # Character-level fallback: map chars to ASCII/Unicode ordinals
            # Reserve 0 for padding, 1 for unknown, 2 for EOS
            return [ord(c) + 3 for c in text if ord(c) < 65535]

    def _detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        else:
            # Character-level fallback
            return ''.join(chr(max(0, t - 3)) for t in tokens if t > 2)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Tuple of (input_ids, target_ids) where target is input shifted by 1
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length

        # Get chunk of tokens
        chunk = self.tokens[start_idx:end_idx]

        # Pad if necessary
        if len(chunk) < self.seq_length:
            chunk = chunk + [0] * (self.seq_length - len(chunk))

        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids

    def get_vocab_size(self) -> int:
        """Get vocabulary size (for character-level, this is max Unicode)."""
        if self.tokenizer is not None:
            return self.tokenizer.vocab_size
        else:
            # Character-level: max token value + 1
            return max(self.tokens) + 1 if self.tokens else 256

    def sample_text(self, n_samples: int = 3) -> List[str]:
        """Sample random chunks and return as text."""
        import random
        samples = []
        for _ in range(n_samples):
            idx = random.randint(0, len(self) - 1)
            input_ids, _ = self[idx]
            # Convert input_ids back to text (add offset for character-level)
            if self.tokenizer is None:
                text = ''.join(chr(max(0, t - 3)) for t in input_ids.tolist() if t > 2)
            else:
                text = self.tokenizer.decode(input_ids.tolist())
            samples.append(text)
        return samples


class TokenizedDataset(Dataset):
    """
    Dataset that loads pre-tokenized data (for use after tokenizer is trained).

    Args:
        token_file: Path to file containing token IDs (one per line, space-separated)
        seq_length: Length of each sequence
    """

    def __init__(self, token_file: str, seq_length: int = 512):
        self.seq_length = seq_length
        self.token_file = Path(token_file)

        if not self.token_file.exists():
            raise FileNotFoundError(f"Token file not found: {token_file}")

        # Load tokens efficiently
        self.tokens = self._load_tokens()
        self.num_samples = max(1, len(self.tokens) // seq_length)

        print(f"Loaded {token_file}: {len(self.tokens):,} tokens, {self.num_samples:,} samples")

    def _load_tokens(self) -> List[int]:
        """Load token IDs from file."""
        tokens = []
        with open(self.token_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_tokens = [int(t) for t in line.strip().split() if t]
                tokens.extend(line_tokens)
        return tokens

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input/target pair."""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1

        chunk = self.tokens[start_idx:end_idx]

        # Pad if necessary
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [0] * (self.seq_length + 1 - len(chunk))

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


def create_dataloader(
    data_file: str,
    seq_length: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    tokenizer=None
) -> DataLoader:
    """
    Convenience function to create a DataLoader.

    Args:
        data_file: Path to text file
        seq_length: Sequence length for chunks
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 for main thread)
        tokenizer: Optional tokenizer instance

    Returns:
        DataLoader instance
    """
    dataset = LiteratureDataset(
        file_path=data_file,
        seq_length=seq_length,
        tokenizer=tokenizer
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def main():
    """Test the dataset."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_file> [seq_length]")
        sys.exit(1)

    data_file = sys.argv[1]
    seq_length = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    print("=" * 60)
    print("Testing LiteratureDataset")
    print("=" * 60)

    try:
        dataset = LiteratureDataset(data_file, seq_length=seq_length)

        print(f"\nDataset size: {len(dataset)}")
        print(f"Vocab size: {dataset.get_vocab_size():,}")

        # Sample some text
        print("\nSample chunks:")
        print("-" * 60)
        for i, text in enumerate(dataset.sample_text(3)):
            print(f"\nSample {i+1}:")
            print(text[:200] + "..." if len(text) > 200 else text)
            print()

        # Test dataloader
        print("\nTesting DataLoader:")
        loader = create_dataloader(data_file, seq_length=seq_length, batch_size=2)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print(f"  Batch {batch_idx+1}: inputs shape {inputs.shape}, targets shape {targets.shape}")
            if batch_idx >= 2:  # Just show first 3 batches
                break

        print("\n✅ Dataset test successful!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
