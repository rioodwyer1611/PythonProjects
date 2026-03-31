#!/usr/bin/env python3
"""
Preprocess and clean public domain literature for LLM training.

This script:
- Removes Project Gutenberg headers/footers
- Normalizes text encoding
- Cleans formatting artifacts
- Filters out non-literary content
- Splits into train/val/test sets

Usage:
    python prepare_data.py --raw-dir ./raw --output-dir ./processed
"""

import os
import re
import argparse
import random
from pathlib import Path
from typing import List, Tuple


# Patterns to detect and remove Project Gutenberg headers/footers
GUTENBERG_HEADER_PATTERNS = [
    r'.*?The Project Gutenberg eBook.*?\*\*\* START OF',  # Old format
    r'.*?\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK',  # Common start
    r'^.*?Project Gutenberg.*?\*\*\*',  # Simplified
]

GUTENBERG_FOOTER_PATTERNS = [
    r'\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?$',  # Common end
    r'End of Project Gutenberg.*?$',  # Alternate
    r'\*\*\* END.*?Project Gutenberg',  # Simplified
]


def remove_gutenberg_headers(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Try to find the actual content between markers
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT!"
    ]

    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THIS PROJECT GUTENBERG EBOOK"
    ]

    # Look for start marker
    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the newline after the marker line
            nl_idx = text.find('\n', idx)
            if nl_idx != -1:
                start_idx = nl_idx + 1
            break

    # Look for end marker
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    # If we found markers, extract content between them
    if start_idx != -1:
        return text[start_idx:end_idx]

    # Fallback: try to remove obvious header/footer
    lines = text.split('\n')
    content_lines = []
    in_content = False

    for line in lines:
        # Skip lines that are clearly header/footer
        if any(marker in line for marker in start_markers):
            in_content = True
            continue
        if any(marker in line for marker in end_markers):
            break
        if in_content:
            content_lines.append(line)

    return '\n'.join(content_lines) if content_lines else text


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove carriage returns
    text = text.replace('\r', '')

    # Normalize whitespace - replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Normalize line breaks - collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just page numbers (standalone numbers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip if line is just a number (likely page number)
        if re.match(r'^\s*\d+\s*$', stripped) and len(stripped) < 5:
            continue
        # Skip if line contains mostly dots (table of contents)
        if stripped.count('.') > len(stripped) * 0.3:
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Remove excessive whitespace at line starts (but preserve some indentation for poetry)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Keep up to 8 spaces of indentation (for poetry formatting)
        # But remove excessive spaces
        stripped = line.lstrip()
        spaces = len(line) - len(stripped)
        if spaces > 8:
            line = ' ' * 8 + stripped
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Remove excessive newlines again
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


def is_literary_content(text: str) -> bool:
    """
    Check if text contains substantial literary content.
    Filter out empty files or files that are just metadata.
    """
    if len(text) < 1000:  # Skip very short files
        return False

    # Check for sufficient alphabetic content
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:  # Less than 70% letters/spaces
        return False

    return True


def process_file(input_path: Path) -> Tuple[str, int]:
    """
    Process a single raw text file.

    Returns:
        Tuple of (cleaned_text, original_size)
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"    Error reading {input_path}: {e}")
        return "", 0

    original_size = len(text)

    # Remove Gutenberg headers/footers
    text = remove_gutenberg_headers(text)

    # Clean text
    text = clean_text(text)

    # Check if content is valid
    if not is_literary_content(text):
        print(f"    ⚠️  Skipping {input_path.name} - insufficient literary content")
        return "", original_size

    return text, original_size


def split_data(documents: List[str], train_ratio=0.9, val_ratio=0.05) -> Tuple[List[str], List[str], List[str]]:
    """
    Split documents into train/val/test sets.
    Splits by document (not by chunks) to avoid leakage.

    Returns:
        Tuple of (train_docs, val_docs, test_docs)
    """
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    shuffled = documents.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_docs = shuffled[:n_train]
    val_docs = shuffled[n_train:n_train + n_val]
    test_docs = shuffled[n_train + n_val:]

    return train_docs, val_docs, test_docs


def prepare_dataset(raw_dir: Path, output_dir: Path) -> dict:
    """
    Process all raw files and create train/val/test splits.

    Returns:
        Dictionary with processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'processed': 0,
        'skipped': 0,
        'train_docs': 0,
        'val_docs': 0,
        'test_docs': 0,
        'train_chars': 0,
        'val_chars': 0,
        'test_chars': 0,
        'total_original_chars': 0,
    }

    # Find all .txt files in raw directory
    raw_files = list(raw_dir.glob('*.txt'))

    if not raw_files:
        print(f"⚠️  No .txt files found in {raw_dir}")
        return stats

    print(f"Processing {len(raw_files)} files from {raw_dir}...")

    documents = []

    for file_path in raw_files:
        print(f"  Processing: {file_path.name}")

        text, original_size = process_file(file_path)
        stats['total_original_chars'] += original_size

        if text:
            documents.append(text)
            stats['processed'] += 1
        else:
            stats['skipped'] += 1

    if not documents:
        print("⚠️  No valid documents to process")
        return stats

    # Split into train/val/test
    train_docs, val_docs, test_docs = split_data(documents)

    stats['train_docs'] = len(train_docs)
    stats['val_docs'] = len(val_docs)
    stats['test_docs'] = len(test_docs)

    # Write splits to files
    def write_docs(docs: List[str], filename: str) -> int:
        """Write documents to file and return total characters."""
        output_path = output_dir / filename
        combined = '\n\n' + '=' * 80 + '\nNEW DOCUMENT\n' + '=' * 80 + '\n\n'
        text = combined.join(docs)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return len(text)

    stats['train_chars'] = write_docs(train_docs, 'train.txt')
    stats['val_chars'] = write_docs(val_docs, 'val.txt')
    stats['test_chars'] = write_docs(test_docs, 'test.txt')

    return stats


def print_statistics(stats: dict):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Files processed: {stats['processed']}")
    print(f"Files skipped: {stats['skipped']}")

    if stats['processed'] == 0:
        print("\n⚠️  No files were processed")
        return

    print(f"\nDataset splits:")
    print(f"  Train: {stats['train_docs']} documents ({stats['train_chars']:,} chars)")
    print(f"  Val:   {stats['val_docs']} documents ({stats['val_chars']:,} chars)")
    print(f"  Test:  {stats['test_docs']} documents ({stats['test_chars']:,} chars)")

    total_chars = stats['train_chars'] + stats['val_chars'] + stats['test_chars']
    print(f"\nTotal: {total_chars:,} characters ({total_chars / 1_000_000:.2f} MB)")
    print(f"Original size: {stats['total_original_chars']:,} characters")
    print(f"Retention: {total_chars / stats['total_original_chars'] * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw literature files for LLM training"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="./raw",
        help="Directory containing raw downloaded books (default: ./raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed",
        help="Directory to save processed files (default: ./processed)"
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"❌ Error: Raw directory {raw_dir} does not exist")
        print("Run download_data.py first to download the books")
        return 1

    print("=" * 60)
    print("Literature Preprocessing Pipeline")
    print("=" * 60)
    print()

    stats = prepare_dataset(raw_dir, output_dir)
    print_statistics(stats)

    if stats['processed'] > 0:
        print(f"\n✅ Preprocessing complete! Files saved to {output_dir}")
        return 0
    else:
        print("\n❌ No files were processed")
        return 1


if __name__ == "__main__":
    exit(main())
