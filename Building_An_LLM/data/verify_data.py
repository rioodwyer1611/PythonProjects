#!/usr/bin/env python3
"""
Verify the prepared literature dataset.

Checks:
- Data files exist and are non-empty
- Train/val/test splits are valid
- Statistics on size and content
- Sample random snippets for quality check

Usage:
    python verify_data.py --data-dir ./processed
"""

import os
import random
import argparse
from pathlib import Path
from collections import Counter


def count_tokens_simple(text: str) -> int:
    """Simple token count (whitespace split)."""
    return len(text.split())


def analyze_file(file_path: Path) -> dict:
    """Analyze a single data file."""
    stats = {
        'exists': False,
        'size_bytes': 0,
        'num_chars': 0,
        'num_lines': 0,
        'num_documents': 0,
        'estimated_tokens': 0,
        'sample': ''
    }

    if not file_path.exists():
        return stats

    stats['exists'] = True
    stats['size_bytes'] = file_path.stat().st_size

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    stats['num_chars'] = len(content)
    stats['num_lines'] = content.count('\n') + 1
    stats['estimated_tokens'] = count_tokens_simple(content)

    # Count documents (separated by NEW DOCUMENT marker)
    stats['num_documents'] = content.count('=' * 80 + '\nNEW DOCUMENT') + 1

    # Extract random sample
    if len(content) > 500:
        start = random.randint(0, len(content) - 500)
        stats['sample'] = content[start:start + 500]
    else:
        stats['sample'] = content

    return stats


def verify_dataset(data_dir: Path) -> bool:
    """Verify the complete dataset."""
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")

    files_to_check = ['train.txt', 'val.txt', 'test.txt']
    all_ok = True
    total_stats = {
        'total_chars': 0,
        'total_docs': 0,
        'total_tokens': 0
    }

    for filename in files_to_check:
        file_path = data_dir / filename
        print(f"\n📄 {filename}")
        print("-" * 40)

        stats = analyze_file(file_path)

        if not stats['exists']:
            print(f"  ❌ File not found")
            all_ok = False
            continue

        if stats['num_chars'] == 0:
            print(f"  ❌ File is empty")
            all_ok = False
            continue

        print(f"  ✅ Exists: {stats['size_bytes']:,} bytes")
        print(f"  📊 Characters: {stats['num_chars']:,}")
        print(f"  📊 Lines: {stats['num_lines']:,}")
        print(f"  📊 Documents: {stats['num_documents']:,}")
        print(f"  📊 Est. tokens: {stats['estimated_tokens']:,}")

        total_stats['total_chars'] += stats['num_chars']
        total_stats['total_docs'] += stats['num_documents']
        total_stats['total_tokens'] += stats['estimated_tokens']

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total characters: {total_stats['total_chars']:,}")
    print(f"Total documents: {total_stats['total_docs']:,}")
    print(f"Estimated tokens: {total_stats['total_tokens']:,}")
    print(f"Size: {total_stats['total_chars'] / (1024*1024):.2f} MB")

    if total_stats['total_chars'] < 1_000_000:
        print("\n⚠️  WARNING: Dataset is very small (< 1MB)")
        print("   Consider downloading more books.")

    return all_ok


def show_samples(data_dir: Path, n_samples: int = 3):
    """Show random samples from the training data."""
    train_file = data_dir / 'train.txt'
    if not train_file.exists():
        print("\n⚠️  Cannot show samples - train.txt not found")
        return

    print("\n" + "=" * 60)
    print(f"Random Samples from Training Data")
    print("=" * 60)

    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into documents
    docs = content.split('=' * 80 + '\nNEW DOCUMENT\n' + '=' * 80)
    docs = [d.strip() for d in docs if d.strip()]

    if not docs:
        print("No documents found")
        return

    sampled = random.sample(docs, min(n_samples, len(docs)))

    for i, doc in enumerate(sampled, 1):
        print(f"\n{'─' * 60}")
        print(f"Sample {i}/{len(sampled)} ({len(doc):,} chars)")
        print('─' * 60)
        # Show first 800 chars
        preview = doc[:800]
        print(preview)
        if len(doc) > 800:
            print("\n... [truncated]")


def main():
    parser = argparse.ArgumentParser(
        description="Verify the prepared literature dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./processed",
        help="Directory containing processed data (default: ./processed)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of random samples to show (default: 3)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Set random seed for reproducible samples
    random.seed(42)

    # Verify dataset
    success = verify_dataset(data_dir)

    # Show samples
    show_samples(data_dir, args.samples)

    print("\n" + "=" * 60)
    if success:
        print("✅ Verification passed!")
        return 0
    else:
        print("❌ Verification failed - check errors above")
        return 1


if __name__ == "__main__":
    exit(main())
