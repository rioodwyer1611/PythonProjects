#!/usr/bin/env python3
"""
Download public domain literature from Project Gutenberg for LLM training.

Usage:
    python download_data.py [--output-dir ./raw]

This script downloads classic literature, poetry, and creative writing
including works by Shakespeare, Poe, Dickinson, and other public domain authors.
"""

import os
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Project Gutenberg book IDs for classic literature and poetry
# Format: (gutenberg_id, author, title, genre)
BOOKS = [
    # Shakespeare
    (100, "Shakespeare", "The Complete Works of William Shakespeare", "plays_poetry"),
    (1041, "Shakespeare", "Shakespeare's Sonnets", "poetry"),
    (1513, "Shakespeare", "Romeo and Juliet", "plays"),
    (1533, "Shakespeare", "Macbeth", "plays"),
    (1524, "Shakespeare", "Hamlet", "plays"),

    # Edgar Allan Poe
    (1062, "Poe", "The Works of Edgar Allan Poe", "poetry_prose"),
    (10031, "Poe", "The Raven", "poetry"),
    (17192, "Poe", "The Tell-Tale Heart", "short_stories"),

    # Emily Dickinson
    (12242, "Dickinson", "Poems by Emily Dickinson", "poetry"),
    (2678, "Dickinson", "Poems: Second Series", "poetry"),

    # Walt Whitman
    (1322, "Whitman", "Leaves of Grass", "poetry"),

    # Robert Frost (selected public domain)
    (59812, "Frost", "North of Boston", "poetry"),

    # William Wordsworth
    (8918, "Wordsworth", "Lyrical Ballads", "poetry"),

    # Classic Novels
    (1342, "Austen", "Pride and Prejudice", "novel"),
    (84, "Shelley", "Frankenstein", "novel"),
    (11, "Carroll", "Alice's Adventures in Wonderland", "novel"),
    (174, "Dickens", "The Chimes", "short_stories"),
    (46, "Dickens", "A Christmas Carol", "short_stories"),
    (766, "Dickens", "David Copperfield", "novel"),

    # Short Stories
    (2038, "Chekhov", "The Lady with the Dog", "short_stories"),
    (2542, "Kafka", "The Metamorphosis", "short_stories"),

    # Poetry Collections
    (23662, "Yeats", "The Collected Poems of W.B. Yeats", "poetry"),
    (16786, "Keats", "The Poems of John Keats", "poetry"),
    (50852, "Blake", "Songs of Innocence and Experience", "poetry"),

    # Fairy Tales and Fantasy
    (2591, "Grimm", "Grimm's Fairy Tales", "fairy_tales"),
    (7439, "Baum", "The Wonderful Wizard of Oz", "fantasy"),
    (236, "Barrie", "Peter Pan", "fantasy"),

    # Essays and Creative Non-fiction
    (910, "Emerson", "Essays", "essays"),
    (2943, "Thoreau", "Walden", "essays"),
]


def download_gutenberg_text(book_id: int, output_path: Path, retries: int = 3) -> bool:
    """
    Download a text from Project Gutenberg by book ID.

    Args:
        book_id: Project Gutenberg book ID
        output_path: Where to save the file
        retries: Number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    for attempt in range(retries):
        try:
            # Add delay to be respectful to Project Gutenberg servers
            time.sleep(1)

            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'LLM-Training-Project/1.0 (educational use)'
                }
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8', errors='ignore')

                # Write to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Try alternative URL format
                alt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                try:
                    time.sleep(1)
                    req = urllib.request.Request(
                        alt_url,
                        headers={
                            'User-Agent': 'LLM-Training-Project/1.0 (educational use)'
                        }
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        content = response.read().decode('utf-8', errors='ignore')
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        return True
                except:
                    pass

            print(f"  Attempt {attempt + 1} failed: HTTP {e.code}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return False


def download_all_books(output_dir: Path) -> dict:
    """
    Download all books in the BOOKS list.

    Args:
        output_dir: Directory to save downloaded books

    Returns:
        Dictionary with download statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'successful': 0,
        'failed': 0,
        'by_genre': {},
        'by_author': {},
        'total_bytes': 0
    }

    print(f"Downloading {len(BOOKS)} books from Project Gutenberg...")
    print(f"Output directory: {output_dir}\n")

    for book_id, author, title, genre in BOOKS:
        # Create filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        filename = f"{book_id}_{author.lower()}_{safe_title}.txt"
        output_path = output_dir / filename

        # Skip if already exists
        if output_path.exists():
            print(f"⏭️  Skipping (exists): {title}")
            file_size = output_path.stat().st_size
            stats['successful'] += 1
            stats['total_bytes'] += file_size
            stats['by_genre'][genre] = stats['by_genre'].get(genre, 0) + 1
            stats['by_author'][author] = stats['by_author'].get(author, 0) + 1
            continue

        print(f"⬇️  Downloading: {title} by {author}")

        if download_gutenberg_text(book_id, output_path):
            file_size = output_path.stat().st_size
            stats['successful'] += 1
            stats['total_bytes'] += file_size
            stats['by_genre'][genre] = stats['by_genre'].get(genre, 0) + 1
            stats['by_author'][author] = stats['by_author'].get(author, 0) + 1
            print(f"   ✅ Success ({file_size:,} bytes)")
        else:
            stats['failed'] += 1
            print(f"   ❌ Failed")

    return stats


def print_statistics(stats: dict):
    """Print download statistics."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successfully downloaded: {stats['successful']} books")
    print(f"Failed: {stats['failed']} books")
    print(f"Total size: {stats['total_bytes'] / (1024*1024):.2f} MB")

    print("\nBy Genre:")
    for genre, count in sorted(stats['by_genre'].items()):
        print(f"  {genre:20s}: {count} books")

    print("\nBy Author:")
    for author, count in sorted(stats['by_author'].items(), key=lambda x: -x[1]):
        print(f"  {author:20s}: {count} books")


def main():
    parser = argparse.ArgumentParser(
        description="Download public domain literature from Project Gutenberg"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./raw",
        help="Directory to save downloaded books (default: ./raw)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Project Gutenberg Literature Downloader")
    print("=" * 60)
    print()

    stats = download_all_books(output_dir)
    print_statistics(stats)

    if stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} downloads failed. You may want to re-run the script.")
        return 1

    print("\n✅ All downloads complete!")
    return 0


if __name__ == "__main__":
    exit(main())
