#!/usr/bin/env python3
"""
Migrate Ranked Data to PostgreSQL
===================================
One-time migration script that loads medical_ranked.jsonl from the
pces_rlhf_experiments module into the PostgreSQL sft_ranked_data table.

Usage:
    python migrate_ranked_data.py
    python migrate_ranked_data.py --file /path/to/custom.jsonl
"""

import argparse
import sys

from sft_experiment_manager import ensure_tables, import_from_jsonl, get_ranked_data_stats


def main():
    parser = argparse.ArgumentParser(description="Migrate ranked JSONL data to PostgreSQL")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to JSONL file (default: pces_rlhf_experiments/medical_ranked.jsonl)")
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Ranked Data Migration")
    print("=" * 60)

    # Ensure tables exist
    print("\n📦 Ensuring database tables exist...")
    if not ensure_tables():
        print("❌ Failed to create tables. Check database connection.")
        sys.exit(1)
    print("✅ Tables ready")

    # Check current state
    print("\n📊 Current database state:")
    stats = get_ranked_data_stats()
    if stats.get("success"):
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Total groups:  {stats['total_groups']}")
        print(f"   Rank-1 count:  {stats['rank1_count']}")
    else:
        print(f"   Error: {stats.get('error')}")

    # Import data
    print("\n📥 Importing ranked data...")
    result = import_from_jsonl(args.file)

    if result.get("success"):
        print(f"✅ Import complete!")
        print(f"   Imported: {result['imported']} entries")
        print(f"   Skipped:  {result['skipped']} (duplicates or malformed)")
        print(f"   Source:   {result['file']}")
    else:
        print(f"❌ Import failed: {result.get('error')}")
        sys.exit(1)

    # Final state
    print("\n📊 Final database state:")
    stats = get_ranked_data_stats()
    if stats.get("success"):
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Total groups:  {stats['total_groups']}")
        print(f"   Rank-1 count:  {stats['rank1_count']}")

    print("\n" + "=" * 60)
    print("✅ Migration complete! Data is now available in the RLHF Admin panel.")
    print("=" * 60)


if __name__ == "__main__":
    main()
