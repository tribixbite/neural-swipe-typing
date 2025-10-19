#!/usr/bin/env python3
"""
Fix Time Values in Existing Synthetic Traces

Processes all existing synthetic trace files to fix non-sequential time values.
The API returns time deltas that can be negative or non-monotonic, but we need
sequential timestamps for proper gesture traces.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
from tqdm import tqdm
import shutil


def fix_time_values(gesture_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fix time values in a single gesture to be sequential starting from 0"""
    if 'word_seq' not in gesture_data or 'time' not in gesture_data['word_seq']:
        return gesture_data

    time_values = gesture_data['word_seq']['time']
    if not time_values:
        return gesture_data

    # Convert to cumulative time starting from 0
    cumulative_time = 0.0
    fixed_times = []

    for i, t in enumerate(time_values):
        if i == 0:
            # Start at 0
            cumulative_time = 0.0
        else:
            # Add absolute value of time delta, with minimum step
            time_delta = max(abs(t - time_values[i-1]), 0.001)
            cumulative_time += time_delta

        fixed_times.append(cumulative_time)

    # Create a copy and update the time values
    fixed_gesture = gesture_data.copy()
    fixed_gesture['word_seq'] = gesture_data['word_seq'].copy()
    fixed_gesture['word_seq']['time'] = fixed_times

    return fixed_gesture


def process_file(input_file: Path, output_file: Path, backup: bool = True) -> tuple[int, int]:
    """Process a single JSONL file to fix time values

    Returns:
        tuple: (total_traces, fixed_traces)
    """
    if backup and input_file.exists():
        backup_file = input_file.with_suffix(input_file.suffix + '.backup')
        if not backup_file.exists():
            shutil.copy2(input_file, backup_file)

    total_traces = 0
    fixed_traces = 0

    # Read all traces
    traces = []
    if input_file.exists():
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trace = json.loads(line)
                    traces.append(trace)
                    total_traces += 1

    # Process and write fixed traces
    with open(output_file, 'w', encoding='utf-8') as f:
        for trace in traces:
            original_times = None
            if 'word_seq' in trace and 'time' in trace['word_seq']:
                original_times = trace['word_seq']['time'][:5]  # First 5 for comparison

            fixed_trace = fix_time_values(trace)

            # Check if anything was actually fixed
            if 'word_seq' in fixed_trace and 'time' in fixed_trace['word_seq']:
                fixed_times = fixed_trace['word_seq']['time'][:5]
                if original_times != fixed_times:
                    fixed_traces += 1

            json.dump(fixed_trace, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')

    return total_traces, fixed_traces


def main():
    parser = argparse.ArgumentParser(description='Fix time values in existing synthetic trace files')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/synthetic_traces',
        help='Directory containing synthetic trace files (default: data/synthetic_traces)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for fixed files (default: same as input, overwrites original)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='Create backup files before processing (default: True)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without making changes'
    )

    args = parser.parse_args()

    # Handle backup flag
    backup = args.backup and not args.no_backup

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1

    # Find all JSONL files
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return 1

    print(f"Found {len(jsonl_files)} JSONL files to process")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
        for file in jsonl_files:
            print(f"  Would process: {file}")
        return 0

    if output_dir != input_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    if backup:
        print("Creating backups before processing...")

    # Process files
    total_files = 0
    total_traces = 0
    total_fixed = 0

    for input_file in tqdm(jsonl_files, desc="Processing files"):
        output_file = output_dir / input_file.name

        try:
            traces, fixed = process_file(input_file, output_file, backup)
            total_files += 1
            total_traces += traces
            total_fixed += fixed

        except Exception as e:
            print(f"\nError processing {input_file}: {e}")
            continue

    # Summary
    print(f"\nProcessing complete:")
    print(f"  Files processed: {total_files}")
    print(f"  Total traces: {total_traces}")
    print(f"  Traces with fixed times: {total_fixed}")
    print(f"  Fix rate: {total_fixed/total_traces:.1%}" if total_traces > 0 else "  Fix rate: 0%")

    if backup and total_fixed > 0:
        print(f"\nBackup files created with .backup extension")
        print(f"Remove backups with: rm {input_dir}/*.backup")


if __name__ == "__main__":
    main()