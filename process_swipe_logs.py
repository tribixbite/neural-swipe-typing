#!/usr/bin/env python3
"""
Process swipe trace log files from data/how_we_swipe_logs/swipetraces/ into JSONL format.

Input format: sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius y_radius angle word is_err
Output format: {"word": "word", "points": [{"t": 0.0, "x": 0.5, "y": 0.5}, ...], "id": 1, "session": "session", "timestamp": 123}
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional

def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line into components."""
    parts = line.strip().split()
    if len(parts) < 12:
        return None

    try:
        return {
            'sentence': parts[0],
            'timestamp': int(parts[1]),
            'keyb_width': int(parts[2]),
            'keyb_height': int(parts[3]),
            'event': parts[4],
            'x_pos': int(parts[5]),
            'y_pos': int(parts[6]),
            'x_radius': float(parts[7]),
            'y_radius': float(parts[8]),
            'angle': float(parts[9]),
            'word': parts[10],
            'is_err': int(parts[11])
        }
    except (ValueError, IndexError):
        return None

def process_log_file(file_path: str) -> List[Dict[str, Any]]:
    """Process a single log file and extract swipe traces."""
    traces = []
    current_trace = []
    trace_start_time = None
    keyb_width = None
    keyb_height = None
    word = None
    sentence = None
    trace_id = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parsed = parse_log_line(line)
        if not parsed:
            continue

        # Skip error traces
        if parsed['is_err'] == 1:
            continue

        if parsed['event'] == 'touchstart':
            # Start new trace
            if current_trace and len(current_trace) >= 3:  # Need at least 3 points for a meaningful trace
                # Save previous trace if it exists
                trace = create_trace_json(current_trace, trace_start_time, keyb_width, keyb_height, word, sentence, trace_id, file_path)
                if trace:
                    traces.append(trace)
                    trace_id += 1

            # Initialize new trace
            current_trace = [parsed]
            trace_start_time = parsed['timestamp']
            keyb_width = parsed['keyb_width']
            keyb_height = parsed['keyb_height']
            word = parsed['word']
            sentence = parsed['sentence']

        elif parsed['event'] in ['touchmove', 'touchend'] and current_trace:
            # Continue current trace
            current_trace.append(parsed)

            if parsed['event'] == 'touchend':
                # End trace
                if len(current_trace) >= 3:
                    trace = create_trace_json(current_trace, trace_start_time, keyb_width, keyb_height, word, sentence, trace_id, file_path)
                    if trace:
                        traces.append(trace)
                        trace_id += 1
                current_trace = []

    # Handle final trace if file doesn't end with touchend
    if current_trace and len(current_trace) >= 3:
        trace = create_trace_json(current_trace, trace_start_time, keyb_width, keyb_height, word, sentence, trace_id, file_path)
        if trace:
            traces.append(trace)

    return traces

def create_trace_json(trace_points: List[Dict], start_time: int, keyb_width: int, keyb_height: int,
                     word: str, sentence: str, trace_id: int, file_path: str) -> Optional[Dict[str, Any]]:
    """Convert trace points to target JSON format."""
    if len(trace_points) < 3:
        return None

    points = []
    for point in trace_points:
        # Normalize coordinates and time
        t = float(point['timestamp'] - start_time)
        x = float(point['x_pos']) / float(keyb_width)
        y = float(point['y_pos']) / float(keyb_height)

        points.append({
            't': t,
            'x': x,
            'y': y
        })

    # Extract session from filename (remove .log extension)
    session = os.path.basename(file_path).replace('.log', '')

    return {
        'word': word,
        'points': points,
        'id': trace_id,
        'session': session,
        'timestamp': start_time
    }

def main():
    """Main processing function."""
    log_dir = './data/how_we_swipe_logs/swipetraces/'
    output_file = './how_we_swipe.jsonl'

    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} not found")
        return

    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    print(f"Found {len(log_files)} log files")

    all_traces = []
    global_trace_id = 0

    for i, log_file in enumerate(log_files):
        print(f"Processing {log_file} ({i+1}/{len(log_files)})")
        try:
            traces = process_log_file(log_file)
            # Update trace IDs to be globally unique
            for trace in traces:
                trace['id'] = global_trace_id
                global_trace_id += 1
            all_traces.extend(traces)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue

    print(f"Extracted {len(all_traces)} traces total")

    # Write to JSONL
    with open(output_file, 'w') as f:
        for trace in all_traces:
            f.write(json.dumps(trace) + '\n')

    print(f"Wrote traces to {output_file}")

    # Print some statistics
    words = [trace['word'] for trace in all_traces]
    unique_words = set(words)
    print(f"Unique words: {len(unique_words)}")
    print(f"Average points per trace: {sum(len(trace['points']) for trace in all_traces) / len(all_traces):.1f}")

if __name__ == '__main__':
    main()