#!/usr/bin/env python3
"""
Comprehensive data analysis script for English swipe dataset.
Checks for outliers, verifies statistics, and identifies data quality issues.
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import statistics


def load_stats_file(stats_path: str) -> Dict[str, Any]:
    """Load the reference statistics file"""
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}


def analyze_single_file(file_path: str, max_samples: int = None) -> Dict[str, Any]:
    """Analyze a single JSONL file and return comprehensive statistics"""
    print(f"\n📊 Analyzing {os.path.basename(file_path)}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return {}
    
    stats = {
        'total_samples': 0,
        'coordinate_stats': {'x': [], 'y': []},
        'timestamp_stats': [],
        'sequence_lengths': [],
        'word_lengths': [],
        'words': [],
        'grid_names': [],
        'trajectory_durations': [],
        'velocity_stats': {'x': [], 'y': []},
        'errors': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                stats['total_samples'] += 1
                
                # Extract curve data
                curve = data['curve']
                x_coords = curve['x']
                y_coords = curve['y']
                timestamps = curve['t']
                
                # Basic validation
                if len(x_coords) != len(y_coords) or len(x_coords) != len(timestamps):
                    stats['errors'].append(f"Line {line_num + 1}: Mismatched coordinate/timestamp lengths")
                    continue
                
                if len(x_coords) == 0:
                    stats['errors'].append(f"Line {line_num + 1}: Empty trajectory")
                    continue
                
                # Coordinate statistics
                stats['coordinate_stats']['x'].extend(x_coords)
                stats['coordinate_stats']['y'].extend(y_coords)
                stats['sequence_lengths'].append(len(x_coords))
                
                # Timestamp statistics
                stats['timestamp_stats'].extend(timestamps)
                if len(timestamps) > 1:
                    duration = timestamps[-1] - timestamps[0]
                    stats['trajectory_durations'].append(duration)
                
                # Velocity calculation
                if len(x_coords) > 1:
                    dt_list = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                    dx_list = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
                    dy_list = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
                    
                    # Calculate velocities (pixels per millisecond)
                    for i, dt in enumerate(dt_list):
                        if dt > 0:  # Avoid division by zero
                            vx = dx_list[i] / dt
                            vy = dy_list[i] / dt
                            stats['velocity_stats']['x'].append(abs(vx))
                            stats['velocity_stats']['y'].append(abs(vy))
                
                # Word statistics
                if 'word' in data and data['word']:
                    word = data['word']
                    stats['words'].append(word)
                    stats['word_lengths'].append(len(word))
                
                # Grid name
                stats['grid_names'].append(curve['grid_name'])
                
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num + 1}: JSON decode error - {e}")
            except Exception as e:
                stats['errors'].append(f"Line {line_num + 1}: Unexpected error - {e}")
    
    print(f"  Processed {stats['total_samples']} samples")
    if stats['errors']:
        print(f"  ⚠️ Found {len(stats['errors'])} errors")
    
    return stats


def compute_summary_stats(values: List[float], name: str) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of values"""
    if not values:
        return {}
    
    values = [v for v in values if v is not None]
    if not values:
        return {}
    
    try:
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'q95': np.percentile(values, 95),
            'q99': np.percentile(values, 99)
        }
    except Exception as e:
        print(f"  ⚠️ Error computing stats for {name}: {e}")
        return {}


def identify_outliers(values: List[float], name: str, threshold_std: float = 3.0) -> Tuple[List[int], Dict[str, Any]]:
    """Identify outliers using standard deviation method"""
    if len(values) < 2:
        return [], {}
    
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)
    
    if std_val == 0:
        return [], {}
    
    outlier_indices = []
    for i, val in enumerate(values):
        z_score = abs(val - mean_val) / std_val
        if z_score > threshold_std:
            outlier_indices.append(i)
    
    outlier_info = {
        'count': len(outlier_indices),
        'percentage': (len(outlier_indices) / len(values)) * 100,
        'threshold_std': threshold_std,
        'mean': mean_val,
        'std': std_val
    }
    
    return outlier_indices, outlier_info


def analyze_all_files():
    """Analyze all converted dataset files"""
    data_dir = "data"
    files_to_analyze = [
        'converted_english_swipes_train.jsonl',
        'converted_english_swipes_val.jsonl',
        'converted_english_swipes_test.jsonl'
    ]
    
    stats_file = os.path.join(data_dir, "english_swipes_stats.json")
    reference_stats = load_stats_file(stats_file)
    
    print("🔍 Comprehensive Data Analysis for English Swipe Dataset")
    print("=" * 60)
    
    all_file_stats = {}
    
    # Analyze each file
    for filename in files_to_analyze:
        file_path = os.path.join(data_dir, filename)
        file_stats = analyze_single_file(file_path, max_samples=None)  # Analyze all samples
        all_file_stats[filename] = file_stats
    
    # Generate comprehensive report
    print(f"\n" + "=" * 60)
    print("📈 COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_samples = sum(stats['total_samples'] for stats in all_file_stats.values())
    total_errors = sum(len(stats['errors']) for stats in all_file_stats.values())
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total samples across all files: {total_samples:,}")
    print(f"  Total errors found: {total_errors}")
    
    # Compare with reference stats
    if reference_stats:
        print(f"\n📋 COMPARISON WITH REFERENCE STATS:")
        for filename, expected_count in reference_stats.items():
            if isinstance(expected_count, int):
                actual_filename = f"converted_{filename}"
                if actual_filename in all_file_stats:
                    actual_count = all_file_stats[actual_filename]['total_samples']
                    match = "✅" if actual_count == expected_count else "❌"
                    print(f"  {filename}: Expected {expected_count:,}, Got {actual_count:,} {match}")
    
    # Detailed analysis for each file
    for filename, stats in all_file_stats.items():
        if stats['total_samples'] == 0:
            continue
            
        print(f"\n" + "-" * 50)
        print(f"📄 {filename.upper()}")
        print("-" * 50)
        
        # Basic info
        print(f"Samples: {stats['total_samples']:,}")
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(stats['errors']) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        
        # Coordinate analysis
        if stats['coordinate_stats']['x']:
            print(f"\n🎯 COORDINATE ANALYSIS:")
            
            x_stats = compute_summary_stats(stats['coordinate_stats']['x'], 'X coordinates')
            y_stats = compute_summary_stats(stats['coordinate_stats']['y'], 'Y coordinates')
            
            print(f"  X coordinates: [{x_stats['min']:.1f}, {x_stats['max']:.1f}], mean={x_stats['mean']:.1f}, std={x_stats['std']:.1f}")
            print(f"  Y coordinates: [{y_stats['min']:.1f}, {y_stats['max']:.1f}], mean={y_stats['mean']:.1f}, std={y_stats['std']:.1f}")
            
            # Check for coordinates outside expected keyboard bounds
            keyboard_width, keyboard_height = 360, 215
            x_outliers = [x for x in stats['coordinate_stats']['x'] if x < 0 or x > keyboard_width]
            y_outliers = [y for y in stats['coordinate_stats']['y'] if y < 0 or y > keyboard_height]
            
            if x_outliers:
                print(f"  ⚠️ X coordinates outside [0, {keyboard_width}]: {len(x_outliers)} values")
                print(f"    Range: [{min(x_outliers):.1f}, {max(x_outliers):.1f}]")
            
            if y_outliers:
                print(f"  ⚠️ Y coordinates outside [0, {keyboard_height}]: {len(y_outliers)} values")
                print(f"    Range: [{min(y_outliers):.1f}, {max(y_outliers):.1f}]")
        
        # Sequence length analysis
        if stats['sequence_lengths']:
            print(f"\n📏 SEQUENCE LENGTH ANALYSIS:")
            seq_stats = compute_summary_stats(stats['sequence_lengths'], 'Sequence lengths')
            print(f"  Length: [{seq_stats['min']:.0f}, {seq_stats['max']:.0f}], mean={seq_stats['mean']:.1f}, median={seq_stats['median']:.1f}")
            
            # Check for extremely short or long sequences
            short_seqs = [s for s in stats['sequence_lengths'] if s < 5]
            long_seqs = [s for s in stats['sequence_lengths'] if s > 1000]
            
            if short_seqs:
                print(f"  ⚠️ Very short sequences (< 5 points): {len(short_seqs)}")
            if long_seqs:
                print(f"  ⚠️ Very long sequences (> 1000 points): {len(long_seqs)}")
                print(f"    Longest: {max(long_seqs)} points")
        
        # Timestamp analysis
        if stats['timestamp_stats']:
            print(f"\n⏰ TIMESTAMP ANALYSIS:")
            ts_stats = compute_summary_stats(stats['timestamp_stats'], 'Timestamps')
            print(f"  Range: [{ts_stats['min']:.0f}, {ts_stats['max']:.0f}] ms")
            
            # Check for negative or zero timestamps
            negative_ts = [t for t in stats['timestamp_stats'] if t < 0]
            if negative_ts:
                print(f"  ⚠️ Negative timestamps: {len(negative_ts)}")
            
            # Check trajectory durations
            if stats['trajectory_durations']:
                dur_stats = compute_summary_stats(stats['trajectory_durations'], 'Durations')
                print(f"  Durations: mean={dur_stats['mean']:.0f}ms, median={dur_stats['median']:.0f}ms")
                
                # Check for extremely short or long durations
                short_dur = [d for d in stats['trajectory_durations'] if d < 100]  # < 100ms
                long_dur = [d for d in stats['trajectory_durations'] if d > 10000]  # > 10s
                
                if short_dur:
                    print(f"  ⚠️ Very short trajectories (< 100ms): {len(short_dur)}")
                if long_dur:
                    print(f"  ⚠️ Very long trajectories (> 10s): {len(long_dur)}")
        
        # Velocity analysis
        if stats['velocity_stats']['x'] and stats['velocity_stats']['y']:
            print(f"\n🏃 VELOCITY ANALYSIS:")
            vx_stats = compute_summary_stats(stats['velocity_stats']['x'], 'X velocity')
            vy_stats = compute_summary_stats(stats['velocity_stats']['y'], 'Y velocity')
            
            print(f"  X velocity: mean={vx_stats['mean']:.3f}, q95={vx_stats['q95']:.3f} px/ms")
            print(f"  Y velocity: mean={vy_stats['mean']:.3f}, q95={vy_stats['q95']:.3f} px/ms")
            
            # Check for extremely high velocities (potential data errors)
            high_vx = [v for v in stats['velocity_stats']['x'] if v > 2.0]  # > 2 px/ms
            high_vy = [v for v in stats['velocity_stats']['y'] if v > 2.0]
            
            if high_vx:
                print(f"  ⚠️ Very high X velocities (> 2 px/ms): {len(high_vx)}")
                print(f"    Max: {max(high_vx):.3f} px/ms")
            if high_vy:
                print(f"  ⚠️ Very high Y velocities (> 2 px/ms): {len(high_vy)}")
                print(f"    Max: {max(high_vy):.3f} px/ms")
        
        # Word analysis (for train/val datasets)
        if stats['words']:
            print(f"\n📝 WORD ANALYSIS:")
            word_stats = compute_summary_stats(stats['word_lengths'], 'Word lengths')
            print(f"  Word count: {len(stats['words'])}")
            print(f"  Word lengths: [{word_stats['min']:.0f}, {word_stats['max']:.0f}], mean={word_stats['mean']:.1f}")
            
            # Most common words
            word_counts = Counter(stats['words'])
            print(f"  Most common words: {dict(word_counts.most_common(10))}")
            
            # Check for unusually long words
            long_words = [w for w in stats['words'] if len(w) > 15]
            if long_words:
                print(f"  ⚠️ Very long words (> 15 chars): {len(long_words)}")
                print(f"    Examples: {long_words[:5]}")
        
        # Grid name analysis
        if stats['grid_names']:
            grid_counts = Counter(stats['grid_names'])
            print(f"\n⌨️ GRID ANALYSIS:")
            print(f"  Grid names: {dict(grid_counts)}")
    
    # Final recommendations
    print(f"\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    # Check for coordinate issues
    for filename, stats in all_file_stats.items():
        if stats['coordinate_stats']['x']:
            x_outliers = [x for x in stats['coordinate_stats']['x'] if x < 0 or x > 360]
            y_outliers = [y for y in stats['coordinate_stats']['y'] if y < 0 or y > 215]
            if x_outliers or y_outliers:
                recommendations.append(f"Clean coordinate outliers in {filename}")
    
    # Check for sequence length issues
    for filename, stats in all_file_stats.items():
        if stats['sequence_lengths']:
            very_short = [s for s in stats['sequence_lengths'] if s < 3]
            if very_short:
                recommendations.append(f"Consider filtering very short sequences in {filename}")
    
    # Check for velocity outliers
    for filename, stats in all_file_stats.items():
        if stats['velocity_stats']['x']:
            high_vel = [v for v in stats['velocity_stats']['x'] + stats['velocity_stats']['y'] if v > 5.0]
            if high_vel:
                recommendations.append(f"Consider capping extreme velocities in {filename}")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✅ No major data quality issues detected!")
    
    print(f"\n✅ Analysis complete!")
    return all_file_stats


if __name__ == "__main__":
    try:
        results = analyze_all_files()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()