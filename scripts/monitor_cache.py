#!/usr/bin/env python3
"""
Virtual Sensor Cache Monitoring Script

Monitors cache performance and provides statistics for cache hit/miss rates,
storage usage, and cache key distribution.
"""

import sys
import os
sys.path.append('/app')

from src.oft.transformer.utils.virtual_sensor_cache import get_virtual_sensor_cache
import json
from pathlib import Path

def monitor_cache():
    """Monitor virtual sensor cache statistics."""
    
    print("🔍 VIRTUAL SENSOR CACHE MONITOR")
    print("=" * 50)
    
    # Get cache instance (use correct Docker mount path)
    cache = get_virtual_sensor_cache(cache_dir="/cache")
    
    # Get statistics
    stats = cache.get_cache_stats()
    
    print(f"📁 Cache Directory: {stats['cache_dir']}")
    print(f"📊 Total Files: {stats['total_files']}")
    print(f"💾 Total Size: {stats['total_size_mb']:.2f} MB")
    print(f"📄 Average File Size: {stats['average_file_size_kb']:.2f} KB")
    
    # Analyze cache files
    cache_dir = Path(stats['cache_dir'])
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        
        # Group by sensor type
        sensor_counts = {}
        for cache_file in cache_files:
            # Extract sensor name from filename: cache_key_sensor_name.pkl
            parts = cache_file.stem.split('_')
            if len(parts) >= 2:
                sensor_name = '_'.join(parts[1:])  # Everything after first underscore
                sensor_counts[sensor_name] = sensor_counts.get(sensor_name, 0) + 1
        
        print("\n📈 CACHE DISTRIBUTION BY SENSOR:")
        for sensor_name, count in sorted(sensor_counts.items()):
            print(f"  {sensor_name}: {count} files")
        
        # Show recent cache files
        print(f"\n🕒 RECENT CACHE FILES (Last 10):")
        recent_files = sorted(cache_files, key=lambda f: f.stat().st_mtime, reverse=True)[:10]
        for cache_file in recent_files:
            size_kb = cache_file.stat().st_size / 1024
            print(f"  {cache_file.name} ({size_kb:.1f} KB)")
    
    print("\n✅ Cache monitoring complete")

if __name__ == "__main__":
    monitor_cache()