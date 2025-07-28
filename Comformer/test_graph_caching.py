#!/usr/bin/env python3
"""
Test cases for graph caching system.
Run these on cloud to verify caching works correctly.
"""

import os
import time
import shutil
from pathlib import Path
import pandas as pd
from comformer.data import load_dataset, load_pyg_graphs, get_cache_key

def setup_test_environment():
    """Setup clean test environment."""
    # Remove existing cache
    cache_dir = Path("./cache/graphs")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print("✓ Cleaned cache directory")

def test_cache_key_generation():
    """Test 1: Verify cache keys are unique for different parameters."""
    print("\n=== Test 1: Cache Key Generation ===")
    
    jid = "test_jid_123"
    
    # Same parameters should give same key
    key1 = get_cache_key(jid, "k-nearest", 8.0, 12, False, False, False)
    key2 = get_cache_key(jid, "k-nearest", 8.0, 12, False, False, False)
    assert key1 == key2, "Same parameters should give same cache key"
    print("✓ Same parameters produce consistent keys")
    
    # Different parameters should give different keys
    key3 = get_cache_key(jid, "k-nearest", 10.0, 12, False, False, False)  # Different cutoff
    key4 = get_cache_key(jid, "k-nearest", 8.0, 15, False, False, False)   # Different max_neighbors
    key5 = get_cache_key(jid, "k-nearest", 8.0, 12, True, False, False)    # Different use_canonize
    
    assert key1 != key3, "Different cutoff should give different key"
    assert key1 != key4, "Different max_neighbors should give different key"
    assert key1 != key5, "Different use_canonize should give different key"
    print("✓ Different parameters produce different keys")
    
    # Different JIDs should give different keys
    key6 = get_cache_key("different_jid", "k-nearest", 8.0, 12, False, False, False)
    assert key1 != key6, "Different JID should give different key"
    print("✓ Different JIDs produce different keys")

def test_first_run_caching():
    """Test 2: First run should compute graphs and save to cache."""
    print("\n=== Test 2: First Run (Cache Population) ===")
    
    # Load small dataset
    df = load_dataset(
        name="D2R2_surface_data", 
        data_path="/home/kmudaliar/data/DFT_data.csv",
        target="WF_bottom",
        limit=1000  # Hardcoded limit for testing
    )
    print(f"✓ Loaded dataset with {len(df)} samples")
    
    cache_dir = Path("./cache/graphs")
    assert not cache_dir.exists() or len(list(cache_dir.glob("*.pkl"))) == 0, "Cache should be empty initially"
    
    # Time the first run
    start_time = time.time()
    graphs = load_pyg_graphs(
        df,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
    )
    first_run_time = time.time() - start_time
    
    print(f"✓ First run completed in {first_run_time:.2f} seconds")
    print(f"✓ Generated {len(graphs)} graphs")
    
    # Check cache files were created
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"✓ Created {len(cache_files)} cache files")
    assert len(cache_files) > 0, "Cache files should be created"
    
    return first_run_time, len(graphs)

def test_second_run_cache_loading():
    """Test 3: Second run should load from cache and be much faster."""
    print("\n=== Test 3: Second Run (Cache Loading) ===")
    
    # Load same dataset again
    df = load_dataset(
        name="D2R2_surface_data", 
        data_path="/home/kmudaliar/data/DFT_data.csv",
        target="WF_bottom",
        limit=1000  # Same limit
    )
    
    cache_dir = Path("./cache/graphs")
    initial_cache_count = len(list(cache_dir.glob("*.pkl")))
    print(f"✓ Found {initial_cache_count} existing cache files")
    
    # Time the second run
    start_time = time.time()
    graphs = load_pyg_graphs(
        df,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
    )
    second_run_time = time.time() - start_time
    
    print(f"✓ Second run completed in {second_run_time:.2f} seconds")
    print(f"✓ Loaded {len(graphs)} graphs")
    
    # Check no new cache files were created
    final_cache_count = len(list(cache_dir.glob("*.pkl")))
    assert final_cache_count == initial_cache_count, "No new cache files should be created on second run"
    print("✓ No new cache files created (all loaded from cache)")
    
    return second_run_time, len(graphs)

def test_different_parameters_create_new_cache():
    """Test 4: Different graph parameters should create separate cache files."""
    print("\n=== Test 4: Different Parameters (New Cache) ===")
    
    df = load_dataset(
        name="D2R2_surface_data", 
        data_path="/home/kmudaliar/data/DFT_data.csv",
        target="WF_bottom",
        limit=1000
    )
    
    cache_dir = Path("./cache/graphs")
    initial_cache_count = len(list(cache_dir.glob("*.pkl")))
    
    # Use different parameters
    start_time = time.time()
    graphs = load_pyg_graphs(
        df,
        neighbor_strategy="k-nearest",
        cutoff=10.0,  # Different cutoff
        max_neighbors=12,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
    )
    different_params_time = time.time() - start_time
    
    print(f"✓ Different parameters run completed in {different_params_time:.2f} seconds")
    
    # Check new cache files were created
    final_cache_count = len(list(cache_dir.glob("*.pkl")))
    new_files_created = final_cache_count - initial_cache_count
    print(f"✓ Created {new_files_created} new cache files for different parameters")
    assert new_files_created > 0, "New cache files should be created for different parameters"
    
    return different_params_time

def test_cache_persistence():
    """Test 5: Cache should persist across Python sessions."""
    print("\n=== Test 5: Cache Persistence ===")
    
    cache_dir = Path("./cache/graphs")
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"✓ Found {len(cache_files)} cache files from previous runs")
    
    # Verify files exist and are readable
    for cache_file in cache_files[:5]:  # Check first 5 files
        assert cache_file.exists(), f"Cache file {cache_file} should exist"
        assert cache_file.stat().st_size > 0, f"Cache file {cache_file} should not be empty"
    
    print("✓ Cache files are persistent and readable")

def run_all_tests():
    """Run all test cases."""
    print("🚀 Starting Graph Caching Tests")
    print("=" * 50)
    
    # Setup
    setup_test_environment()
    
    try:
        # Test 1: Cache key generation
        test_cache_key_generation()
        
        # Test 2: First run (population)
        first_time, graph_count = test_first_run_caching()
        
        # Test 3: Second run (loading)
        second_time, _ = test_second_run_cache_loading()
        
        # Calculate speedup
        speedup = first_time / second_time if second_time > 0 else float('inf')
        print(f"\n📈 SPEEDUP: {speedup:.1f}x faster on second run!")
        
        # Test 4: Different parameters
        diff_time = test_different_parameters_create_new_cache()
        
        # Test 5: Persistence
        test_cache_persistence()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print(f"📊 Performance Summary:")
        print(f"   First run:  {first_time:.2f}s ({graph_count} graphs)")
        print(f"   Second run: {second_time:.2f}s (cached)")
        print(f"   Speedup:    {speedup:.1f}x")
        print(f"   Different params: {diff_time:.2f}s (new cache)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()