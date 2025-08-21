#!/bin/bash

# GPU utility functions for dynamic GPU allocation

get_free_gpus() {
    # Get GPU utilization and return available GPUs
    local available_gpus=()
    
    # Check each GPU (0-3 for 4 GPUs)
    for gpu_id in {0..3}; do
        # Check GPU memory usage (returns percentage)
        gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id)
        gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$gpu_id)
        
        # Calculate usage percentage
        if [ "$gpu_mem_total" -gt 0 ]; then
            usage_percent=$((gpu_mem_used * 100 / gpu_mem_total))
            
            # Consider GPU available if less than 10% memory used
            if [ "$usage_percent" -lt 10 ]; then
                available_gpus+=($gpu_id)
            fi
        fi
    done
    
    echo "${available_gpus[@]}"
}

allocate_gpu() {
    local available_gpus=($(get_free_gpus))
    
    if [ ${#available_gpus[@]} -lt 1 ]; then
        echo "❌ ERROR: No free GPUs available"
        echo "All GPUs appear to be in use"
        return 1
    fi
    
    # Take the first available GPU
    local allocated_gpu=${available_gpus[0]}
    
    echo "$allocated_gpu"
    return 0
}

export_gpu_env() {
    local gpu_id=$1
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    echo "🎯 Allocated GPU: $gpu_id"
    echo "🔧 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}