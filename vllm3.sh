export CUDA_VISIBLE_DEVICES=4,5,6,7

python benchmarks/benchmark_prefix_caching_sweep.py \
    --model ibm-granite/granite-4.0-h-small \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --log-prefix dev \

python benchmarks/benchmark_prefix_caching_sweep.py \
    --model ibm-granite/granite-4.0-h-small \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --log-prefix base_no_prefix_cache