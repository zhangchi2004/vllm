export CUDA_VISIBLE_DEVICES=0,1,2,3
python benchmarks/benchmark_prefix_caching.py \
    --model ibm-granite/granite-4.0-h-small \
    --num-prompts 30 \
    --repeat-count 200 \
    --input-length-range 800:900 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    # --enable-prefix-caching