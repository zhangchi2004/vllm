# For num
git switch zc-1
for num_prompts in 1 5 10 25 50 100 500; do
    for input_length in 0 100 200 400 800 1600; do
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        input_length_upper=$((input_length + 100))
        repeat_count=$((5000 / num_prompts))
        python benchmarks/benchmark_prefix_caching.py \
            --model ibm-granite/granite-4.0-h-small \
            --num-prompts $num_prompts \
            --repeat-count 200 \
            --input-length-range $input_length:$input_length_upper \
            --tensor-parallel-size 4 \
            --dtype bfloat16 \
            --enable-prefix-caching > benchmarks/vllm2/dev_granite4_h_small_prefix_caching_numprompts_${num_prompts}_inputlen_${input_length}.log 2>&1
        done
  done

  # For num
for num_prompts in 1 5 10 25 50 100 500; do
    for input_length in 0 100 200 400 800 1600; do
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        input_length_upper=$((input_length + 100))
        repeat_count=$((5000 / num_prompts))
        python benchmarks/benchmark_prefix_caching.py \
            --model ibm-granite/granite-4.0-h-small \
            --num-prompts $num_prompts \
            --repeat-count 200 \
            --input-length-range $input_length:$input_length_upper \
            --tensor-parallel-size 4 \
            --dtype bfloat16 > benchmarks/vllm2/base_no_prefix_cache_granite4_h_small_prefix_caching_numprompts_${num_prompts}_inputlen_${input_length}.log 2>&1
        done
  done

git switch main
# For num
for num_prompts in 1 5 10 25 50 100 500; do
    for input_length in 0 100 200 400 800 1600; do
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        input_length_upper=$((input_length + 100))
        repeat_count=$((5000 / num_prompts))
        python benchmarks/benchmark_prefix_caching.py \
            --model ibm-granite/granite-4.0-h-small \
            --num-prompts $num_prompts \
            --repeat-count 200 \
            --input-length-range $input_length:$input_length_upper \
            --tensor-parallel-size 4 \
            --dtype bfloat16 \
            --enable-prefix-caching > benchmarks/vllm2/base_prefix_cache_granite4_h_small_prefix_caching_numprompts_${num_prompts}_inputlen_${input_length}.log 2>&1
        done
  done