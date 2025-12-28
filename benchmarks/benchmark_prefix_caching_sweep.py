# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the efficiency of prefix caching with a sweep of parameters.

This script replicates the logic of vllm2.sh but avoids re-initializing the LLM engine
for each parameter combination.
"""

import dataclasses
import json
import random
import time
import os
import contextlib
import sys

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {end_time - start_time}")
    return outputs, end_time - start_time


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove the special tokens.
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
) -> list[Request]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
    prefix_len: int,
) -> list[Request]:
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert min_len <= prompt_len <= max_len, (
            f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        )
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False
) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def run_sweep(args, llm, tokenizer):
    num_prompts_list = [1, 10, 20, 100, 500]
    input_length_starts = [10, 50, 100, 200, 400, 600, 800, 1200, 1600, 2400, 3200]
    
    results_dir = "benchmarks/vllm-sweep"
    os.makedirs(results_dir, exist_ok=True)

    for num_prompts in num_prompts_list:
        for input_length in input_length_starts:
            input_length_upper = input_length + 1
            repeat_count = 5000 // num_prompts
            
            log_filename = f"{args.log_prefix}_numprompts_{num_prompts}_inputlen_{input_length}.log"
            log_path = os.path.join(results_dir, log_filename)
            
            print(f"Running sweep: num_prompts={num_prompts}, input_length={input_length} -> {log_path}")
            
            # Update args for this iteration
            current_input_length_range = (input_length, input_length_upper)
            
            # Sample requests
            if args.dataset_path is not None:
                 filtered_requests = sample_requests_from_dataset(
                    dataset_path=args.dataset_path,
                    num_requests=num_prompts,
                    tokenizer=tokenizer,
                    input_length_range=current_input_length_range,
                    fixed_output_len=args.output_len,
                )
            else:
                filtered_requests = sample_requests_from_random(
                    num_requests=num_prompts,
                    tokenizer=tokenizer,
                    input_length_range=current_input_length_range,
                    fixed_output_len=args.output_len,
                    prefix_len=args.prefix_len,
                )
            
            prompts = repeat_and_sort_requests(
                filtered_requests, repeat_count=repeat_count, sort=args.sort
            )
            
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=args.output_len,
                detokenize=not args.disable_detokenize,
            )
            
            print(f"Running for num_prompts={num_prompts}, input_length={input_length}, repeat_count={repeat_count}")
            print(f"Sampled {len(filtered_requests)} requests.")
            
            # Run benchmark
            outputs, duration = test_prefix(llm=llm, prompts=prompts, sampling_params=sampling_params)
            
            # Calculate metrics
            total_requests = len(outputs)
            total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
            total_output_tokens = sum(sum(len(c.token_ids) for c in o.outputs) for o in outputs)
            total_cached_tokens = sum(o.num_cached_tokens or 0 for o in outputs)
            
            avg_latency = 0
            avg_ttft = 0
            if total_requests > 0:
                latencies = []
                ttfts = []
                for o in outputs:
                    if o.metrics:
                        arrival = o.metrics.arrival_time
                        # finished_time might be None if not finished, but generate() waits.
                        # if o.metrics.finished_time:
                        #     latencies.append(o.metrics.finished_time - arrival)
                        # if o.metrics.first_token_time:
                        #     ttfts.append(o.metrics.first_token_time - arrival)
                
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                if ttfts:
                    avg_ttft = sum(ttfts) / len(ttfts)

            requests_per_second = total_requests / duration if duration > 0 else 0
            output_tokens_per_second = total_output_tokens / duration if duration > 0 else 0
            prefix_cache_hit_rate = total_cached_tokens / total_input_tokens if total_input_tokens > 0 else 0

            # Write to log file
            with open(log_path, 'w') as f:
                f.write(f"Command: num_prompts={num_prompts}, input_length={input_length}, repeat_count={repeat_count}\n")
                f.write(f"Sampled requests: {len(filtered_requests)}\n")
                
                prompt_lens = [req.prompt_len for req in filtered_requests]
                if prompt_lens:
                    f.write(f"Average input length: {sum(prompt_lens) / len(prompt_lens):.2f}\n")
                    f.write(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}\n")
                    f.write(f"Min Prompt Length: {min(prompt_lens)}\n")
                    f.write(f"Max Prompt Length: {max(prompt_lens)}\n")
                
                f.write("-" * 20 + "\n")
                f.write(f"Duration: {duration:.4f} s\n")
                f.write(f"Total Requests: {total_requests}\n")
                f.write(f"Total Input Tokens: {total_input_tokens}\n")
                f.write(f"Total Output Tokens: {total_output_tokens}\n")
                f.write(f"Total Cached Tokens: {total_cached_tokens}\n")
                f.write(f"Prefix Cache Hit Rate: {prefix_cache_hit_rate:.2%}\n")
                f.write(f"Requests/sec: {requests_per_second:.2f}\n")
                f.write(f"Output Tokens/sec: {output_tokens_per_second:.2f}\n")
                f.write(f"Average Latency: {avg_latency:.4f} s\n")
                f.write(f"Average TTFT: {avg_ttft:.4f} s\n")



def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the performance with or without "
        "automatic prefix caching."
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument("--output-len", type=int, default=10)
    # Note: num-prompts, repeat-count, input-length-range are ignored in sweep mode
    # but kept for compatibility with EngineArgs if needed, or just as placeholders
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Ignored in sweep mode",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Ignored in sweep mode",
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort prompts by input length"
    )
    parser.add_argument(
        "--input-length-range",
        type=str,
        default="0:100",
        help='Ignored in sweep mode',
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Specifies the length of a common prefix to be "
        "added to the input prompt. The input-length-range will "
        "subtract this length when filtering prompts. Only used "
        "when dataset-path is not provided.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="dev",
        help="Prefix for the log files (e.g. 'dev', 'base_no_prefix_cache')",
    )

    parser = EngineArgs.add_cli_args(parser)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Init LLM once
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    
    run_sweep(args, llm, tokenizer)
