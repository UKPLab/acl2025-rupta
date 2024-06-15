import os
import argparse
from immediate_refinement import run_immediate_refinement
from immediate_reflexion import run_immediate_reflexion

from simple import run_simple
from reflexion import run_reflexion
from reflexion_ucs import run_reflexion_ucs
from test_acc import run_test_acc
from reddit_clss import run_reddit_clss
from utils import read_jsonl, read_jsonl_gz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument(
        "--pe_model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)

    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")  # Temporary

    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--no_utility", action='store_true',
                        help="Whether add utility evaluation module")
    parser.add_argument("--cot", action='store_true',
                        help="Whether use COT prompt to generate the initial state")
    parser.add_argument("--mem_len", type=int,
                        help="The maximum length of memory", default=3)
    parser.add_argument("--p_threshold", type=int,
                        help="The maximum number of distinguishable people", default=10)
    parser.add_argument("--rag_data_path", type=str,
                        help="The path of rag data")
    parser.add_argument("--rag_embed_cache_dir", type=str,
                        help="The embedding cache directory")
    parser.add_argument("--rag_num", type=int,
                        help="The maximum number of retrieved documents", default=5)
    parser.add_argument(
        "--ue_model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument(
        "--act_model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument(
        "--parser_model", type=str, help="OpenAI models only for now. For best results, use GPT-4")

    # TODO: implement this
    # parser.add_argument("--is_resume", action='store_true', help="To resume run")
    # parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters", "no_utility", "p_threshold",
                                                           "mem_len", "rag_embed_cache_dir", "rag_num", "rag_data_path",
                                                           "cot"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor", "no_utility", "p_threshold",
                                                                        "mem_len", "rag_embed_cache_dir", "rag_num"
                                                                        , "rag_data_path", "cot"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor", "no_utility", "p_threshold"
                                                                         , "mem_len", "rag_embed_cache_dir", "rag_num",
                                                                         "rag_data_path", "cot"])
    elif strategy == "reflexion-ucs":
        return kwargs_wrapper_gen(run_reflexion_ucs)
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters", "mem_len", "ue_model_name",
                                                             "act_model_name", "parser_model_name", "cot"])
    elif strategy == 'reddit_clss':
        return kwargs_wrapper_gen(run_reddit_clss, delete_keys=["expansion_factor", "max_iters", "mem_len",
                                                                "pe_model_name", "act_model_name", "parser_model_name",
                                                                "cot"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    # dataset_name = os.path.basename(args.dataset_path).replace(".jsonl", "")
    dataset_name = args.language

    # check if log path already exists
    log_dir = str(os.path.join(args.root_dir, args.run_name))
    log_path = os.path.join(log_dir,
                            f"{dataset_name}_{args.strategy}_{args.run_name}_{args.max_iters}_act_{args.act_model.replace('/', '-')}"
                            f"_pe_{args.pe_model.replace('/', '-')}_ue_{args.ue_model.replace('/', '-')}"
                            f"_parser_{args.parser_model.replace('/', '-')}"
                            f"_pass_at_k_{args.pass_at_k}_{args.language}_no-utility_{args.no_utility}"
                            f"_COT_{args.cot}_p-threshold_{args.p_threshold}_mem-len_{args.mem_len}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    # start the run
    # evaluate with pass@k
    run_strategy(
        dataset=dataset,
        pe_model_name=args.pe_model,
        ue_model_name=args.ue_model,
        act_model_name=args.act_model,
        parser_model_name=args.parser_model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        expansion_factor=args.expansion_factor,
        is_leetcode=args.is_leetcode,
        no_utility=args.no_utility,
        cot=args.cot,
        mem_len=args.mem_len,
        p_threshold=args.p_threshold,
        rag_data_path=args.rag_data_path,
        rag_num=args.rag_num,
        rag_embed_cache_dir=args.rag_embed_cache_dir,
    )

    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
