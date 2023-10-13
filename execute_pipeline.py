import argparse
from text_metrics_wrapper.preprocessing.preprocess_text_blocks import preprocess_text_blocks
from text_metrics_wrapper.utils.manage_jsonl_files import load_jsonl_file
from text_metrics_wrapper.utils.manage_json_files import write_json_file
from text_metrics_wrapper.metrics.parent.parent import Parent
from text_metrics_wrapper.metrics.bleurt.bleurt import Bleurt
from text_metrics_wrapper.metrics.gem.gem import Gem
from text_metrics_wrapper.metrics.sacrebleu.bleu import Bleu_sacrebleu
from text_metrics_wrapper.metrics.sacrebleu.chrf import Chrf_sacrebleu
from text_metrics_wrapper.metrics.sacrebleu.ter import Ter_sacrebleu
from text_metrics_wrapper.metrics.nltk.bleu import Bleu_nltk


metric_to_function_map = {
    "Parent": Parent,
    "Bleurt": Bleurt,
    "Gem": Gem,
    "Bleu_sacrebleu": Bleu_sacrebleu,
    "Chrf_sacrebleu": Chrf_sacrebleu,
    "Ter_sacrebleu": Ter_sacrebleu,
    "Bleu_nltk": Bleu_nltk,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references", type=str, help="The path to the strings that are going to be used as reference text."
    )
    parser.add_argument(
        "--hypothesis", type=str, help="The path to the strings that are going to be used as system generated text."
    )
    parser.add_argument(
        "--tokenizer", type=str, default="identity", help="The name of the tokenizer to use during preprocessing."
    )
    parser.add_argument("--tables", type=str, default=None, help="The path to the tables.")
    parser.add_argument("--metric", type=str, default="", help="The name of the metric to compute.")
    parser.add_argument("--o", type=str, default="", help="The path to the output file")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the model checkpoint")
    parser.add_argument("--method", type=str, default=None, help="The modality of the metric")
    parser.add_argument("--metrics_list", type=str, default=None, help="List of metrics to compute with GEM.")
    parser.add_argument("--bleu_n", type=int, default=None, help="The n-gram order for BLEU.")
    parser.add_argument(
        "--return_all_scores",
        type=bool,
        default=False,
        help="Whether to return all scores or just the average, not available for all metrics.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)

    references = load_jsonl_file(args.references)
    hypothesis = load_jsonl_file(args.hypothesis)

    prep_references = preprocess_text_blocks(references, args.tokenizer)
    prep_hypothesis = preprocess_text_blocks(hypothesis, args.tokenizer)

    kwargs = {}
    if args.tables is not None:
        kwargs["tables"] = load_jsonl_file(args.tables)
    if args.checkpoint is not None:
        kwargs["checkpoint"] = args.checkpoint
    if args.method is not None:
        kwargs["method"] = args.method
    if args.metrics_list is not None:
        kwargs["metrics_list"] = args.metrics_list
    if args.bleu_n is not None:
        kwargs["bleu_n"] = args.bleu_n
    kwargs["return_all_scores"] = args.return_all_scores

    metric_function = metric_to_function_map[args.metric]

    results = metric_function(prep_hypothesis, prep_references, **kwargs)

    write_json_file(args.o, results, overwrite=True)


if __name__ == "__main__":
    main()
