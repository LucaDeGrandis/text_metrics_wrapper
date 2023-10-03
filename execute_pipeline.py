import argparse
from text_metrics_wrapper.preprocessing.preprocess_text_blocks import preprocess_text_blocks
from text_metrics_wrapper.utils.manage_jsonl_files import load_jsonl_file
from text_metrics_wrapper.metrics.parent.parent import Parent


metric_to_function_map = {"Parent": Parent}


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
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)

    references = load_jsonl_file(args.references)
    hypothesis = load_jsonl_file(args.hypothesis)

    prep_hypothesis = preprocess_text_blocks(hypothesis, args.tokenizer)
    prep_hypothesis = preprocess_text_blocks(hypothesis, args.tokenizer)

    if args.tables is not None:
        tables = load_jsonl_file(args.tables)

    metric_function = metric_to_function_map[args.metric]

    results = metric_function(prep_hypothesis, references, tables)


if __name__ == "__main__":
    main()
