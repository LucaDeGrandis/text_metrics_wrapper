from text_metrics_wrapper.utils.manage_jsonl_files import load_jsonl_file
from text_metrics_wrapper.utils.manage_json_files import write_json_file
import shutil
import os


def Gem(hypothesis: List[str], references: Union[List[str], List[List[str]]], tables: List[List[Dict[str, Any]]], **kwargs)
) -> Tuple[float, float, float]:
    print(f"Computing GEM metrics...")
    
    # Create a hidden directory
    text_metrics_wrapper_dir = os.environ.get("TEXT_METRICS_WRAPPER_DIR")
    text_metrics_wrapper_dir = os.path.abspath(text_metrics_wrapper_dir)
    temp_dir = os.path.join(text_metrics_wrapper_dir, ".temporary")
    os.mkdir(temp_dir)

    # Write files to the hidden directory
    write_json_file(
        'predictions.json',
        hypothesis
    )
    write_json_file(
        'references.json',
        references
    )

    # Move to the GEM directory
    saved_dir = os.getcwd()
    gem_dir = os.path.join(text_metrics_wrapper_dir, 'GEM-metrics')
    os.chdir(gem_dir)

    # Compute the metrics
    code = f'./run_metrics.py {temp_dir}/predictions.json -r {temp_dir}/references.json -o "{kwargs["o"]}" --metric-list bleu meteor ter rouge bertscore'
    exec(code)

    # Go back to the original directory
    os.chdir(saved_dir)


# Remove the hidden directory
shutil.rmtree(temp_dir)
