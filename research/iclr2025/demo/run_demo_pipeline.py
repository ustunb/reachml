import argparse
import itertools
import subprocess
import sys

import rich
from rich.panel import Panel


def main():
    parser = argparse.ArgumentParser(description="run the experimental pipeline.")
    parser.add_argument("--data_name", default="givemecredit_cts")
    parser.add_argument("--action_set_name", default="complex_nD")
    parser.add_argument("--models", nargs="+", type=str, default=["logreg", "xgb"])
    parser.add_argument(
        "--explainers",
        nargs="+",
        type=str,
        default=["SHAP", "LIME", "SHAP_actionAware", "LIME_actionAware"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "train", "explain", "score"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore_errors", default=False, action="store_true")
    args = parser.parse_args()

    pipeline = []
    if "setup" in args.stages:
        pipeline.append(f"python demo/setup_dataset_actionset_{args.data_name}.py")

    if "train" in args.stages:
        for model in args.models:
            pipeline.append(
                f"python experiment/train_models.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model}"
            )

    if "explain" in args.stages:
        for model, explainer in itertools.product(args.models, args.explainers):
            pipeline.append(
                f"python experiment/explain_models.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} --explainer_type={explainer} {'--overwrite' if args.overwrite else ''}",
            )

    if "score" in args.stages:
        for model in args.models:
            pipeline.append(
                f"python demo/calc_resp.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} {'--overwrite' if args.overwrite else ''}"
            )

    # Run each command in the list
    failed = False
    for command in pipeline:
        rich.print(Panel(f"[bold]{command}[/bold]"))
        try:
            subprocess.run(
                command, shell=True, check=True, text=True, capture_output=False
            )
        except KeyboardInterrupt:
            failed = True
            break
        except:
            failed = True
            if not args.ignore_errors:
                break

    sys.exit(failed)


if __name__ == "__main__":
    main()
