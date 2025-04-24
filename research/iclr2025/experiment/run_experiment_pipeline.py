import argparse
import itertools
import subprocess
import sys

import rich
from rich.panel import Panel

LR_MODEL_NAME = "logreg"
GEN_DB_ACTION_SET = "complex_nD"


def main():
    parser = argparse.ArgumentParser(description="run the experimental pipeline.")
    parser.add_argument("--data_name", default="german")
    parser.add_argument("--action_set_name", default="complex_nD")
    parser.add_argument("--methods", nargs="+", type=str, default=["reach"])
    parser.add_argument(
        "--models", nargs="+", type=str, default=[LR_MODEL_NAME, "rf", "xgb"]
    )
    parser.add_argument(
        "--explainers",
        nargs="+",
        type=str,
        default=["SHAP", "LIME", "SHAP_actionAware", "LIME_actionAware"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "db", "train", "explain", "audit", "score", "metrics"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore_errors", default=False, action="store_true")
    args = parser.parse_args()

    pipeline = []
    if "setup" in args.stages:
        pipeline.append(
            f"python experiment/setup_dataset_actionset_{args.data_name}.py"
        )

    if "db" in args.stages:
        if args.action_set_name == GEN_DB_ACTION_SET:
            pipeline.append(
                f"python experiment/generate_reachable_sets.py --data_name={args.data_name} --action_set_name={args.action_set_name} {'--overwrite' if args.overwrite else ''}"
            )

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

    if "audit" in args.stages:
        for method, model in itertools.product(args.methods, args.models):
            pipeline.append(
                f"python experiment/run_audit.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} --method_name={method}",
            )

    if "plot" in args.stages:
        for model, explainer in itertools.product(args.models, args.explainers):
            pipeline.append(
                f"python experiment/generate_plotting_data.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} --explainer_type={explainer}"
            )

    if "score" in args.stages:
        for model in args.models:
            pipeline.append(
                f"python experiment/calc_resp.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} {'--overwrite' if args.overwrite else ''}"
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
