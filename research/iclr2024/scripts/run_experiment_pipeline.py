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
    parser.add_argument(
        "--methods", nargs="+", type=str, default=["reach", "ar", "dice"]
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=[LR_MODEL_NAME, "rf", "xgb"]
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["setup", "db", "train", "baselines", "audit", "stats"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore_errors", default=False, action="store_true")
    args = parser.parse_args()

    pipeline = []
    if "setup" in args.stages:
        pipeline.append(f"python scripts/setup_dataset_actionset_{args.data_name}.py")
        # todo: pipeline.append(f"python scripts/setup_dataset_actionset_{args.data_name}.py")

    if "db" in args.stages:
        if args.action_set_name == GEN_DB_ACTION_SET:
            pipeline.append(
                f"python scripts/generate_reachable_sets.py --data_name={args.data_name} --action_set_name={args.action_set_name} {'--overwrite' if args.overwrite else ''}"
            )

    if "train" in args.stages:
        for model in args.models:
            pipeline.append(
                f"python scripts/train_models.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model}"
            )

    if "baselines" in args.stages:
        baseline_methods = [m for m in args.methods if m != "reach"]
        for method in baseline_methods:
            for model in args.models:
                if method == "ar" and model != LR_MODEL_NAME:
                    continue
                else:
                    pipeline.append(
                        f"python scripts/run_{method}.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model}"
                    )

    if "audit" in args.stages:
        for method, model in itertools.product(args.methods, args.models):
            if method == "ar" and model != LR_MODEL_NAME:
                continue
            else:
                pipeline.append(
                    f"python scripts/run_audit.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} --method_name={method}",
                )

    if "stats" in args.stages:
        for method, model in itertools.product(args.methods, args.models):
            if method == "ar" and model != LR_MODEL_NAME:
                continue
            else:
                pipeline.append(
                    f"python scripts/compute_stats.py --data_name={args.data_name} --action_set_name={args.action_set_name} --model_type={model} --method_name={method}",
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
