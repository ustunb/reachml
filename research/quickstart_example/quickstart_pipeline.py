import argparse
import itertools
import subprocess
import sys

import rich
from rich.panel import Panel

LR_MODEL_NAME = "logreg"
GEN_DB_ACTION_SET = "complex_nD"


def main():
    parser = argparse.ArgumentParser(description="run the experimental pipeline (db -> train -> audit).")
    parser.add_argument("--data_name", default="german")
    parser.add_argument("--action_set_name", default="complex_nD")
    parser.add_argument("--methods", nargs="+", type=str, default=["reach"])
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=[LR_MODEL_NAME],  # you said "train the model only" so default to one
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore_errors", default=False, action="store_true")
    args = parser.parse_args()

    pipeline = []

    # 1) setup dataset + action set
    pipeline.append(
        f"python experiment/setup_dataset_actionset_{args.data_name}.py"
    )

    # 2) generate reachable sets (db stage)
    if args.action_set_name == GEN_DB_ACTION_SET:
        pipeline.append(
            f"python experiment/generate_reachable_sets.py "
            f"--data_name={args.data_name} "
            f"--action_set_name={args.action_set_name} "
            f"{'--overwrite' if args.overwrite else ''}"
        )

    # 3) train model(s)
    for model in args.models:
        pipeline.append(
            f"python experiment/train_models.py "
            f"--data_name={args.data_name} "
            f"--action_set_name={args.action_set_name} "
            f"--model_type={model}"
        )

    # 4) audit model(s) with method(s)
    for method, model in itertools.product(args.methods, args.models):
        pipeline.append(
            f"python experiment/run_audit.py "
            f"--data_name={args.data_name} "
            f"--action_set_name={args.action_set_name} "
            f"--model_type={model} "
            f"--method_name={method}"
        )

    # run commands
    failed = False
    for command in pipeline:
        rich.print(Panel(f"[bold]{command}[/bold]"))
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )
        except KeyboardInterrupt:
            failed = True
            break
        except Exception:
            failed = True
            if not args.ignore_errors:
                break

    sys.exit(failed)


if __name__ == "__main__":
    main()
