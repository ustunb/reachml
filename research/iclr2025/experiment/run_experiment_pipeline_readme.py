import argparse
import itertools
import subprocess
import sys

import rich
from rich.panel import Panel

LR_MODEL_NAME = "logreg"
GEN_DB_ACTION_SET = "complex_nD"


def main():
    pipeline = []
    pipeline.append(
        f"python experiment/action_set_setup_readme.py"
    )
    pipeline.append(
        f"python experiment/generate_reachable_sets.py --data_name=readme_example --action_set_name=readme_example_action_set --overwrite"
    )

    pipeline.append(
        f"python experiment/train_models.py --data_name=readme_example --action_set_name=readme_example_action_set --model_type=logreg"
    )

    pipeline.append(
        f"python experiment/run_audit.py --data_name=readme_example --action_set_name=readme_example_action_set --model_type=logreg --method_name=reach",
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
            break

    sys.exit(failed)


if __name__ == "__main__":
    main()
