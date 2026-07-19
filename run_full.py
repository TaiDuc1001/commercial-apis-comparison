from __future__ import annotations

from run_test import DEFAULT_MODELS, RunConfig, rerun_invalid_results, run_evaluation


def main() -> None:
    run_evaluation(
        RunConfig(
            name="full_results",
            models=DEFAULT_MODELS,
            repetitions=1,
            task_limit=None,
            concurrency=12,
        )
    )


def rerun_invalid() -> None:
    rerun_invalid_results(
        RunConfig(
            name="full_results",
            models=DEFAULT_MODELS,
            repetitions=1,
            task_limit=None,
            concurrency=12,
        )
    )


if __name__ == "__main__":
    main()
