import time
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable
from tqdm import tqdm

from netfilterbench.data_loader.loader import list_graph_files
from netfilterbench.experiments.experiment import run_experiment


def _generate_combinations(paths, filters, benchmarks, noise_levels, indicator_funcs):
    for path, f, b, n in itertools.product(paths, filters, benchmarks, noise_levels):
        yield (path, f, b, n, indicator_funcs)


def _run_wrapper(args):
    return run_experiment(*args)


def run_all(
    data_path: str,
    output_path: str,
    filters: List[Callable],
    benchmarks: List[Callable],
    noise_levels: List[float],
    indicator_funcs: List[Callable],
    supported_extensions=(".pickle",),
    recursive: bool = True,
    max_workers: int = 4,
):
    """
    Run all experiments for the given combinations of filters, benchmarks, and noise levels.
    Results are written to CSV incrementally as they arrive, without accumulating in memory.

    Args:
        data_path (str): Directory containing the graph files.
        output_path (str): Path to write the results CSV.
        filters (list[Callable]): Filter functions to benchmark.
        benchmarks (list[Callable]): Benchmark functions to use (e.g. NBF, NAF).
        noise_levels (list[float]): Noise levels to test.
        indicator_funcs (list[Callable]): Indicator functions to compute.
        supported_extensions (tuple, optional): File extensions to load. Defaults to ('.pickle',).
        recursive (bool, optional): Whether to search data_path recursively. Defaults to True.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.
    """
    start_time = time.time()

    paths = list_graph_files(
        base_path=data_path,
        extensions=supported_extensions,
        recursive=recursive,
    )

    total = len(paths) * len(filters) * len(benchmarks) * len(noise_levels)

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         open(output_path, 'w', newline='') as csvfile:

        writer = None

        for result in tqdm(
            executor.map(_run_wrapper, _generate_combinations(paths, filters, benchmarks, noise_levels, indicator_funcs)),
            total=total,
            desc="Running experiments",
        ):
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Execution time: {elapsed}")