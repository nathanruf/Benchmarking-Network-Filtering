import os
from typing import List, Optional

def list_graph_files(
    base_path: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    List graph files from a directory.

    Args:
        base_path (str): Base directory to search.
        extensions (list[str], optional): File extensions to filter (e.g., ['.pickle', '.gml']).
                                          If None, returns all files.
        recursive (bool, optional): Whether to search recursively. Defaults to True.

    Returns:
        list[str]: List of file paths.
    """
    file_paths = []

    if extensions is not None:
        extensions = [ext.lower() for ext in extensions]

    if recursive:
        for root, _, files in os.walk(base_path):
            for file in files:
                filepath = os.path.join(root, file)

                if extensions:
                    if os.path.splitext(file)[1].lower() in extensions:
                        file_paths.append(filepath)
                else:
                    file_paths.append(filepath)
    else:
        for file in os.listdir(base_path):
            filepath = os.path.join(base_path, file)

            if not os.path.isfile(filepath):
                continue

            if extensions:
                if os.path.splitext(file)[1].lower() in extensions:
                    file_paths.append(filepath)
            else:
                file_paths.append(filepath)

    return file_paths