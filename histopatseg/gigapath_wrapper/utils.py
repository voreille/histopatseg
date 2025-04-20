from pathlib import Path
import subprocess


def run_gigapath_docker(wsi_path: Path, output_dir: Path, model_path: Path):
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{wsi_path.parent}:/input",
        "-v",
        f"{output_dir}:/output",
        "-v",
        f"{model_path.parent}:/model",
        "my-gigapath-cli",
        f"/input/{wsi_path.name}",
        "--output-dir",
        "/output",
        "--model-path",
        f"/model/{model_path.name}",
    ]
    subprocess.run(cmd, check=True)
