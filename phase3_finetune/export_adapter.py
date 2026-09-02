"""Unzips/validates the LoRA adapter downloaded from Colab
(colab_dpo_train.py's final cell) and places it at
phase3_finetune/final_adapter/, where app/detection/local_model_detector.py
expects to find it. Small folder (tens of MB) - not the full base model.
"""
import argparse
import os
import shutil
import zipfile

DEFAULT_TARGET = os.path.join(os.path.dirname(__file__), "final_adapter")
REQUIRED_FILES = {"adapter_config.json"}


def export_adapter(source_path: str, target_dir: str = DEFAULT_TARGET) -> str:
    if os.path.isdir(source_path):
        if os.path.abspath(source_path) != os.path.abspath(target_dir):
            shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
    elif zipfile.is_zipfile(source_path):
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(source_path) as zf:
            zf.extractall(target_dir)
    else:
        raise ValueError(f"{source_path} is neither a directory nor a zip file.")

    _flatten_single_nested_dir(target_dir)

    missing = REQUIRED_FILES - set(os.listdir(target_dir))
    if missing:
        raise RuntimeError(f"Adapter at {target_dir} is missing expected files: {missing}. Check the zip contents.")

    print(f"Adapter ready at {target_dir}")
    return target_dir


def _flatten_single_nested_dir(target_dir: str) -> None:
    """Colab's shutil.make_archive sometimes wraps output in a single extra
    folder - if extraction produced exactly one subdirectory and nothing
    else, hoist its contents up a level."""
    entries = os.listdir(target_dir)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(target_dir, e))]
    if len(entries) == 1 and len(subdirs) == 1:
        nested = os.path.join(target_dir, subdirs[0])
        for f in os.listdir(nested):
            shutil.move(os.path.join(nested, f), os.path.join(target_dir, f))
        os.rmdir(nested)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to the downloaded adapter .zip, or an already-unzipped folder.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()
    export_adapter(args.source, args.target)
