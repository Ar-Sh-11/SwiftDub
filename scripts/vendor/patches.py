"""Apply SwiftDub-owned patches to downloaded vendor snapshots.

Vendor code is downloaded once by scripts/download/models.py and stored under
models/vendor/ without a .git directory. Patches live in this repo only — never
edit vendor files directly in the upstream project repos.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "models" / "vendor"


def _replace_in_file(path: Path, old: str, new: str, *, label: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    if old not in text:
        if new in text:
            return False
        raise RuntimeError(f"Patch anchor missing for {label}: {path}")
    path.write_text(text.replace(old, new, 1))
    return True


def patch_latentsync() -> None:
    repo = VENDOR / "latentsync"
    inference = repo / "scripts" / "inference.py"
    if inference.exists() and "SWIFTDUB_LATENTSYNC_WHISPER" not in inference.read_text():
        text = inference.read_text()
        old = (
            "    if config.model.cross_attention_dim == 768:\n"
            "        whisper_model_path = \"checkpoints/whisper/small.pt\"\n"
            "    elif config.model.cross_attention_dim == 384:\n"
            "        whisper_model_path = \"checkpoints/whisper/tiny.pt\"\n"
            "    else:\n"
            "        raise NotImplementedError(\"cross_attention_dim must be 768 or 384\")"
        )
        new = (
            "    env_whisper = os.environ.get(\"SWIFTDUB_LATENTSYNC_WHISPER\")\n"
            "    if env_whisper:\n"
            "        whisper_model_path = env_whisper\n"
            "    elif config.model.cross_attention_dim == 768:\n"
            "        whisper_model_path = \"checkpoints/whisper/small.pt\"\n"
            "    elif config.model.cross_attention_dim == 384:\n"
            "        whisper_model_path = \"checkpoints/whisper/tiny.pt\"\n"
            "    else:\n"
            "        raise NotImplementedError(\"cross_attention_dim must be 768 or 384\")"
        )
        if old in text:
            inference.write_text(text.replace(old, new, 1))

    util = repo / "latentsync" / "utils" / "util.py"
    if util.exists() and "import shlex" not in util.read_text():
        text = util.read_text()
        text = text.replace(
            "import subprocess\n",
            "import shlex\nimport subprocess\n",
            1,
        )
        text = text.replace(
            'f"ffmpeg -y -v error -i {video_path}',
            'f"ffmpeg -y -v error -i {shlex.quote(str(video_path))}',
        )
        util.write_text(text)

    pipeline = repo / "latentsync" / "pipelines" / "lipsync_pipeline.py"
    if pipeline.exists() and "import shlex" not in pipeline.read_text():
        text = pipeline.read_text()
        text = text.replace(
            "import subprocess\n",
            "import shlex\nimport subprocess\n",
            1,
        )
        text = text.replace(
            'f"ffmpeg -y -v error -i {video_path}',
            'f"ffmpeg -y -v error -i {shlex.quote(str(video_path))}',
        )
        pipeline.write_text(text)


def patch_musetalk() -> None:
    prep = VENDOR / "musetalk" / "musetalk" / "utils" / "preprocessing.py"
    if not prep.exists():
        return
    text = prep.read_text()
    marker = "if average_range_minus and average_range_plus:"
    if marker in text:
        return
    old = (
        '    print(f"Total frame:「{len(frames)}」 Manually adjust range : '
        '[ -{int(sum(average_range_minus) / len(average_range_minus))}~'
        '{int(sum(average_range_plus) / len(average_range_plus))} ] , '
        'the current value: {upperbondrange}")'
    )
    new = (
        "    if average_range_minus and average_range_plus:\n"
        "        print(f\"Total frame:「{len(frames)}」 Manually adjust range : "
        "[ -{int(sum(average_range_minus) / len(average_range_minus))}~"
        "{int(sum(average_range_plus) / len(average_range_plus))} ] , "
        "the current value: {upperbondrange}\")\n"
        "    else:\n"
        "        print(f\"Total frame:「{len(frames)}」 Using face_alignment bbox "
        "(mmpose unavailable). bbox_shift: {upperbondrange}\")"
    )
    if old in text:
        prep.write_text(text.replace(old, new, 1))


def apply_all() -> None:
    patch_latentsync()
    patch_musetalk()


if __name__ == "__main__":
    apply_all()
    print("Vendor patches applied.")
