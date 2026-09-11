"""Overlay Raman spectrum, ground-truth baseline, and extracted prediction.

Which spectra to plot comes from the extract manifest, not a second list.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXTRACT = Path(r"C:\HypercubeCascade\RamanModels\extracted")
OUT = Path(__file__).with_name("extracted_baselines.png")
N = 2048


def load_row(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"missing {path} (run cascade_raman_extract first)")
    return np.fromstring(path.read_text(encoding="ascii").strip(), sep=",", dtype=np.float64)


def read_manifest(path: Path) -> tuple[Path, tuple[int, ...]]:
    split = None
    indices: tuple[int, ...] | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("split="):
            split = Path(line.split("=", 1)[1].strip())
        elif line.startswith("indices="):
            raw = line.split("=", 1)[1].strip()
            if not raw:
                raise SystemExit(f"empty indices= in {path}")
            indices = tuple(int(p) for p in raw.split(","))
    if split is None or indices is None:
        raise SystemExit(f"manifest must have split= and indices= ({path})")
    return split, indices


def load_axis(split: Path, n: int) -> np.ndarray:
    path = split / "xaxis.txt"
    if not path.is_file():
        return np.arange(n, dtype=np.float64)
    x = load_row(path)
    if x.size != n:
        raise SystemExit(f"xaxis length {x.size} != {n}")
    return x


def main() -> None:
    manifest = EXTRACT / "manifest.txt"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest} (run cascade_raman_extract first)")
    split, indices = read_manifest(manifest)

    first = load_row(split / f"{indices[0]}.data.txt")
    if first.size != N:
        raise SystemExit(f"spectrum length {first.size} != {N}")
    x = load_axis(split, first.size)

    fig, axes = plt.subplots(
        len(indices), 1, sharex=True, figsize=(10, 2.6 * len(indices))
    )
    if len(indices) == 1:
        axes = [axes]

    fig.suptitle("Raman baseline extract (spectrum + label + pred)")
    split_name = split.name

    for ax, idx in zip(axes, indices):
        spec = load_row(split / f"{idx}.data.txt")
        base = load_row(split / f"{idx}.label.txt")
        pred = load_row(EXTRACT / f"{idx}.pred.txt")
        if spec.size != x.size or base.size != x.size or pred.size != x.size:
            raise SystemExit(
                f"length mismatch at {idx}: x={x.size} data={spec.size} "
                f"label={base.size} pred={pred.size}"
            )
        ax.plot(x, spec, color="0.25", lw=0.9, label="spectrum")
        ax.plot(x, base, color="C3", lw=1.2, label="baseline")
        ax.plot(x, pred, color="C0", lw=1.2, label="extracted")
        ax.set_ylabel("amplitude")
        ax.set_title(f"{split_name}/{idx}")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(
        "wavenumber" if (split / "xaxis.txt").is_file() else "bin"
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
