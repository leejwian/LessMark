"""Exercise the public transfer command against a small local rsync destination."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("rsync"), reason="requires rsync")


@pytest.fixture
def transfer(tmp_path):
    # Keep the actual CLI and rsync, substituting only the 86 GiB inventory.
    checkout = tmp_path / "clone"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO / "lessmark", checkout)
    shutil.copy2(REPO / "scripts/copy_inputs.sh", scripts)
    source = tmp_path / "source volume"
    images = source / "data"
    images.mkdir(parents=True)
    (images / "registered.png").write_bytes(b"registered")
    (images / "unused.png").write_bytes(b"unused")
    destination = tmp_path / "target volume"
    destination.mkdir()
    listing = scripts / "assets.py"
    listing.write_text(
        'import sys\nfrom pathlib import Path\n'
        'Path(sys.argv[sys.argv.index("--files-from") + 1]).write_text("data/registered.png\\n")\n'
    )

    def run(*args):
        env = dict(os.environ, LERRA_PYTHON=sys.executable, LERRA_DATA_ROOT=str(source))
        return subprocess.run([checkout / "lessmark", "copy-inputs", *args, destination],
                              capture_output=True, text=True, env=env)

    return run, destination, listing


def test_preview_and_copy_only_registered_files(transfer):
    run, destination, _ = transfer
    preview = run("--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert list(destination.iterdir()) == []
    copied = run()
    assert copied.returncode == 0, copied.stderr
    assert (destination / "data/registered.png").read_bytes() == b"registered"
    assert not (destination / "data/unused.png").exists()


def test_failed_inventory_cannot_copy_a_partial_list(transfer):
    run, destination, listing = transfer
    with listing.open("a") as script:
        script.write("raise SystemExit(3)\n")
    assert run().returncode == 3
    assert list(destination.iterdir()) == []
