"""compare_engines: statistics between two reflectance LUTs on the same grid, exit codes."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from spires_lut.io import write_lut

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "compare_engines.py"


@pytest.fixture(scope="module")
def ce():
    spec = importlib.util.spec_from_file_location("compare_engines", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stats_against_shifted_copy(ce, hyperspectral_lut):
    b = hyperspectral_lut.copy(deep=True)
    b["reflectance"] = b["reflectance"] + np.float32(0.01)
    stats = ce.compare(hyperspectral_lut, b)

    assert stats["overall"]["bias"] == pytest.approx(0.01, abs=1e-6)
    assert stats["overall"]["rmse"] == pytest.approx(0.01, abs=1e-6)
    assert stats["overall"]["max_abs"] == pytest.approx(0.01, abs=1e-6)
    assert set(stats["region"]) == {"VIS 400-700", "NIR 700-1400", "SWIR 1400-2500"}
    assert set(stats["solar_angle"]) == {0.0, 30.0}
    for row in list(stats["region"].values()) + list(stats["solar_angle"].values()):
        assert row["bias"] == pytest.approx(0.01, abs=1e-6)


def test_main_exits_nonzero_on_grid_mismatch(ce, hyperspectral_lut, tmp_path, monkeypatch):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    write_lut(hyperspectral_lut, a)
    write_lut(hyperspectral_lut.isel(sqrt_grain_radius=slice(0, 2)), b)
    monkeypatch.setattr(sys, "argv", ["compare_engines.py", str(a), str(b)])
    with pytest.raises(SystemExit) as exc:
        ce.main()
    assert exc.value.code == 2


def test_main_prints_table_and_exits_zero(ce, hyperspectral_lut, tmp_path, monkeypatch, capsys):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    write_lut(hyperspectral_lut, a)
    write_lut(hyperspectral_lut, b)
    monkeypatch.setattr(sys, "argv", ["compare_engines.py", str(a), str(b)])
    assert ce.main() == 0
    out = capsys.readouterr().out
    assert "overall" in out and "VIS 400-700" in out and "solar_angle=30" in out
