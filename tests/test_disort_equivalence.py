"""The DISORT wrapper must see exactly the mixture optics.snow_lap_mixture returns."""
from pathlib import Path

import numpy as np
import pytest

disort = pytest.importorskip("disort")

from spires_lut import optics                     # noqa: E402
from spires_lut.disort_wrapper import DISORT      # noqa: E402

pytestmark = pytest.mark.skipif(not optics.OP_DIR.exists(), reason="biosnicar optical-property data not downloaded")

# 3 layers, LAP + algae + liquid water, hexagonal plates: exercises every branch of the mixture code
THREE_LAYER = dict(
    rds_snw=[100.0, 300.0, 800.0],
    rho_snw=[150.0, 300.0, 400.0],
    dz=[0.01, 0.1, 5.0],
    lap_type="dust_skiles_size3",
    mss_cnc_lap=[1000.0, 50.0, 0.0],
    mss_cnc_snw_alg=[1e6, 0.0, 0.0],
    lw_frac=[10.0, 5.0, 0.0],
    grain_shape=3,
)


def test_disort_attributes_equal_snow_lap_mixture():
    # lw_frac is converted in place by both call paths, so each gets its own copy
    model = DISORT(sza=30.0, saa=0.0, vza=0.0, vaa=0.0, e_dir=1.0, e_dif=0.0, nbr_lyr=3,
                   **{**THREE_LAYER, "lw_frac": list(THREE_LAYER["lw_frac"])})
    mix = optics.snow_lap_mixture(**{**THREE_LAYER, "lw_frac": list(THREE_LAYER["lw_frac"])})

    np.testing.assert_array_equal(model.SSA, mix.ssa)
    np.testing.assert_array_equal(model.g, mix.g)
    np.testing.assert_array_equal(model.tau, mix.tau)
    np.testing.assert_array_equal(model.wvl, mix.wavelength_um)
    assert model.DTAUC is model.tau and model.SSALB is model.SSA   # multi-stream: no delta-scaling
    assert mix.ssa.shape == (3, 475)


def test_mixture_matches_pre_refactor_golden():
    """Arrays recorded from DISORT.__init__ at commit 182476a (before the mixture moved to optics)."""
    golden = np.load(Path(__file__).parent / "data" / "mixture_golden_3layer_182476a.npz")
    mix = optics.snow_lap_mixture(**{**THREE_LAYER, "lw_frac": list(THREE_LAYER["lw_frac"])})
    np.testing.assert_array_equal(mix.ssa, golden["ssa"])
    np.testing.assert_array_equal(mix.g, golden["g"])
    np.testing.assert_array_equal(mix.tau, golden["tau"])
    np.testing.assert_array_equal(mix.wavelength_um, golden["wvl"])


def test_disort_rejects_layer_count_mismatch():
    with pytest.raises(ValueError, match="nbr_lyr"):
        DISORT(sza=30.0, saa=0.0, vza=0.0, vaa=0.0, e_dir=1.0, e_dif=0.0, nbr_lyr=2,
               **{**THREE_LAYER, "lw_frac": list(THREE_LAYER["lw_frac"])})


def test_delta_eddington_tracks_disort_at_one_point():
    """One 2 s DISORT run vs the closed form; tolerance to be tightened once the gap is known."""
    from spires_lut import simulate, two_stream

    point = dict(sza=30.0, vza=0.0, raa=0.0, n_layers=1, grain_shape=1, grain_radius=[300.0],
                 lap_concentration=[0.0], algae_concentration=[0.0], lwc=[0.0], snow_density=[300.0],
                 dz=[100.0], lap_type="dust_skiles_size3")
    _, _, r_disort, _, wl_d = simulate.run_point(white_sky=False, **point)
    _, _, r_de, _, wl_t = two_stream.run_point(**point)

    np.testing.assert_array_equal(wl_d, wl_t)
    band = (wl_d >= 400.0) & (wl_d <= 2500.0)
    max_abs = float(np.max(np.abs(r_de[band] - r_disort[band])))
    print(f"\nmax |delta-Eddington - DISORT| over 400-2500 nm: {max_abs:.4f}")
    assert max_abs < 0.03
