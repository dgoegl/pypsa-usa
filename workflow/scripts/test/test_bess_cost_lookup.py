# ruff: noqa: D101, D102
"""Tests for the cost lookup in ``solve_network.update_bess_costs``.

The function used to continue with hard-coded defaults (wacc 0.055, fixed O&M
26.25 USD/kW-yr, lifetime 20 years) when the cost file or a battery carrier was
missing, so a run could finish with wrong battery costs and no error. These tests
pin that every such case now raises, and that the normal path is unchanged.

The harness is the smallest one that calls the real function: a one-bus PyPSA
network with one extendable battery, a cost CSV written to a temporary working
directory, and stand-ins for the cost engine and the experience state that return
fixed component costs.
"""

import os
import sys
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from solve_network import update_bess_costs

CARRIER = "4hr_battery_storage"
YEAR = 2030


class _Component:
    def __init__(self, name, scaling_factor, capex):
        self.config = SimpleNamespace(name=name, scaling_factor=scaling_factor)
        self._capex = capex

    def capex_us(self, year, x):
        return self._capex


class _Experience:
    def x_for(self, comp, tech):
        return 1.0

    def __str__(self):
        return "fixed"


# Energy component: 400 / 4.0 = 100 USD/kWh, times 4 h = 400 USD/kW.
# Power component: 200 / 1.0 = 200 USD/kW. Total overnight cost: 600 USD/kW.
ENGINE = SimpleNamespace(components=[_Component("cell", 4.0, 400.0), _Component("pcs", 1.0, 200.0)])


def _network():
    n = pypsa.Network()
    n.add("Bus", "b0")
    n.add(
        "StorageUnit",
        "b0 battery",
        bus="b0",
        carrier=CARRIER,
        max_hours=4.0,
        p_nom_extendable=True,
        build_year=YEAR,
        lifetime=20.0,
        capital_cost=1.0,
    )
    return n


def _write_costs(tmp_path, rows):
    path = tmp_path / "resources" / "costs"
    path.mkdir(parents=True)
    pd.DataFrame(rows, columns=["pypsa-name", "parameter", "value"]).to_csv(
        path / f"costs_{YEAR}.csv",
        index=False,
    )


CONFIG = {"costs": {"itc_modifier": {CARRIER: 0.3}}}


class TestMissingCostData:
    def test_missing_cost_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"resources/costs/costs_2030\.csv"):
            update_bess_costs(_network(), YEAR, _Experience(), ENGINE, "lfp", CONFIG)

    def test_missing_carrier_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_costs(tmp_path, [["solar", "wacc_real", 0.05], ["solar", "opex_fixed_per_kw", 10.0]])
        with pytest.raises(KeyError, match=rf"{CARRIER}.*{YEAR}.*costs_{YEAR}\.csv"):
            update_bess_costs(_network(), YEAR, _Experience(), ENGINE, "lfp", CONFIG)

    def test_missing_parameter_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_costs(tmp_path, [[CARRIER, "wacc_real", 0.05]])
        with pytest.raises(KeyError, match=rf"opex_fixed_per_kw.*{CARRIER}.*{YEAR}"):
            update_bess_costs(_network(), YEAR, _Experience(), ENGINE, "lfp", CONFIG)

    def test_empty_value_raises(self, tmp_path, monkeypatch):
        """A blank cell pivots to NaN; it must not flow into the capital cost."""
        monkeypatch.chdir(tmp_path)
        _write_costs(
            tmp_path,
            [[CARRIER, "wacc_real", None], [CARRIER, "opex_fixed_per_kw", 10.0], ["solar", "wacc_real", 0.05]],
        )
        with pytest.raises(ValueError, match=rf"wacc_real.*{CARRIER}.*{YEAR}"):
            update_bess_costs(_network(), YEAR, _Experience(), ENGINE, "lfp", CONFIG)

    def test_non_numeric_value_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_costs(tmp_path, [[CARRIER, "wacc_real", 0.05], [CARRIER, "opex_fixed_per_kw", "tbd"]])
        with pytest.raises(ValueError, match=rf"opex_fixed_per_kw.*{CARRIER}.*not a number"):
            update_bess_costs(_network(), YEAR, _Experience(), ENGINE, "lfp", CONFIG)


class TestNormalPath:
    def test_capital_cost_matches_hand_computation(self, tmp_path, monkeypatch):
        """wacc 0.05, lifetime 20 y, 600 USD/kW overnight, fixed O&M 10 USD/kW-yr, ITC 30%."""
        monkeypatch.chdir(tmp_path)
        _write_costs(tmp_path, [[CARRIER, "wacc_real", 0.05], [CARRIER, "opex_fixed_per_kw", 10.0]])
        n = _network()
        update_bess_costs(n, YEAR, _Experience(), ENGINE, "lfp", CONFIG)

        annuity = 0.05 / (1.0 - (1.0 + 0.05) ** (-20.0))  # 0.0802425872...
        itc_factor = 1.0 - ((1.0 - 0.1) * 0.3)  # 0.73
        expected = (annuity * 600.0 * 1e3 + 10.0 * 1e3) * itc_factor - 0.0
        got = n.storage_units.at["b0 battery", "capital_cost"]
        assert got == expected  # same arithmetic as the function: bit-for-bit
        assert got == pytest.approx(42446.25, abs=0.01)  # 48145.55 + 10000 = 58145.55 USD/MW-yr, times 0.73
