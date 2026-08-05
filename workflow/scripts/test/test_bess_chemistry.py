"""Tests for the BESS chemistry map and the two silent-wrong-answer traps it closes.

Both traps sit on the Li-vs-Na comparison path
(00_ADMIN/AB_Provenance_Audit_and_Strategy.md §3.9 items 3 and 5):

* Trap 5 — ``solve_network.update_bess_costs`` matched storage units by substring
  on ``carrier.str.contains("battery_storage")``. A ``4hr_battery_storage_naion``
  unit would have been included in the LFP update and silently received LFP's cost
  trajectory. This suite pins that :func:`chemistry_of` raises on unknown carriers
  and that :func:`carriers_for_tech` only returns carriers of that chemistry.

* Trap 3 — ``config.costs.itc_modifier`` was read with ``.get(carrier, 0.0)``.
  A new Na carrier with no entry silently got 0% ITC while LFP got 30%. This suite
  pins that :func:`assert_itc_covers_bess` fails loudly at startup for any
  extendable BESS carrier that has no explicit entry.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from opts.steer_carriers import (
    BESS_CHEMISTRY_BY_CARRIER,
    STEER_CONFIG_BY_TECH,
    assert_itc_covers_bess,
    carriers_for_tech,
    chemistry_of,
    enabled_chemistries,
    is_bess_carrier,
)


class TestChemistryOf:
    """Trap 5: no BESS carrier may pass through unclassified."""

    def test_known_lfp_carrier(self):
        assert chemistry_of("4hr_battery_storage") == "lfp"

    def test_known_na_carrier(self):
        assert chemistry_of("4hr_battery_storage_naion") == "na"

    def test_unknown_bess_carrier_raises(self):
        """An unrecognised BESS carrier is a bug, not a default-to-LFP situation."""
        with pytest.raises(KeyError, match="not in BESS_CHEMISTRY_BY_CARRIER"):
            chemistry_of("12hr_battery_storage_mystery")

    def test_typo_raises_not_defaults(self):
        """A typo (``naiom``) must fail — a silent default here is the whole trap."""
        with pytest.raises(KeyError):
            chemistry_of("4hr_battery_storage_naiom")


class TestCarriersForTech:
    """Trap 5: filtering storage units by tech must be exact, not by substring."""

    def test_lfp_carriers_do_not_include_naion(self):
        lfp = carriers_for_tech("lfp")
        for carrier in lfp:
            assert "naion" not in carrier, (
                f"carriers_for_tech('lfp') returned {carrier!r} which contains 'naion' — "
                f"this is exactly the substring-match bug the module exists to prevent."
            )

    def test_na_carriers_are_naion_only(self):
        na = carriers_for_tech("na")
        assert na, "expected at least one Na carrier configured"
        for carrier in na:
            assert "naion" in carrier

    def test_lfp_and_na_are_disjoint(self):
        assert set(carriers_for_tech("lfp")) & set(carriers_for_tech("na")) == set()


class TestIsBessCarrier:
    def test_battery_carriers_recognised(self):
        assert is_bess_carrier("4hr_battery_storage")
        assert is_bess_carrier("4hr_battery_storage_naion")

    def test_non_battery_ignored(self):
        assert not is_bess_carrier("solar")
        assert not is_bess_carrier("onwind")
        assert not is_bess_carrier("4hr_PHS")


class TestEnabledChemistries:
    def test_only_lfp_configured(self):
        assert enabled_chemistries(["4hr_battery_storage", "8hr_battery_storage"]) == ["lfp"]

    def test_only_na_configured(self):
        assert enabled_chemistries(["4hr_battery_storage_naion"]) == ["na"]

    def test_both_chemistries_returns_stable_order(self):
        got = enabled_chemistries(
            ["4hr_battery_storage_naion", "4hr_battery_storage", "8hr_battery_storage_naion"],
        )
        assert got[0] == "na"
        assert "lfp" in got

    def test_non_bess_carriers_ignored(self):
        assert enabled_chemistries(["4hr_PHS", "solar", "4hr_battery_storage"]) == ["lfp"]

    def test_every_enabled_chemistry_has_a_steer_config(self):
        """A chemistry the config asks for must have a STEER config to load."""
        for tech in enabled_chemistries(list(BESS_CHEMISTRY_BY_CARRIER)):
            assert tech in STEER_CONFIG_BY_TECH, (
                f"chemistry {tech!r} appears in BESS_CHEMISTRY_BY_CARRIER but has no "
                f"STEER config in STEER_CONFIG_BY_TECH — solve_network cannot load it."
            )


class TestAssertItcCoversBess:
    """Trap 3: no BESS carrier may enter optimisation without an explicit ITC entry."""

    def test_lfp_only_config_passes(self):
        itc = {"4hr_battery_storage": 0.3, "8hr_battery_storage": 0.3}
        assert_itc_covers_bess(itc, ["4hr_battery_storage", "8hr_battery_storage"])

    def test_na_carrier_without_itc_raises(self):
        itc = {"4hr_battery_storage": 0.3}
        with pytest.raises(ValueError, match=r"4hr_battery_storage_naion.*itc_modifier"):
            assert_itc_covers_bess(
                itc, ["4hr_battery_storage", "4hr_battery_storage_naion"],
            )

    def test_explicit_zero_is_accepted(self):
        """An explicit 0.0 is a valid config; only silence is rejected."""
        itc = {"4hr_battery_storage_naion": 0.0}
        assert_itc_covers_bess(itc, ["4hr_battery_storage_naion"])

    def test_non_bess_carriers_do_not_need_itc_entries(self):
        """Only BESS carriers get the check; gas plants legitimately have no ITC."""
        itc = {"4hr_battery_storage": 0.3}
        assert_itc_covers_bess(itc, ["4hr_battery_storage", "OCGT", "solar"])
