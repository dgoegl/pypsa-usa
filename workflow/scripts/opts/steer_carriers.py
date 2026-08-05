"""Single source of truth: which BESS carrier is which chemistry.

Any storage-unit carrier whose name contains ``battery_storage`` MUST be listed here.
Two consumers rely on the mapping and both were silent-wrong-answer traps before it
existed (00_ADMIN/AB_Provenance_Audit_and_Strategy.md §3.9 items 3 and 5):

1. ``solve_network.update_bess_costs`` picked storage units by a substring match on
   ``carrier.str.contains("battery_storage")``. With Na-ion added as
   ``4hr_battery_storage_naion`` the mask matched Na too, so STEER would have quietly
   applied LFP's cost trajectory to Na units.
2. ``solve_network`` and ``add_extra_components.apply_itc`` both read
   ``config.costs.itc_modifier`` with ``.get(carrier, 0.0)``. A new Na carrier with no
   entry silently gets 0% ITC while LFP gets 30% — a hidden 30% thumb on the LFP scale
   in what is meant to be a symmetric comparison.

Both callers now use :func:`chemistry_of` (raises on unknown carriers) and
:func:`carriers_for_tech` (filters storage units by chemistry). Adding a new duration
or a new chemistry is one edit to :data:`BESS_CHEMISTRY_BY_CARRIER`; nothing else has
to change.
"""

from __future__ import annotations

#: Every BESS carrier the workflow knows about, mapped to the chemistry tech id
#: used by :class:`steer.experience.ExperienceState` (``"lfp"``, ``"na"``, ...).
BESS_CHEMISTRY_BY_CARRIER: dict[str, str] = {
    "2hr_battery_storage": "lfp",
    "4hr_battery_storage": "lfp",
    "6hr_battery_storage": "lfp",
    "8hr_battery_storage": "lfp",
    "10hr_battery_storage": "lfp",
    "4hr_battery_storage_naion": "na",
    "8hr_battery_storage_naion": "na",
}

#: STEER config file to load for each chemistry, relative to ``02_STEERMODEL/``.
STEER_CONFIG_BY_TECH: dict[str, str] = {
    "lfp": "config_li_ion.yaml",
    "na": "config_na_ion.yaml",
}


def is_bess_carrier(carrier: str) -> bool:
    """True if *carrier* names a battery-storage unit (any chemistry, any duration)."""
    return "battery_storage" in carrier


def chemistry_of(carrier: str) -> str:
    """The chemistry tech id for a BESS carrier.

    Raises
    ------
    KeyError
        If the carrier looks like a BESS carrier (contains ``battery_storage``) but is
        not classified. Adding a new duration or chemistry requires updating
        :data:`BESS_CHEMISTRY_BY_CARRIER`; a silent default here would let a mislabelled
        unit ride LFP's cost trajectory.
    """
    if carrier not in BESS_CHEMISTRY_BY_CARRIER:
        raise KeyError(
            f"BESS carrier {carrier!r} is not in BESS_CHEMISTRY_BY_CARRIER. "
            f"Add it to opts/steer_carriers.py before running — a silent default here "
            f"would apply the wrong chemistry's cost trajectory to this unit.",
        )
    return BESS_CHEMISTRY_BY_CARRIER[carrier]


def carriers_for_tech(tech: str) -> list[str]:
    """Every known carrier belonging to a given chemistry."""
    return [c for c, t in BESS_CHEMISTRY_BY_CARRIER.items() if t == tech]


def enabled_chemistries(extendable_storage_carriers: list[str]) -> list[str]:
    """Which chemistries are actually enabled in this run.

    Reads ``config.electricity.extendable_carriers.StorageUnit``; returns the unique
    set of chemistries in dependency order (``lfp`` before ``na``) so the caller can
    load STEER engines in a stable order.
    """
    seen: list[str] = []
    for carrier in extendable_storage_carriers:
        if not is_bess_carrier(carrier):
            continue
        tech = chemistry_of(carrier)
        if tech not in seen:
            seen.append(tech)
    return seen


def assert_itc_covers_bess(itc_modifier: dict, extendable_storage_carriers: list[str]) -> None:
    """Raise if any extendable BESS carrier has no ITC entry.

    ``config.costs.itc_modifier`` is read with ``.get(carrier, 0.0)`` by both consumers,
    so a missing entry means a silent 0% ITC. That is fine for non-BESS carriers that
    are genuinely not eligible for the credit, but for BESS it always represents a
    forgotten config edit, and it always tips the Li-vs-Na comparison. Fail loudly
    at startup instead of silently at optimisation time.
    """
    missing = [
        c for c in extendable_storage_carriers if is_bess_carrier(c) and c not in itc_modifier
    ]
    if missing:
        raise ValueError(
            f"BESS carriers {missing} are extendable but have no entry in "
            f"config.costs.itc_modifier. A missing entry silently gives them 0% ITC "
            f"and rigs the Li-vs-Na comparison. Add explicit entries (0.0 is a valid "
            f"choice, but must be explicit).",
        )
