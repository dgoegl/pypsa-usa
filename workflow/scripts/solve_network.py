"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import (
    configure_logging,
    resolve_period_value,
    update_config_from_wildcards,
)
from opts.bidirectional_link import add_bidirectional_link_constraints
from opts.interchange import add_interchange_constraints
from opts.land import add_land_use_constraints
from opts.policy import (
    add_regional_co2limit,
    add_RPS_constraints,
    add_RPS_constraints_sector,
    add_technology_capacity_target_constraints,
)
from opts.reserves import (
    add_ERM_constraints,
    add_operational_reserve_margin,
    store_ERM_duals,
)
from opts.sector import (
    add_cooling_heat_pump_constraints,
    add_demand_response_constraint,
    add_ev_generation_constraint,
    add_fossil_generation_constraint,
    add_gshp_capacity_constraint,
    add_ng_import_export_limits,
    add_sector_co2_constraints,
    add_sector_demand_response_constraints,
    add_water_heater_constraints,
)
from opts.steer_carriers import (
    STEER_CONFIG_BY_TECH,
    assert_itc_covers_bess,
    carriers_for_tech,
    enabled_chemistries,
)

logger_gurobi = logging.getLogger("gurobipy")
logger_gurobi.propagate = False

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def calculate_annuity(n_years, interest_rate):
    """Calculate the annuity factor for a lifetime n_years and discount rate interest_rate."""
    if interest_rate == 0:
        return 1.0 / n_years
    return interest_rate / (1.0 - (1.0 + interest_rate) ** (-n_years))


def update_bess_costs(n, planning_horizon, experience, sys_engine, tech, config):
    """
    Computes new BESS costs using the STEER engine and updates the network
    in-memory for extendable units in the current planning horizon.

    `experience` is a steer.experience.ExperienceState; each component is evaluated
    against the deployment pool it actually learns from — its own chemistry's for cells,
    all chemistries summed for pack/PCS/BoS/EPC, in GWh or GW as it declares. `tech`
    names the chemistry these units are ("lfp"). See steer/experience.py.
    """
    # 1. Load the corresponding year's costs from the correct run directory
    run_name = config.get("run", {}).get("name", "Default")
    possible_paths = [
        f"resources/{run_name}/costs/costs_{planning_horizon}.csv",
        f"resources/costs/costs_{planning_horizon}.csv",
    ]

    costs_df = None
    for cost_file in possible_paths:
        try:
            costs_df = pd.read_csv(cost_file)
            logger.info(f"Successfully loaded cost data from {cost_file}")
            break
        except FileNotFoundError:
            continue

    if costs_df is not None:
        costs = costs_df.pivot(index="pypsa-name", columns="parameter", values="value")
    else:
        logger.warning(
            f"Could not load cost data for horizon {planning_horizon} from any of {possible_paths}. Using fallback default factors.",
        )
        # fallback defaults
        costs = pd.DataFrame(columns=["wacc_real", "opex_fixed_per_kw", "lifetime"])

    # 2. Update BESS units — filter by *this chemistry's* carriers only. A
    # substring match on "battery_storage" would sweep in Na-ion units (their
    # carrier is 4hr_battery_storage_naion) and silently apply LFP's cost
    # trajectory to them. See opts/steer_carriers.py and audit §3.9 item 5.
    my_carriers = carriers_for_tech(tech)
    bess_mask = (
        n.storage_units.carrier.isin(my_carriers)
        & (n.storage_units.build_year == planning_horizon)
        & (n.storage_units.p_nom_extendable)
    )

    if not bess_mask.any():
        logger.info(f"No extendable BESS units found to update for horizon {planning_horizon}")
        return

    bess_units = n.storage_units[bess_mask]

    # We will log a comparison table
    comparison_rows = []

    for idx, row in bess_units.iterrows():
        carrier = row.carrier
        duration = float(row.max_hours)

        # Look up parameters from the cost file
        try:
            wacc = float(costs.at[carrier, "wacc_real"])
            fom = float(costs.at[carrier, "opex_fixed_per_kw"])
            lifetime = float(row.lifetime)  # read directly from network component
        except KeyError:
            # default fallback values if carrier not in costs
            wacc = 0.055
            fom = 26.25  # $/kW-year (approx 2.5% of $1050/kW)
            lifetime = 20.0

        # Evaluate overnight CAPEX components using the STEER system engine
        # We need Cell and Pack (energy components, config.scaling_factor == 4.0) scaled by duration,
        # and PCS, BoS, EPC (power components, config.scaling_factor == 1.0) scaled by 1.0.
        capex_components = {}
        for comp in sys_engine.components:
            # Compute native cost at component level, against the deployment this
            # component actually learns from (own chemistry for cells, all chemistries
            # for the rest; GWh or GW as declared).
            x_local = experience.x_for(comp, tech)
            native_cost = comp.capex_us(planning_horizon, x_local) / comp.config.scaling_factor
            if comp.config.scaling_factor > 1.0:
                # Energy component
                capex_components[comp.config.name] = native_cost * duration
            else:
                # Power component
                capex_components[comp.config.name] = native_cost

        # Sum total overnight CAPEX in $/kW
        total_capex_per_kw = sum(capex_components.values())

        # Calculate annualized CAPEX + FOM in $/MW-year
        annuity = calculate_annuity(lifetime, wacc)
        annualized_capex_per_mw = annuity * total_capex_per_kw * 1e3

        # Apply the Investment Tax Credit (ITC) modifier with a 10% monetization cost haircut.
        # A missing entry for a BESS carrier here was previously the second silent trap in
        # the Li-vs-Na comparison (audit §3.9 item 3) — .get(carrier, 0.0) would silently
        # give a new Na carrier 0% ITC while LFP got 30%. assert_itc_covers_bess (called
        # at STEER init below) fails loudly at startup instead; this .get default now only
        # ever fires for non-BESS carriers that don't qualify for the credit.
        itc_modifier = config.get("costs", {}).get("itc_modifier", {})
        itc_value = itc_modifier.get(carrier, 0.0)
        monetization_cost = 0.1
        itc_factor = 1.0 - ((1.0 - monetization_cost) * itc_value)

        steer_cost_per_mw_year = (annualized_capex_per_mw + fom * 1e3) * itc_factor

        # Re-apply the exogenous storage revenue offset that apply_storage_revenue applied at
        # network-build time. STEER's overwrite here would otherwise silently discard that
        # offset (bug found 2026-08-18: R11M-v1 and all children ran with revenue effectively
        # zero on STEER-on paths, because this line replaced capital_cost with STEER's raw
        # cost). Applying it again here keeps STEER-on and STEER-off paths symmetric in
        # revenue treatment, per Findings_2026-08-18.md.
        elec_cfg = config.get("electricity", {})
        cap_rev = elec_cfg.get("storage_capacity_revenue", {}) or {}
        anc_rev = elec_cfg.get("storage_ancillary_revenue", {}) or {}
        # Per-period credits (dict values) resolve at this vintage's planning
        # horizon; the bess_mask above already restricts to build_year ==
        # planning_horizon, keeping this symmetric with apply_storage_revenue's
        # per-build_year resolution on the STEER-off path.
        revenue_kwyr = resolve_period_value(
            cap_rev.get(carrier, 0),
            planning_horizon,
        ) + resolve_period_value(anc_rev.get(carrier, 0), planning_horizon)
        revenue_per_mw_year = 1000.0 * revenue_kwyr
        new_capital_cost_per_mw_year = steer_cost_per_mw_year - revenue_per_mw_year

        if new_capital_cost_per_mw_year < 0:
            logger.warning(
                f"{idx} ({carrier} {planning_horizon}): NET capital_cost after revenue = "
                f"${new_capital_cost_per_mw_year:,.0f}/MW-yr is NEGATIVE. LP will likely be "
                f"UNBOUNDED. STEER=${steer_cost_per_mw_year:,.0f} minus revenue="
                f"${revenue_per_mw_year:,.0f}. Reduce storage_capacity_revenue or "
                f"storage_ancillary_revenue in config so credit stays strictly below STEER cost.",
            )

        # Read the original cost for comparison logging
        original_capital_cost = row.capital_cost

        # Overwrite in-memory in the network
        n.storage_units.at[idx, "capital_cost"] = new_capital_cost_per_mw_year

        comparison_rows.append(
            {
                "Unit": idx,
                "Carrier": carrier,
                "Duration (h)": duration,
                "ATB Cost ($/MW-yr)": f"{original_capital_cost:.1f}",
                "STEER CAPEX ($/kW)": f"{total_capex_per_kw:.1f}",
                "STEER Cost ($/MW-yr)": f"{steer_cost_per_mw_year:.1f}",
                "Revenue Offset ($/MW-yr)": f"{revenue_per_mw_year:.1f}",
                "Net LP Cost ($/MW-yr)": f"{new_capital_cost_per_mw_year:.1f}",
            },
        )

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        logger.info(
            f"\n=== BESS Cost Comparison (Year {planning_horizon}, Experience {experience}) ===\n"
            + comp_df.to_string(index=False)
            + "\n",
        )


def prepare_network(n, solve_opts=None):
    if "clip_p_max_pu" in solve_opts:
        df = n.generators_t.p_max_pu
        n.generators_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.generators_t.p_min_pu
        n.generators_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.links_t.p_max_pu
        n.links_t.p_max_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
        df = n.links_t.p_min_pu
        n.links_t.p_min_pu = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)

        df = n.storage_units_t.inflow
        n.storage_units_t.inflow = df.where(df > solve_opts["clip_p_max_pu"], other=0.0)
    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        #
        # `load_shedding: true` is REJECTED rather than silently defaulted. Upstream
        # intends `if not np.isscalar(load_shedding): load_shedding = 1e2`, but
        # np.isscalar(True) is True, so with a bare `true` the override never runs and
        # marginal_cost becomes the bool itself -> 1.0 $/kWh = $1,000/MWh, 100x below
        # the intended $100,000/MWh. At $1,000/MWh, blacking out customers is cheaper
        # than building a battery (a 4hr battery at ~$89,600/MW-yr needs to displace
        # only ~90 MWh of shed per MW-yr to pay for itself), so the price silently
        # decides whether the model invests for reliability at all. A full-year
        # Sherlock run (job 33948825, 2026-07-14) was spent before this was noticed.
        # There is no planning reserve margin in this scenario, so this number is the
        # ONLY thing making the model care about reliability. It must be stated.
        # NB `np.isscalar` is the wrong test here twice over: it is True for bools AND
        # True for strings. Require an actual number.
        if isinstance(load_shedding, bool) or not isinstance(load_shedding, (int, float)):
            raise ValueError(
                "solving.options.load_shedding must be an explicit value of lost load "
                f"in $/kWh (e.g. 100 -> $100,000/MWh), got {load_shedding!r}. "
                "A bare `true` is not accepted: np.isscalar(True) is True, so it would "
                "silently become 1.0 $/kWh = $1,000/MWh -- cheap enough that the "
                "optimiser sheds load instead of building capacity. Set a number, or "
                "set `false` to disable load shedding entirely.",
            )
        # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
        logger.warning(
            "Adding load shedding generators at %s $/kWh (= %s $/MWh). ALWAYS check the "
            "shed VOLUME afterwards: a large shed means the model 'solved' by blacking "
            "out load and the result is not meaningful.",
            load_shedding,
            load_shedding * 1e3,
        )
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index

        # Size the backstop. Upstream hardcodes p_nom=1e9 kW = 1 TW PER BUS, roughly
        # 17x California's entire peak on every one of 58 buses. That single number is
        # the source of the coefficient range Gurobi complains about on every run:
        #     Bounds range  [3e+09, 3e+09]
        #     Warning: Model contains large rhs / large bounds
        # and it is the leading suspect for the barrier stalls that killed R21, R23,
        # R24, R25, R26, R27, R31 and R32 (all "Numerical trouble encountered" at
        # 50-90 iterations with primal infeasibility frozen).
        #
        # load_shedding_p_nom_factor sizes each shed generator to a multiple of ITS OWN
        # bus's peak load instead. It stays a backstop that cannot bind -- R11N shed
        # 2.3 MWh across a whole year against a fleet of tens of GW -- while removing
        # the 1e9 bound. Omit the option and behaviour is byte-identical to upstream,
        # so this changes nothing for any run that does not opt in.
        shed_factor = solve_opts.get("load_shedding_p_nom_factor")
        if shed_factor:
            bus_peak_mw = n.loads_t.p_set.T.groupby(n.loads.bus).sum().T.max().reindex(buses_i)
            # a bus with no load still needs a finite backstop; give it the fleet median
            bus_peak_mw = bus_peak_mw.fillna(bus_peak_mw.median()).clip(lower=1.0)
            shed_p_nom = bus_peak_mw * 1e3 * float(shed_factor)  # MW -> kW, then scale
            logger.warning(
                "Load-shed p_nom sized to %.1fx each bus's own peak instead of the "
                "upstream 1e9 kW: range %.3g to %.3g kW (was 1e+09 everywhere). This "
                "compresses the LP bound range by about %.0fx.",
                float(shed_factor),
                shed_p_nom.min(),
                shed_p_nom.max(),
                1e9 / max(shed_p_nom.max(), 1.0),
            )
        else:
            shed_p_nom = 1e9  # kW, upstream default

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=shed_p_nom,  # kW
        )

    if solve_opts.get("noisy_costs"):  ##random noise to costs of generators
        for t in n.iterate_components():
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        # Get first nhours for each level of the multi-index
        first_nhours = pd.MultiIndex.from_tuples(
            [
                snap
                for year in n.snapshots.get_level_values(0).unique()
                for snap in n.snapshots[n.snapshots.get_level_values(0) == year][:nhours]
            ],
            names=n.snapshots.names,
        )
        n.set_snapshots(first_nhours)
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    sector_enabled = "sector" in opts

    # Make snakemake available in function scope if it exists in global scope
    global_snakemake = globals().get("snakemake")

    # Define constraint application functions in a registry
    # Each function should take network and necessary parameters
    constraint_registry = {
        "RPS": lambda: (
            add_RPS_constraints(n, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        ),
        "REM": lambda: add_regional_co2limit(n, config) if n.generators.p_nom_extendable.any() else None,
        "ERM": lambda: (
            add_ERM_constraints(n, snapshots, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        ),
        "TCT": lambda: (
            add_technology_capacity_target_constraints(n, config) if n.generators.p_nom_extendable.any() else None
        ),
    }

    # Some constraints have different logic for sector networks
    if sector_enabled:
        constraint_registry["RPS"] = lambda: (
            add_RPS_constraints_sector(n, config, global_snakemake) if n.generators.p_nom_extendable.any() else None
        )
        constraint_registry["REM"] = lambda: (
            add_sector_co2_constraints(n, config) if n.generators.p_nom_extendable.any() else None
        )

    # Apply constraints based on options
    for opt in opts:
        if opt in constraint_registry:
            constraint_registry[opt]()

    # Always apply land use constraints
    add_land_use_constraints(n)

    # Always apply bidirectional link constraints
    add_bidirectional_link_constraints(n)

    # Apply operational reserve if configured
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    # Apply demand response if configured
    dr_config = config["electricity"].get("demand_response", {})
    if dr_config:
        add_demand_response_constraint(n, config, sector_enabled)

    # Apply interchange constraints if configured
    if config["electricity"].get("imports", {}).get("enable", False):
        if config["electricity"].get("imports", {}).get("volume_limit", False):
            add_interchange_constraints(n, config, "imports", sector_enabled)

    # Apply interchange constraints if configured
    if config["electricity"].get("exports", {}).get("enable", False):
        if config["electricity"].get("exports", {}).get("volume_limit", False):
            add_interchange_constraints(n, config, "exports", sector_enabled)

    # Apply sector-specific constraints if sector is enabled
    if sector_enabled:
        # Heat pump constraints
        add_cooling_heat_pump_constraints(n, config)

        # Apply GSHP capacity constraint if urban/rural not split
        if not config["sector"]["service_sector"].get("split_urban_rural", False):
            add_gshp_capacity_constraint(n, config, global_snakemake)

        # Natural gas import/export constraints
        if config["sector"]["natural_gas"].get("imports", False):
            add_ng_import_export_limits(n, config)

        # Water heater constraints
        water_config = config["sector"]["service_sector"].get("water_heating", {})
        if not water_config.get("simple_storage", True):
            add_water_heater_constraints(n, config)

        # EV generation constraints
        if config["sector"]["transport_sector"].get("ev_policy", {}):
            add_ev_generation_constraint(n, config, global_snakemake)

        # Sector demand response constraints
        add_sector_demand_response_constraints(n, config)

        # Fossil generation constraints
        add_fossil_generation_constraint(n, config)


def jitter_extendable_capital_costs(n, cf_solving):
    """Break capital-cost ties among extendable generators and storage units.

    noisy_costs (prepare_network) perturbs marginal costs everywhere but capital
    costs only for Line/Link, so extendable Generator/StorageUnit capital costs
    stay exactly equal across nodes. When no policy floor pins the build, that
    equality is a flat optimal face and barrier stalls sub-optimal: R40-R43
    (2026-08-26) all died this way at 2030 the moment the TCT mandate was
    loosened by crediting the existing fleet, while the identically-configured
    tight-mandate parent R36 certified. Multiplicative +/-5e-5 jitter sits above
    BarConvTol (1e-5) and orders of magnitude below any decision-relevant cost
    difference. Called from run_optimize so it lands AFTER update_bess_costs,
    which overwrites battery capital costs from STEER before every horizon.
    """
    if not cf_solving.get("noisy_costs"):
        return
    for t in n.iterate_components(["Generator", "StorageUnit"]):
        ext = t.df["p_nom_extendable"].to_numpy(dtype=bool)
        if ext.any():
            t.df.loc[ext, "capital_cost"] *= 1 + 1e-4 * (np.random.random(int(ext.sum())) - 0.5)


def run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs):
    """Initiate the correct type of pypsa.optimize function."""
    jitter_extendable_capital_costs(n, cf_solving)
    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs,
        )

    if "infeasible" in condition:
        # n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    # RAISE, do not warn. A barrier that stalls returns status 'warning' with condition
    # 'suboptimal', and the old code logged one line and carried on: the network was
    # written, snakemake reported success, slurm reported COMPLETED, exit 0. R21
    # (job 40651525, 2026-08-24) did exactly that -- "Numerical trouble encountered" in
    # 2030 and "Sub-optimal termination" in 2040, with a 21% primal-dual gap, and still
    # produced a 383 MB .nc that looks like a result. Under myopic foresight a bad early
    # horizon also propagates, because prepare_brownfield carries its build forward.
    #
    # CLAUDE.md 5: make the silent path impossible.
    allow_suboptimal = bool(cf_solving.get("allow_suboptimal", False))
    if not rolling_horizon and not allow_suboptimal and (status != "ok" or condition != "optimal"):
        raise RuntimeError(
            f"Solve did not reach optimality: status '{status}', condition "
            f"'{condition}'. The network was NOT written. This is usually numerical: "
            "check the Gurobi log for 'Numerical trouble', 'Sub-optimal termination', "
            "or the 'large bounds'/'large rhs' warnings, and consider NumericFocus, "
            "enabling crossover, or reducing the coefficient range. Set "
            "solving.options.allow_suboptimal: true to override, but then say so "
            "explicitly in the run's README -- the numbers are not trustworthy.",
        )
    if allow_suboptimal and (status != "ok" or condition != "optimal"):
        logger.error(
            "ALLOW_SUBOPTIMAL IS SET and the solve did not reach optimality "
            "(status %r, condition %r). Continuing anyway. Any number produced by "
            "this run is provisional and must be labelled as such.",
            status,
            condition,
        )


def prepare_brownfield(n, planning_horizon):
    """Prepare the network for the next planning horizon by setting up brownfield constraints.
    Used for myopic foresight.

    This function:
    1. Sets minimum capacities for transmission lines and DC links
    2. Updates generator, link, and storage unit capacities
    3. Handles time-dependent data transfer between planning periods
    """
    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n.lines.s_nom_opt  # for lines
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n.links.loc[dc_i, "p_nom_opt"]  # for links

    for c in n.iterate_components(["Generator", "Link", "StorageUnit"]):
        nm = c.name
        # limit our components that we remove/modify to those prior to this time horizon
        c_lim = c.df.loc[n.get_active_assets(nm, planning_horizon)]

        logger.info(f"Preparing brownfield for the component {nm}")
        # attribute selection for naming convention
        attr = "p"
        # copy over asset sizing from previous period
        c_lim[f"{attr}_nom"] = c_lim[f"{attr}_nom_opt"]
        c_lim[f"{attr}_nom_extendable"] = False
        df = copy.deepcopy(c_lim)
        time_df = copy.deepcopy(c.pnl)

        for c_idx in c_lim.index:
            n.remove(nm, c_idx)

        # Rebuild each asset from its own saved row.
        #
        # This used to enumerate Generator attributes by hand and pass them one by
        # one, which silently dropped every attribute not on that list. `sign` was
        # missing, and pypsa's default for it is 1 (pypsa/component_attrs/
        # generators.csv: "sign,float,n/a,1,power sign,Input (optional)"). So from
        # the SECOND horizon onwards the load-shedding generators reverted from
        # sign=1e-3 to sign=1.0, and their marginal_cost -- deliberately written in
        # $/kWh, which only means "$/MWh x 1000" while sign=1e-3 -- was read as
        # plain $/MWh. A blackout cost $100/MWh instead of $100,000/MWh and became
        # a system-wide price cap.
        #
        # Measured in job 34070598 (2026-07-15), which is what exposed this:
        #   2030 (before this function runs) shed 0 MWh          <- price correct
        #   2040 (after)  shed 323,170 MWh, peaking at 11,086 MW <- price 1000x low
        #   every 2040 bus price pinned at exactly $100.01/MWh   <- the cap
        #   2030 prices reached $20,932/MWh                      <- legal at $100,010
        # Links and StorageUnits always used the whole-row form below and were
        # never affected.
        #
        # KNOWN, NOT FIXED HERE (keep one variable moving at a time): heat_rate,
        # fuel_cost, vom_cost, carrier_base and land_region are not declared pypsa
        # Generator attributes, so pypsa ignores them on add and they are lost here
        # regardless of which form is used. Same class of bug, separate change.
        for df_idx in df.index:
            n.add(nm, df_idx, **df.loc[df_idx])
        logger.info(n.consistency_check())

        # Do not trust the rebuild -- verify it. An attribute silently reverting to
        # its default is precisely what this function has been doing, and nothing
        # objected for as long as it went unmeasured.
        if nm == "Generator":
            shed = n.generators[n.generators.carrier == "load"]
            if not shed.empty and not np.isclose(shed.sign, 1e-3).all():
                raise ValueError(
                    "prepare_brownfield lost `sign` on the load-shedding generators: got "
                    f"{sorted(set(shed.sign))}, expected 1e-3. Their marginal_cost is "
                    "denominated in $/kWh and only means $/MWh x 1000 while sign=1e-3. At "
                    "sign=1.0 a blackout costs 1000x less than intended and silently caps "
                    "every price in the system, so the optimiser blacks out load instead of "
                    "building capacity.",
                )

        # copy time-dependent
        selection = n.component_attrs[nm].type.str.contains("series")
        for tattr in n.component_attrs[nm].index[selection]:
            n.import_series_from_dataframe(time_df[tattr], nm, tattr)

    # roll over the last snapshot of time varying storage state of charge to be the state_of_charge_initial for the next time period
    n.storage_units.loc[:, "state_of_charge_initial"] = n.storage_units_t.state_of_charge.loc[planning_horizon].iloc[-1]


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    foresight = snakemake.params.foresight
    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"

    kwargs["solver_options"] = solving["solver_options"][set_of_options] if set_of_options else {}
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment",
        False,
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)

    sns_portion = cf_solving.get("snapshot_portion", None)
    if sns_portion:
        logger.info(f"Optimizing over snapshots from {sns_portion['start']} to {sns_portion['end']}")
        sns_portion = pd.date_range(start=sns_portion["start"], end=sns_portion["end"], freq="h")
        sns = n.snapshots
        sns_portion = sns[sns.get_level_values(1).isin(sns_portion)]
        sns_portion.name = "snapshot"
        kwargs["snapshots"] = sns_portion

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for additional_constraints
    n.config = config
    n.opts = opts

    steer_dynamic = config.get("costs", {}).get("steer_dynamic", False)
    # Frozen-experience twin (Act 2 counterfactual, 2026-08-27): STEER still prices
    # every horizon (commodity floors, global exogenous forecast and macro factor all
    # keep their year dependence), but the state transition X(t+1) = X(t) + dX(t) is
    # skipped, so California's own builds never feed back into the experience pools.
    # Endo run minus frozen run therefore isolates the endogenous learning feedback
    # alone: identical cost model, identical 2030 starting point by construction.
    steer_freeze_experience = config.get("costs", {}).get("steer_freeze_experience", False)
    if steer_freeze_experience and not steer_dynamic:
        raise ValueError(
            "steer_freeze_experience: true requires steer_dynamic: true — the flag "
            "freezes STEER's experience pools, which only exist on the STEER path.",
        )
    if steer_dynamic:
        if foresight != "myopic":
            raise ValueError(
                "STEER dynamic cost integration is only compatible with myopic foresight. "
                f"Foresight option is set to '{foresight}'.",
            )
        # Load one STEER SystemEngine per enabled chemistry. Chemistries are picked up
        # from the extendable_carriers.StorageUnit list via opts/steer_carriers.py, so
        # Na-ion is opt-in: adding 4hr_battery_storage_naion to that list triggers the
        # Na engine load; leaving it out is a pure LFP run, unchanged from before.
        extendable_storage = config["electricity"]["extendable_carriers"]["StorageUnit"]
        # Fail loudly at startup if any BESS carrier lacks an explicit ITC entry —
        # audit §3.9 item 3, otherwise a new Na carrier silently gets 0% ITC while LFP
        # gets 30% and the "symmetric" competition isn't.
        assert_itc_covers_bess(config.get("costs", {}).get("itc_modifier", {}), extendable_storage)
        steer_techs = enabled_chemistries(extendable_storage)
        if not steer_techs:
            raise ValueError(
                "steer_dynamic: true but no BESS carriers are extendable — nothing to "
                "compute costs for. Add e.g. 4hr_battery_storage to extendable_carriers.",
            )
        steer_dir = Path(__file__).resolve().parents[3] / "02_STEERMODEL"
        if str(steer_dir) not in sys.path:
            sys.path.insert(0, str(steer_dir))
        # Per-run override of which STEER config file a chemistry loads
        # (costs.steer_config_by_tech, e.g. {na: config_na_ion_s005.yaml}). With the
        # key absent every run resolves exactly as before via STEER_CONFIG_BY_TECH.
        # This exists so runs with different spillover_sigma variants can execute in
        # PARALLEL on the cluster: without it they would all read the one shared
        # config_na_ion.yaml, whose content at solve-start time (hours after sbatch
        # submission) is a race. The run config also becomes the provenance record
        # of exactly which STEER file produced its numbers.
        steer_cfg_override = config.get("costs", {}).get("steer_config_by_tech", {}) or {}
        try:
            from steer.experience import ExperienceState
            from steer.loader import load_system

            steer_engines = {}
            for tech in steer_techs:
                cfg_path = steer_dir / steer_cfg_override.get(tech, STEER_CONFIG_BY_TECH[tech])
                if not cfg_path.exists():
                    raise FileNotFoundError(
                        f"STEER config for '{tech}' not found: {cfg_path} "
                        f"(override in costs.steer_config_by_tech: {steer_cfg_override})",
                    )
                engine = load_system(cfg_path)
                for comp in engine.components:
                    comp.validate()
                steer_engines[tech] = engine
                logger.info(f"Loaded and validated STEER engine for {tech} from {cfg_path.name}.")
            # Seed one pool per (chemistry x declared unit). Na cells learn from Na
            # only; pack/PCS/BoS/EPC learn from all deployment summed. See
            # steer/experience.py for why the pooling can't be a shared scalar.
            experience = ExperienceState.seed(steer_engines)
            logger.info(f"Initialized STEER experience pools: {experience}")
        except Exception as e:
            logger.error(f"Failed to load or validate STEER system engines from {steer_dir}: {e}")
            raise e

    match foresight:
        case "perfect":
            run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)
        case "myopic":
            for i, planning_horizon in enumerate(n.investment_periods):
                sns_horizon = n.snapshots[n.snapshots.get_level_values(0) == planning_horizon]
                kwargs["snapshots"] = sns_horizon

                if steer_dynamic:
                    # STEER sets CAPEX for every enabled chemistry BEFORE solve. Each
                    # call is scoped to its own carriers by opts/steer_carriers.carriers_for_tech.
                    for tech, engine in steer_engines.items():
                        update_bess_costs(
                            n,
                            planning_horizon,
                            experience,
                            engine,
                            tech,
                            config,
                        )

                run_optimize(n, rolling_horizon, skip_iterations, cf_solving, **kwargs)  # ← solve this horizon

                if steer_dynamic and steer_freeze_experience:
                    logger.info(
                        f"Horizon {planning_horizon} solved. steer_freeze_experience: true — "
                        f"experience pools NOT updated (frozen exogenous twin). "
                        f"Pools remain: {experience}.",
                    )
                if steer_dynamic and not steer_freeze_experience:
                    # State-transition X(t+1) = X(t) + ΔX(t), split by chemistry.
                    # Every chemistry updates its OWN pool AND the shared "all" pools
                    # inside ExperienceState.add. Using a substring match on
                    # "battery_storage" here (as before Na was wired in) would double-
                    # count Na deployment into LFP's own pool. Audit §3.9 item 6.
                    horizon_summary = []
                    for tech in steer_engines:
                        tech_carriers = carriers_for_tech(tech)
                        tech_mask = n.storage_units.carrier.isin(tech_carriers) & (
                            n.storage_units.build_year == planning_horizon
                        )
                        p_nom_opt = n.storage_units.loc[tech_mask, "p_nom_opt"].fillna(0)
                        max_hours = n.storage_units.loc[tech_mask, "max_hours"].fillna(0)
                        added_gwh = (p_nom_opt * max_hours).sum() / 1e3
                        added_gw = p_nom_opt.sum() / 1e3
                        experience.add(tech, added_gwh=added_gwh, added_gw=added_gw)
                        horizon_summary.append(f"{tech}: {added_gwh:.2f} GWh / {added_gw:.2f} GW")
                    logger.info(
                        f"Horizon {planning_horizon} solved. "
                        f"BESS added — {' | '.join(horizon_summary)}. "
                        f"New cumulative experience: {experience}.",
                    )

                if i == len(n.investment_periods) - 1:
                    logger.info(f"Final time horizon {planning_horizon}")
                    continue
                logger.info(f"Preparing brownfield from {planning_horizon}")
                prepare_brownfield(n, planning_horizon)

        case _:
            raise ValueError(f"Invalid foresight option: '{foresight}'. Must be 'perfect' or 'myopic'.")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            interconnect="eastern",
            simpl="120",
            clusters="6m",
            ll="v1.0",
            opts="1h-TCT",
            sector="E-G",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    opts = snakemake.wildcards.opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    # sector specific co2 options
    if snakemake.wildcards.sector != "E":
        opts.append("sector")

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
    )

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        opts=opts,
        log_fn=snakemake.log.solver,
    )

    if "ERM" in opts:
        store_ERM_duals(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
