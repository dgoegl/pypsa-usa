"""
STEER Decomposition Engine.

This module provides the core Object-Oriented structure for energy storage
cost decomposition into Commodities, Exogenous, and Endogenous buckets.
"""

import yaml


class SubComponent:
    """Reprersents a single cost input (e.g., Lithium, Labor)."""

    def __init__(self, name: str, base_usd: float, bucket_tag: str):
        self.name = name
        self.base_usd = base_usd
        # Allowed buckets: "Commodities", "Exogenous", "Endogenous"
        self.bucket_tag = bucket_tag

    @property
    def current_price(self) -> float:
        """Returns the current price of this subcomponent."""
        return self.base_usd


class Component:
    """Represents a major hardware group (e.g., Cell, EPC)."""

    def __init__(self, name: str, sub_components: list[SubComponent]):
        self.name = name
        self.sub_components = sub_components

    @property
    def total_cost(self) -> float:
        """Total sum of all sub-components (Commodities + Exogenous + Endogenous)."""
        return sum(s.current_price for s in self.sub_components)

    @property
    def commodities_cost(self) -> float:
        """Total sum of sub-components tagged as 'Commodities'."""
        return sum(s.current_price for s in self.sub_components if s.bucket_tag == "Commodities")

    @property
    def learnable_cost(self) -> float:
        """Total cost of non-commodity elements (Exogenous + Endogenous)."""
        return sum(s.current_price for s in self.sub_components if s.bucket_tag in ["Exogenous", "Endogenous"])

    @property
    def exogenous_cost(self) -> float:
        """Total cost of elements tagged as 'Exogenous'."""
        return sum(s.current_price for s in self.sub_components if s.bucket_tag == "Exogenous")

    @property
    def endogenous_cost(self) -> float:
        """Total cost of elements tagged as 'Endogenous'."""
        return sum(s.current_price for s in self.sub_components if s.bucket_tag == "Endogenous")

    @property
    def a_factor(self) -> float:
        """The dynamically cost-weighted Exogenous share of the learnable cost."""
        learnable = self.learnable_cost
        if learnable == 0:
            return 1.0  # Safe default
        return self.exogenous_cost / learnable

    @property
    def b_factor(self) -> float:
        """The dynamically cost-weighted Endogenous share of the learnable cost."""
        learnable = self.learnable_cost
        if learnable == 0:
            return 0.0  # Safe default
        return self.endogenous_cost / learnable


class Technology:
    """Represents a complete energy storage technology."""

    def __init__(self, name: str, components: list[Component]):
        self.name = name
        self.components = components

    @property
    def total_system_cost(self) -> float:
        """Sum of all component costs."""
        return sum(c.total_cost for c in self.components)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Builds a Technology object tree from a configuration YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        tech_name = data.get("technology", "Unknown")
        components_list = []

        # Iterate over component IDs like C1_Cell, C2_Pack
        for comp_id, comp_data in data.get("components", {}).items():
            subs_data = comp_data.get("sub_components", [])
            sub_components = []

            for s in subs_data:
                sub_components.append(
                    SubComponent(
                        name=s["name"],
                        base_usd=s["base_usd"],
                        bucket_tag=s["bucket"],
                    ),
                )

            components_list.append(Component(name=comp_id, sub_components=sub_components))

        return cls(name=tech_name, components=components_list)
