"""Test script for the STEER Decomposition Engine."""

from decomposition_engine import Technology


def main():
    """Main test execution against full LFP data."""
    # Load the full tech architecture from YAML
    yaml_path = "steer/config/steer_lfp_2024.yaml"
    tech = Technology.from_yaml(yaml_path)

    print(f"Loaded Technology: {tech.name}")
    print(f"Total Reference Cost: ${tech.total_system_cost:.2f} / kWh-system")
    print("=" * 60)
    print(f"{'Comp':<8} | {'Total':<8} | {'Commod':<8} | {'Exo (a)':<8} | {'Endo (b)':<8}")
    print("-" * 60)

    for comp in tech.components:
        print(f"{comp.name:<8} | ${comp.total_cost:>6.2f} | ${comp.commodities_cost:>6.2f} | {comp.a_factor:>7.1%} | {comp.b_factor:>7.1%}")
        if comp.name == "C3_PCS":
            print(f"\n   [Debug {comp.name}] Breakdown of learnable sub-components:")
            for s in comp.sub_components:
                if s.bucket_tag != "Commodities":
                    print(f"     - {s.name:<40} | ${s.current_price:>5.2f} | {s.bucket_tag}")
            print("")

    print("-" * 60)


if __name__ == "__main__":
    main()
