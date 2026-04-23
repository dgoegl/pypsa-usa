"""Test script for the STEER Decomposition Engine."""

from decomposition_engine import Technology


def main():
    """Main test execution."""
    # Load the tech architecture from YAML
    yaml_path = "steer/config/steer_dummy_lfp.yaml"
    tech = Technology.from_yaml(yaml_path)

    print(f"Loaded Technology: {tech.name}")
    print("=" * 40)

    # Analyze the C1 Component
    c1 = next(c for c in tech.components if c.name == "C1_Cell")

    print("--- BASELINE SCENARIO ---")
    print(f"C1 Total Cost: ${c1.total_cost:.2f}")
    print(f"C1 Commodities Cost (Floor): ${c1.commodities_cost:.2f}")
    print(f"C1 Learnable Cost: ${c1.learnable_cost:.2f}")
    print(f"C1 a_factor (Exogenous %): {c1.a_factor:.1%}")
    print(f"C1 b_factor (Endogenous %): {c1.b_factor:.1%}")

    print("\n--- INFLATION REDUCTION ACT SCENARIO ---")
    print("Shifting 'Cell Assembly & Overhead' ($25) from Exogenous to Endogenous bucket...")

    # Programmatically flip a subcomponent's bucket to test the dynamic roll-up
    target_sub = next(s for s in c1.sub_components if s.name == "Cell Assembly & Overhead")
    target_sub.bucket_tag = "Endogenous"

    print(f"\nC1 a_factor (Exogenous %): {c1.a_factor:.1%}")
    print(f"C1 b_factor (Endogenous %): {c1.b_factor:.1%}")
    print(f"C1 Commodities Cost (Floor): ${c1.commodities_cost:.2f} (Unchanged)")


if __name__ == "__main__":
    main()
