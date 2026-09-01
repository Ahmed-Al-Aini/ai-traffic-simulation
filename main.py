"""Entry point for the traffic simulation."""

from traffic_sim.simulator import TrafficSimulator


def main() -> None:
    try:
        print("Starting Traffic System v6.2...")
        simulator = TrafficSimulator()
        simulator.run()
    except ImportError as exc:
        print(f"Import error: {exc}")
        print("Please install: pip install torch numpy pygame")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
