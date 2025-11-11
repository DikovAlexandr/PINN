import argparse


def set_default_backend(backend_name):
    """Set default backend (only PyTorch is supported)."""
    if backend_name.lower() != "pytorch":
        raise ValueError(
            f"Only PyTorch backend is supported. Got: {backend_name}"
        )
    print("PyTorch backend is the only supported backend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend",
        nargs=1,
        type=str,
        choices=["pytorch"],
        help="Set default backend (only pytorch is supported)",
    )
    args = parser.parse_args()
    set_default_backend(args.backend[0])
