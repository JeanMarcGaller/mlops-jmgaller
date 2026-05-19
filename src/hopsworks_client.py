"""Hopsworks client helpers for Feature Store and Model Registry access.

This module centralizes all Hopsworks connection logic. Pipeline scripts should
not call `hopsworks.login()` directly. Instead, they use:

    get_feature_store()
    get_model_registry()

This keeps authentication, configuration validation and future changes to the
Hopsworks connection setup in one place.
"""

import hopsworks

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_ENGINE,
    HOPSWORKS_HOST,
    HOPSWORKS_PORT,
    HOPSWORKS_PROJECT,
)


def validate_hopsworks_config() -> None:
    """Validate that required Hopsworks settings are available.

    This validation is intentionally done here instead of in `config.py`.
    Reason: modules such as `features.py` or `weather_api.py` can be tested
    offline without Hopsworks credentials.

    Raises:
        ValueError: If required Hopsworks environment variables are missing.
    """
    missing_values = []

    if not HOPSWORKS_PROJECT:
        missing_values.append("HOPSWORKS_PROJECT")

    if not HOPSWORKS_API_KEY:
        missing_values.append("HOPSWORKS_API_KEY")

    if missing_values:
        missing_values_text = ", ".join(missing_values)
        raise ValueError(
            "Missing required Hopsworks environment variables: "
            f"{missing_values_text}. "
            "Please create a local .env file based on .env.example."
        )


def get_project():
    """Login to Hopsworks and return the project object.

    Returns:
        Authenticated Hopsworks project object.

    Notes:
        The API key is passed to `hopsworks.login()` but is never printed.
    """
    validate_hopsworks_config()

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine=HOPSWORKS_ENGINE,
    )

    return project


def get_feature_store():
    """Return the Hopsworks Feature Store for the configured project."""
    project = get_project()

    return project.get_feature_store()


def get_model_registry():
    """Return the Hopsworks Model Registry for the configured project."""
    project = get_project()

    return project.get_model_registry()


def main() -> None:
    """Run a manual smoke test for the Hopsworks connection."""
    project = get_project()

    print("Connected to Hopsworks project.")
    print(f"Project name: {project.name}")

    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()

    print("Feature Store:")
    print(feature_store)

    print("Model Registry:")
    print(model_registry)


if __name__ == "__main__":
    main()
