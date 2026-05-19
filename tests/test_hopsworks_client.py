import pytest

import hopsworks_client


def test_validate_hopsworks_config_raises_when_project_missing(monkeypatch):
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_PROJECT", None)
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_API_KEY", "fake-key")

    with pytest.raises(ValueError, match="HOPSWORKS_PROJECT"):
        hopsworks_client.validate_hopsworks_config()


def test_validate_hopsworks_config_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_PROJECT", "fake-project")
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_API_KEY", None)

    with pytest.raises(ValueError, match="HOPSWORKS_API_KEY"):
        hopsworks_client.validate_hopsworks_config()


def test_validate_hopsworks_config_passes_when_required_values_exist(monkeypatch):
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_PROJECT", "fake-project")
    monkeypatch.setattr(hopsworks_client, "HOPSWORKS_API_KEY", "fake-key")

    hopsworks_client.validate_hopsworks_config()
