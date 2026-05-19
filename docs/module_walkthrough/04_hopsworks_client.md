# src/hopsworks_client.py

## Role in the Project

`hopsworks_client.py` centralizes the connection logic for Hopsworks.

Pipeline scripts should not call `hopsworks.login()` directly. Instead, they use helper functions from this module to access:

- the Hopsworks project
- the Feature Store
- the Model Registry

This keeps authentication and connection setup in one place.

## Inputs

The module reads Hopsworks configuration values from `config.py`:

- `HOPSWORKS_HOST`
- `HOPSWORKS_PORT`
- `HOPSWORKS_PROJECT`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_ENGINE`

These values are loaded from `.env` through `config.py`.

## Outputs

The module returns authenticated Hopsworks objects:

- project object
- Feature Store object
- Model Registry object

These objects are then used by the pipeline scripts to create, read or write Feature Store and Model Registry artifacts.

## Key Functions

### `validate_hopsworks_config()`

Checks whether the required Hopsworks connection settings are available.

It validates:

- `HOPSWORKS_PROJECT`
- `HOPSWORKS_API_KEY`

This validation is intentionally done in `hopsworks_client.py` instead of `config.py`.

The reason is that many modules and tests can run offline without Hopsworks credentials. Credentials should only be required when a pipeline actually tries to connect to Hopsworks.

### `get_project()`

Logs into Hopsworks and returns the authenticated project object.

The API key is passed to `hopsworks.login()` but is never printed.

### `get_feature_store()`

Returns the Feature Store object for the configured Hopsworks project.

This is used by pipelines that create, read or write Feature Groups and Feature Views.

### `get_model_registry()`

Returns the Model Registry object for the configured Hopsworks project.

This is used by training and inference pipelines to upload or load model artifacts.

### `main()`

Runs a manual smoke test for the Hopsworks connection.

It prints non-secret metadata such as the project name and confirms that the Feature Store and Model Registry can be accessed.

## Key Design Decisions

### Centralized Hopsworks Login

All Hopsworks authentication is centralized in one module.

This avoids duplicated login logic across pipeline scripts and makes future connection changes easier.

### Lazy Credential Validation

Hopsworks credentials are validated only when Hopsworks access is needed.

This allows local unit tests, feature engineering and offline experiments to run without a `.env` file containing Hopsworks credentials.

### Secret Handling

The API key is loaded from environment variables and passed to Hopsworks, but it is never printed or committed.

(The `.env` file is excluded with `.gitignore` from Git.)

## Validation and Error Handling

If required Hopsworks values are missing, the module raises a clear `ValueError` listing the missing environment variables.

## Limitations / Future Work

- The module currently creates a new Hopsworks connection whenever `get_project()` is called.
- A future version could add connection reuse or context management if connection overhead becomes an issue.
- More advanced deployments could support separate Hopsworks configurations for development, staging and production.