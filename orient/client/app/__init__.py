"""Orient Client application package.

Federated SciML client: loads a private local dataset, trains locally on the
global model it downloads, and uploads **weights only** (safetensors).

The ``protocol``, ``weights``, ``models`` and ``problems`` modules are exact
copies of the server's, guaranteeing identical architectures and losses.

See ../README.md and ../../SRS_FederatedSciML_Project.md (v1.4, §4.10, §4.13).
"""

__version__ = "0.1.0"
