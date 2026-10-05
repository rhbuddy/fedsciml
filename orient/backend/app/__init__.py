"""Orient Server (backend) application package.

Federated SciML server: orchestrates federated rounds, collects client weight
updates over HTTP/REST, aggregates them, and tracks metrics.

Reference design: Flower (strategy abstraction) + safetensors wire format.
See ../README.md and ../../SRS_FederatedSciML_Project.md (v1.4, §4.11-4.13).
"""

__version__ = "0.1.0"
