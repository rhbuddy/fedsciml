---
name: system-designer
description: >-
  Use this skill when the user asks to design, review, or discuss software
  system architectures. Covers high-level system design, API design, data
  modeling, scalability, reliability, and trade-off analysis for backend and
  full-stack systems.
---

# System Designer Skill

## When to Use

Use this skill for designing or reviewing software systems: high-level
architecture, API design, data modeling, scalability, reliability, and
trade-off analysis.

## Approach

1. **Gather requirements**: functional, non-functional (scale, latency,
   consistency, availability), constraints.
2. **Define scope**: what is in/out of v1.
3. **Choose high-level architecture**: monolith vs microservices, client-server,
   serverless, hybrid.
4. **Design data model**: entities, relationships, storage choices.
5. **Design APIs**: REST / GraphQL / gRPC, contracts, versioning.
6. **Identify bottlenecks & scaling**: caching, sharding, load balancing,
   async processing.
7. **Address reliability**: fault tolerance, retries, idempotency, monitoring.
8. **Discuss trade-offs**: CAP, consistency models, cost, complexity.

## High-Level Architecture Patterns

| Pattern | When to Use |
| --- | --- |
| **Monolith** | Small team, early stage, low traffic, fast iteration. |
| **Microservices** | Large team, high scale, independent deployability, polyglot. |
| **Serverless** | Event-driven, bursty traffic, minimal ops overhead. |
| **Modular Monolith** | Transition path from monolith to microservices. |

## API Design

- Use **REST** with resource-oriented URLs, HTTP methods, and JSON.
- Version APIs: `/v1/`, `/v2/` or header-based versioning.
- Use **GraphQL** for flexible client queries when over/under-fetching is an issue.
- Use **gRPC** for internal service-to-service communication.
- Define contracts with **OpenAPI (Swagger)** or **Protocol Buffers**.
- Design for idempotency (Idempotency-Key header) on write operations.

## Data Modeling

- Choose storage based on access patterns:
  - **Relational (PostgreSQL)**: strong consistency, joins, transactions.
  - **NoSQL (MongoDB, DynamoDB)**: flexible schema, horizontal scale.
  - **Key-Value (Redis, etcd)**: caching, sessions, fast lookups.
  - **Time-series (InfluxDB, Timescale)**: metrics, logs.
  - **Search (Elasticsearch, Meilisearch)**: full-text search.
- Normalize for OLTP; denormalize for OLAP/reporting.
- Use **eventual consistency** where strong consistency is not required.

## Scalability & Performance

- **Horizontal scaling**: stateless services + load balancer.
- **Caching layers**: CDN (static), Redis (hot data), application cache.
- **Database scaling**: read replicas, connection pooling, sharding/partitioning.
- **Async processing**: message queues (RabbitMQ, Kafka, SQS) for decoupled work.
- **Rate limiting & throttling**: token bucket, leaky bucket.

## Reliability & Observability

- **Fault tolerance**: circuit breakers, retries with backoff, fallbacks.
- **Idempotency**: design writes to be safely retryable.
- **Monitoring**: metrics (Prometheus), logs (ELK/Loki), traces (Jaeger).
- **Health checks**: liveness + readiness probes.
- **Disaster recovery**: backups, multi-region replication, RTO/RPO.

## Trade-off Analysis (Always Consider)

- **Consistency vs Availability** (CAP).
- **Latency vs Freshness** (cache TTL).
- **Complexity vs Flexibility** (microservices vs monolith).
- **Cost vs Performance** (managed vs self-hosted).
- **Build vs Buy** (custom vs off-the-shelf components).

## Deliverables

When asked to design a system, produce:
1. **Context & requirements** summary.
2. **High-level architecture diagram** (components, data flow).
3. **API contract** (endpoints, request/response schemas).
4. **Data model** (entities, relationships).
5. **Scalability & reliability notes**.
6. **Trade-offs & open questions**.
