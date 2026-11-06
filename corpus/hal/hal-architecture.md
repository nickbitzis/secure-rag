---
id: hal-architecture
tenant: hal
title: Architecture overview
department: engineering
acl: engineering
sensitivity: internal
---

# Architecture overview

The platform is a set of services behind an API gateway, with Postgres for transactions and a message queue for events.

Each service owns its own schema. Cross-service reads go through APIs, not shared tables.
