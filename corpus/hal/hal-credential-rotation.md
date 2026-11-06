---
id: hal-credential-rotation
tenant: hal
title: Database credential rotation
department: engineering
acl: engineering
sensitivity: confidential
---

# Database credential rotation

Database credentials are rotated every 30 days by the secrets manager.

Services read credentials at startup and reload them on a signal, so rotation needs no deploy.
