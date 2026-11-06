---
id: hal-postmortem-db
tenant: hal
title: Postmortem: primary database outage
department: engineering
acl: engineering
sensitivity: confidential
---

# Postmortem: primary database outage

The primary database outage lasted 47 minutes after a failed storage migration filled the disk.

Action items: alert at 80 percent disk usage, rehearse failover quarterly, and stage migrations on a replica first.
