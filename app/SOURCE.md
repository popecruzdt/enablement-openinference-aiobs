# Source

Copied from `popecruzdt/workshop-destination-automation`, branch `agentic-workflow`, commit `3730529d1f33c963a2e68aca4deb46d8a4392dd5` (`app/`).

Changes from the source:
- Removed Podman files (`podman-compose.yml`, `PODMAN_COMPOSE_SETUP.md`), `.env.example`, `opentelemetry/` and `nginx/`. Their Kubernetes equivalents are in `k8s/`.
- `src/samples/` removed (unused).
- `Containerfile` bakes `destinations/` into the image and no longer copies `.env.example`.

Syncs from the source repo are manual; update the commit above when you sync.
