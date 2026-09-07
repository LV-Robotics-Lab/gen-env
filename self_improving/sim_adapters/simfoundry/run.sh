#!/usr/bin/env bash
# Thin CLI boundary: upstream owns reconstruction and simulator behavior.
set -eo pipefail
SF_ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SF_ADAPTER_DIR}/env.sh"
SF_REPO_DIR="$(cd "${SF_ADAPTER_DIR}/../../.." && pwd)/external/SimFoundry"

case "${1:-help}" in
  reconstruct) SF_PIPELINE=A_reconstruction ;;
  augment) SF_PIPELINE=B_augmentation ;;
  smoke) SF_PIPELINE=C_application ;;
  doctor)
    shift
    exec python "${SF_ADAPTER_DIR}/doctor.py" "$@"
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: bash self_improving/sim_adapters/simfoundry/run.sh COMMAND [upstream options]
  reconstruct  Run upstream video -> OmniGibson reconstruction (pipeline A).
  augment      Run upstream scene augmentation (pipeline B).
  smoke        Run upstream application smoke test (pipeline C).
  doctor       Print local prerequisite status as JSON; no GPU framework imports.

Use COMMAND --help for upstream flags. Pipeline output defaults to external/SimFoundry/Data.
Set --root-dir to an absolute output directory. Reconstruction does not run pipeline C.
EOF
    exit 0
    ;;
  *) echo "Unknown command: $1" >&2; exit 2 ;;
esac
shift
# Local service configuration is trusted shell code, ignored by Git, and loaded only for pipelines.
SF_SERVICE_ENV="${SIMFOUNDRY_SERVICE_ENV:-${MAMBA_ROOT_PREFIX%/miniforge}/service.env}"
if [[ -f "${SF_SERVICE_ENV}" ]]; then
  source "${SF_SERVICE_ENV}"
fi
if [[ "${SIMFOUNDRY_GEMINI_NONSTREAM_TEXT:-0}" == "1" ]]; then
  export PYTHONPATH="${SF_ADAPTER_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec bash "${SF_REPO_DIR}/scripts/pipeline/${SF_PIPELINE}/run.sh" "$@"
