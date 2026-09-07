#!/usr/bin/env bash
# Install the pinned Mamba distribution into this project, without shell init changes.
set -euo pipefail
SF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SF_DIR}/env.sh"
SF_CACHE="$(dirname "${MAMBA_ROOT_PREFIX}")"
SF_INSTALLER="${SF_CACHE}/bootstrap/Miniforge3-Linux-x86_64.sh"
mkdir -p "${SF_CACHE}/bootstrap"
if [[ ! -e "${CONDARC}" ]]; then
  cat > "${CONDARC}" <<'EOF'
channels:
  - conda-forge
channel_priority: flexible
EOF
fi
if [[ -x "${MAMBA_ROOT_PREFIX}/bin/mamba" ]]; then
  "${MAMBA_ROOT_PREFIX}/bin/mamba" --version
  exit 0
fi
mkdir -p "${SF_CACHE}/bootstrap"
curl -fL --retry 3 --silent --show-error \
  https://github.com/conda-forge/miniforge/releases/download/26.5.3-0/Miniforge3-Linux-x86_64.sh \
  -o "${SF_INSTALLER}"
echo "14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529  ${SF_INSTALLER}" \
  | sha256sum --check
bash "${SF_INSTALLER}" -b -p "${MAMBA_ROOT_PREFIX}"
