#!/usr/bin/env bash
# =============================================================================
# local-runner.sh  —  DevSecOps Local Stage Runner
#
# Each stage runs inside the same
# Docker image used by CI, with the project and ci-scripts bind-mounted so
# the environment is bit-for-bit identical to what GitHub Actions sees.
#
# Usage:  ./local-runner.sh --help
# =============================================================================

set -uo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
BLUE=$'\033[0;34m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
RED=$'\033[0;31m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

# ── Logger ──────────────────────────────────────────────────────────────────
_ts()       { date +"%Y-%m-%d %H:%M:%S"; }
_log()      { local lvl="$1" clr="$2" msg="$3"; printf -v _pad "%-9s" "[$lvl]"; echo -e "${clr}${BOLD}${_pad}${NC} $(_ts) - ${msg}"; }
debug() { [[ "$LOG_LEVEL" == "DEBUG" ]] && _log "DEBUG"   "$BLUE"   "$1" >&2 || true; }
info()  { _log "INFO"    "$BLUE"   "$1"; }
warn()  { _log "WARN"    "$YELLOW" "$1"; }
error() { _log "ERROR"   "$RED"    "$1" >&2; }
ok()    { _log "SUCCESS" "$GREEN"  "$1"; }
fail()      { error "$1"; exit 1; }

LOG_LEVEL="ERROR"

# ── Compatibility ────────────────────────────────────────────────────────────
if [[ "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  error "bash 4+ required — current: ${BASH_VERSION}"
  if [[ "$(uname)" == "Darwin" ]]; then
    error "  → brew install bash"
    error "    Then invoke with: /opt/homebrew/bin/bash $(basename "$0") [OPTIONS]"
  fi
  exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
  command -v grealpath &>/dev/null \
    && realpath() { grealpath "$@"; } \
    || { error "realpath not found — brew install coreutils"; exit 1; }
fi

# ── Constants ───────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_MOUNT="/opt/workspace"
readonly REGISTRY="nexus.quilr.ai/ci-oss-images"
readonly CI_SCRIPTS_REPO="ci-scripts"
readonly CI_SCRIPTS_BRANCH="ci-security-tools-integration"
readonly OUTPUT_SUBDIR="devsecops-precheck-reports"
readonly IMG_SECRET_SCAN="ci-security-tools:python-3-12-slim-secret-scan"
readonly IMG_SCA="ci-security-tools:ubuntu-24-04-security-tools-sca"
readonly IMG_IAC="ci-security-tools:ubuntu-24-04-security-tools-iac"
readonly IMG_ARCH_SUFFIX=$( [[ "$(uname)" == "Darwin" ]] && echo "-arm64" || echo "" )
readonly -a STRAY_REPORTS=(
  gitleaks-report.html    gitleaks-report.json
  trivy-report.html       trivy-report.json
  shellcheck-report.html  shellcheck-report.xml
  shellcheck-report.json  checkstyle-pom.xml
  eslint-report.html      eslint-report.json
  flake8-report-html      flake8-report.json
  checkstyle-report-html  checkstyle-report.xml
  clippy-report.html      clippy-report.xml
  counts.json             ci-scripts
  artifacts               eslint.config.js
  shellcheck-report.json. checkstyle.xml
  target                  clippy-report.json
  .gitleaks.toml
)


debug "LOG_LEVEL       : ${LOG_LEVEL}"
debug "SCRIPT_DIR         : ${SCRIPT_DIR}" 
debug "WORKSPACE_MOUNT    : ${WORKSPACE_MOUNT}"        
debug "REGISTRY           : ${REGISTRY}"
debug "CI_SCRIPTS_REPO    : ${CI_SCRIPTS_REPO}"
debug "CI_SCRIPTS_BRANCH  : ${CI_SCRIPTS_BRANCH}"          
debug "OUTPUT_SUBDIR      : ${OUTPUT_SUBDIR}" 
debug "IMG_SECRET_SCAN    : ${IMG_SECRET_SCAN}" 
debug "IMG_SCA            : ${IMG_SCA}" 
debug "IMG_IAC            : ${IMG_IAC}" 

# ── Defaults ────────────────────────────────────────────────────────────────
CI_SCRIPTS_DIR=""          # empty = auto-clone from GitHub
PROJECT_DIR="$(pwd)"
PROJECT_NAME=""
OUTPUT_DIR=""
ORG="${GITHUB_ORG:-quilrbusiness}"
LINTER_LANGUAGE=""

debug "CI_SCRIPTS_DIR  : ${CI_SCRIPTS_DIR}"
debug "PROJECT_DIR     : ${PROJECT_DIR}"
debug "OUTPUT_DIR      : ${OUTPUT_DIR}"
debug "ORG             : ${ORG}"
debug "LINTER_LANGUAGE : ${LINTER_LANGUAGE}"

declare -a STAGES=()
declare -A STAGE_RESULTS=()
readonly -a ALL_STAGES=(secret-scan sca iac linter)
_CI_SCRIPTS_TMP=""

debug "ALL_STAGES       : ${ALL_STAGES}"
debug "_CI_SCRIPTS_TMP  : ${_CI_SCRIPTS_TMP}"

# Remove the temp clone and any stray report artefacts on any exit
_cleanup() {
  if [[ -n "$_CI_SCRIPTS_TMP" && -d "$_CI_SCRIPTS_TMP" ]]; then
    debug "Removing temp ci-scripts clone: $_CI_SCRIPTS_TMP"
    rm -rf "$_CI_SCRIPTS_TMP"
  fi

  if [[ -n "$PROJECT_DIR" && -d "$PROJECT_DIR" ]]; then
    for f in "${STRAY_REPORTS[@]}"; do
      local target="${PROJECT_DIR}/${f}"
      if [[ -e "$target" ]]; then
        debug "Removing stray report: $target"
        rm -rf "$target"
      fi
    done
  fi
}
trap _cleanup EXIT

_banner() {
  local title="$1" width=66 line
  line=$(printf '%*s' "$width" '' | tr ' ' '=')
  echo -e "\n${CYAN}${BOLD}${line}${NC}"
  printf "${CYAN}${BOLD}  %-62s${NC}\n" "$title"
  echo -e "${CYAN}${BOLD}${line}${NC}"
}

_header() {
  local name="$1"
  echo -e "\n${BOLD}${BLUE}  Stage: ${name}${NC}"
  echo -e "${BLUE}  $(printf '%*s' 56 '' | tr ' ' '-')${NC}"
}

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
  cat <<HELP

${BOLD}local-runner.sh${NC}  —  Run DevSecOps CI stages locally via Docker

${BOLD}USAGE${NC}
  $(basename "$0") [OPTIONS]

${BOLD}STAGES${NC}
  secret-scan   Gitleaks  — detect leaked secrets in source code
  sca           Trivy fs  — dependency vulnerability scan
  iac           Trivy cfg — infrastructure misconfiguration scan
  linter        Language-aware linting (auto-detects JS/TS/Python/Java/Shell/Rust)
  all           Run all four stages above  [default]

${BOLD}OPTIONS${NC}
  -s, --stage <name>           Stage to run; repeat for multiple (e.g. -s sca -s iac)
  -d, --project-dir <path>     Project to scan  [default: current directory]
  -c, --ci-scripts-dir <path>  Path to a local ci-scripts clone  [default: auto-clone from GitHub]
  -o, --org <name>             GitHub org / registry namespace  [default: quilrbusiness]
  -l, --log-level <level>      DEBUG | INFO | WARN | ERROR  [default: INFO]
      --language <lang>        Force linter language instead of auto-detect:
                               javascript | typescript | python | java | shell | rust | go
      --output-dir <path>      Report output directory  [default: <project>/artifacts]
  -h, --help                   Show this help and exit

${BOLD}EXAMPLES${NC}
  # Run only secret scan and SCA against a specific project
  $(basename "$0") -s secret-scan -s sca -d ~/projects/myapp

  # Lint a specific language with debug output
  $(basename "$0") -s linter --language python -l DEBUG

HELP
  exit 0
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -s|--stage)           STAGES+=("$2");          shift 2 ;;
      -d|--project-dir)     PROJECT_DIR="$2";         shift 2 ;;
      -c|--ci-scripts-dir)  CI_SCRIPTS_DIR="$2";      shift 2 ;;
      -l|--log-level)       LOG_LEVEL="${2^^}";       shift 2 ;;
         --language)        LINTER_LANGUAGE="${2,,}"; shift 2 ;;
         --output-dir)      OUTPUT_DIR="$2";          shift 2 ;;
      -h|--help)            usage ;;
      *) fail "Unknown option: '$1'  —  run with --help for usage" ;;
    esac
  done

  debug "# of Stages : ${#STAGES[@]}"
  [[ ${#STAGES[@]} -eq 0 ]] && STAGES=(all)


  for s in "${STAGES[@]}"; do
    if [[ "$s" == "all" ]]; then
      STAGES=("${ALL_STAGES[@]}")
      break
    fi
  done

  debug "# of Stages : ${#STAGES[@]}"
  debug "Stages      : ${STAGES}"
}

_resolve_paths() {
  PROJECT_DIR="$(realpath "$PROJECT_DIR")"
  PROJECT_NAME="$(basename "$PROJECT_DIR")"
  [[ -n "$CI_SCRIPTS_DIR" ]] && CI_SCRIPTS_DIR="$(realpath "$CI_SCRIPTS_DIR")"
  [[ -z "$OUTPUT_DIR"     ]] && OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_SUBDIR}"
  OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

  debug "PROJECT_DIR    : $PROJECT_DIR"
  debug "PROJECT_NAME   : $PROJECT_NAME"
  debug "CI_SCRIPTS_DIR : $CI_SCRIPTS_DIR"
  debug "OUTPUT_DIR     : $OUTPUT_DIR"
}

_clone_ci_scripts() {
  command -v git &>/dev/null \
    || fail "'git' is required to auto-clone ci-scripts. Install git or use --ci-scripts-dir."

  local ssh_url="git@github.com:${ORG}/${CI_SCRIPTS_REPO}.git"
  local https_url="https://github.com/${ORG}/${CI_SCRIPTS_REPO}.git"

  _CI_SCRIPTS_TMP=$(mktemp -d -t ci-scripts-XXXXXX)
  CI_SCRIPTS_DIR="$_CI_SCRIPTS_TMP"

  info "Cloning ${ORG}/${CI_SCRIPTS_REPO}@${CI_SCRIPTS_BRANCH} into ${CI_SCRIPTS_DIR}"

  debug "Trying SSH: $ssh_url"
  if git clone \
      --branch "$CI_SCRIPTS_BRANCH" \
      --depth  1 \
      --quiet  \
      "$ssh_url" \
      "$CI_SCRIPTS_DIR" 2>/dev/null; then
    ok "ci-scripts ready at: $CI_SCRIPTS_DIR  (via SSH)"
    return 0
  fi

  warn "SSH clone failed — falling back to HTTPS"
  debug "Trying HTTPS: $https_url"
  git clone \
    --branch "$CI_SCRIPTS_BRANCH" \
    --depth  1 \
    --quiet  \
    "$https_url" \
    "$CI_SCRIPTS_DIR" \
    || fail "Clone failed via both SSH and HTTPS: ${ORG}/${CI_SCRIPTS_REPO}  (check --org and network access)"

  ok "ci-scripts ready at: $CI_SCRIPTS_DIR  (via HTTPS)"
}

_print_config() {
  _banner "DevSecOps Local Runner"
  echo
  info "Project dir    : $PROJECT_DIR"
  info "ci-scripts dir : $CI_SCRIPTS_DIR${_CI_SCRIPTS_TMP:+  (auto-cloned)}"
  info "Output dir     : $OUTPUT_DIR"
  info "Registry org   : $ORG"
  info "Stages         : ${STAGES[*]}"
  info "Log level      : $LOG_LEVEL"
  echo
}

_check_os_compat() {
  local os
  os="$(uname)"
  local -a missing=()

  case "$os" in
    Darwin)
      command -v realpath &>/dev/null || missing+=("brew install coreutils      # provides realpath")
      command -v docker   &>/dev/null || missing+=("brew install --cask docker  # Docker Desktop")
      command -v jq       &>/dev/null || missing+=("brew install jq")
      command -v yq       &>/dev/null || missing+=("brew install yq")
      command -v git      &>/dev/null || missing+=("brew install git")

      if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing macOS dependencies — install with:"
        for hint in "${missing[@]}"; do error "  → $hint"; done
        exit 1
      fi
      ok "macOS compatibility checks passed"
      ;;

    Linux)
      command -v docker &>/dev/null || missing+=("sudo apt-get install -y docker.io  # or: https://docs.docker.com/engine/install/")
      command -v jq     &>/dev/null || missing+=("sudo apt-get install -y jq")
      command -v yq     &>/dev/null || missing+=("snap install yq                    # or: pip3 install yq")
      command -v git    &>/dev/null || missing+=("sudo apt-get install -y git")

      if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing Linux dependencies — install with:"
        for hint in "${missing[@]}"; do error "  → $hint"; done
        exit 1
      fi
      ok "Linux compatibility checks passed"
      ;;

    *)
      warn "Unsupported OS '${os}' — proceeding without OS-specific dependency checks"
      ;;
  esac
}

preflight_check() {
  _print_config
  _check_os_compat

  docker info &>/dev/null || fail "Docker daemon is not running — start Docker and try again."

  [[ -d "$PROJECT_DIR"                  ]] || fail "Project directory not found: $PROJECT_DIR"
  [[ -d "$CI_SCRIPTS_DIR/scripts/shell" ]] || fail "ci-scripts directory looks wrong: $CI_SCRIPTS_DIR"

  mkdir -p "$OUTPUT_DIR"
  ok "Preflight checks passed"
}

# ── GitHub Context ──────────────────────────────────────────────────────────
build_github_context() {
  local sha branch repo

  sha=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "local-$(date +%s)")
  branch=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD   2>/dev/null || echo "local")
  repo=$(basename "$PROJECT_DIR")

  jq -nc \
    --arg workspace        "${WORKSPACE_MOUNT}/${PROJECT_NAME}" \
    --arg sha              "$sha"             \
    --arg ref_name         "$branch"          \
    --arg repository       "${ORG}/${repo}"   \
    --arg repository_owner "$ORG"             \
    --arg run_id           "local-$(date +%s)" \
    --arg event_name       "push"             \
    --arg server_url       "https://github.com" \
    '{
      workspace:        $workspace,
      sha:              $sha,
      ref_name:         $ref_name,
      repository:       $repository,
      repository_owner: $repository_owner,
      run_id:           $run_id,
      event_name:       $event_name,
      server_url:       $server_url
    }'
}

# ── run_in_docker ───────────────────────────────────────────────────────────
# Mounts:
#   $PROJECT_DIR          → /opt/workspace/$PROJECT_NAME  (GITHUB_WORKSPACE)
#   $CI_SCRIPTS_DIR       → /opt/workspace/ci-scripts
#   $OUTPUT_DIR/<stage>   → /opt/workspace/artifacts
run_in_docker() {

  debug "── ${FUNCNAME[0]} ──"

  local stage_name="$1"
  local image="$2"
  local script="$3"   # absolute path inside the container
  shift 3
  local -a extra_envs=("$@")

  debug "stage_name : ${stage_name}"
  debug "image      : ${image}"
  debug "script     : ${script}"

  local ctx
  ctx=$(build_github_context)

  debug "ctx : ${ctx}"

  local stage_report="${OUTPUT_DIR}/${stage_name}"
  mkdir -p "$stage_report"

  debug "Image  : $image"
  debug "Script : $script"
  debug "Output : $stage_report"

  local -a docker_cmd=(
    docker run --rm
    --user "$(id -u):$(id -g)"
    --platform "linux/amd64"
    --name "devsecops-${stage_name//\//-}-$$"
    -v "${PROJECT_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}"
    -v "${CI_SCRIPTS_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts"
    -v "${stage_report}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/artifacts"
    -e "GITHUB_WORKSPACE=${WORKSPACE_MOUNT}/${PROJECT_NAME}"
    -e "LOG_LEVEL=${LOG_LEVEL}"
    -e "GITHUB_CONTEXT=${ctx}"
  )

  for env_pair in "${extra_envs[@]}"; do
    docker_cmd+=(-e "$env_pair")
  done

  docker_cmd+=("$image" bash "$script")

  debug "Docker cmd: ${docker_cmd[*]}"

  local t0 t1 elapsed rc=0
  t0=$(date +%s)

  if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
    "${docker_cmd[@]}" || rc=$?
  else
    "${docker_cmd[@]}" &>/dev/null || rc=$?
  fi
  
  t1=$(date +%s)
  elapsed=$(( t1 - t0 ))

  if [[ $rc -eq 0 ]]; then
    ok   "Stage '${stage_name}' passed  [${elapsed}s]"
  else
    warn "Stage '${stage_name}' finished with findings  [exit=${rc}, ${elapsed}s]"
    warn "Review ${stage_name} reports in: ${stage_report}/"
  fi

  return $rc
}

# ── Secret Scan ─────────────────────────────────────────────────────────────
run_secret_scan() {

  debug "── ${FUNCNAME[0]} ──"
  _header "Secret Scan  (Gitleaks)"
  
  local image="${REGISTRY}/${IMG_SECRET_SCAN}${IMG_ARCH_SUFFIX}"
  
  debug "image           : ${image}"
  debug "WORKSPACE_MOUNT : ${WORKSPACE_MOUNT}"

  debug "run_in_docker \"secret-scan\" \"$image\" \
    \"${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/secret_scan/secret_scan.sh\""
  run_in_docker "secret-scan" "$image" \
    "${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/secret_scan/secret_scan.sh"

}

# ── SCA ─────────────────────────────────────────────────────────────────────
run_sca() {

  debug "── ${FUNCNAME[0]} ──"
  _header "SCA  (Trivy — dependency vulns)"
  
  local image="${REGISTRY}/${IMG_SCA}"
  
  debug "image           : ${image}"
  debug "WORKSPACE_MOUNT : ${WORKSPACE_MOUNT}"

  debug "run_in_docker \"sca\" \"$image\" \
    \"${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/sca/sca.sh\" \
    \"TRIVY_CACHE_DIR=/tmp/trivy-cache\""
  run_in_docker "sca" "$image" \
    "${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/sca/sca.sh" \
    "TRIVY_CACHE_DIR=/tmp/trivy-cache"

}

# ── IAC ─────────────────────────────────────────────────────────────────────
run_iac() {

  debug "── ${FUNCNAME[0]} ──"
  _header "IAC  (Trivy — infra misconfigs)"
  
  local image="${REGISTRY}/${IMG_IAC}"
  
  debug "image           : ${image}"
  debug "WORKSPACE_MOUNT : ${WORKSPACE_MOUNT}"

  debug "run_in_docker \"iac\" \"$image\" \
    \"${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/iac/iac.sh\" \
    \"TRIVY_CACHE_DIR=/tmp/trivy-cache\""
  run_in_docker "iac" "$image" \
    "${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/iac/iac.sh" \
    "TRIVY_CACHE_DIR=/tmp/trivy-cache"

}

# ── Linter ──────────────────────────────────────────────────────────────────
# ── Resolve the fully-qualified image for a given language by reading
# ── config/yaml/linter/linter.yaml in ci-scripts
_image_for_language() {
  local lang="$1"
  local yaml="${CI_SCRIPTS_DIR}/config/yaml/linter/linter.yaml"

  local img tag
  img=$(yq -r ".language[\"${lang}\"].image"           "$yaml" 2>/dev/null)
  tag=$(yq -r ".language[\"${lang}\"][\"image-tag\"]"  "$yaml" 2>/dev/null)

  if [[ -z "$img" || "$img" == "null" || -z "$tag" || "$tag" == "null" ]]; then
    error "No image entry for language '$lang' in linter.yaml"
    return 1
  fi

  echo "${REGISTRY}/${img}:${tag}"
}

# ── Detect languages and build a matrix JSON by reading linter.yaml directly.
_get_linter_matrix() {
  local yaml="${CI_SCRIPTS_DIR}/config/yaml/linter/linter.yaml"

  # Build prune args from the exclude_dirs list in ${CI_SCRIPTS_DIR}/config/yaml/linter/linter.yaml
  local -a prune=()
  while IFS= read -r d; do
    [[ -n "$d" ]] && prune+=(-path "${PROJECT_DIR}/${d}" -prune -o)
  done < <(yq -r '.exclude_dirs[]?' "$yaml" 2>/dev/null)

  local -a include=()

  while IFS= read -r lang; do
    [[ -z "$lang" ]] && continue

    # Check each file pattern for this language
    local matched=false
    while IFS= read -r pattern; do
      [[ -z "$pattern" ]] && continue
      local hit
      hit=$(find "$PROJECT_DIR" "${prune[@]}" -name "$pattern" -print -quit 2>/dev/null)
      if [[ -n "$hit" ]]; then
        matched=true
        debug "[$lang] detected via: $hit"
        break
      fi
    done < <(yq -r ".language[\"${lang}\"].files[]?" "$yaml" 2>/dev/null)

    if $matched; then
      local img tag
      img=$(yq -r ".language[\"${lang}\"].image"          "$yaml" 2>/dev/null)
      tag=$(yq -r ".language[\"${lang}\"][\"image-tag\"]" "$yaml" 2>/dev/null)
      include+=("{\"language\":\"${lang}\",\"image\":\"${img}:${tag}\"}")
    fi
  done < <(yq -r '.language | keys[]' "$yaml" 2>/dev/null)

  [[ ${#include[@]} -eq 0 ]] && return 0

  printf '%s\n' "${include[@]}" | jq -s '{include: .}'
}

run_linter() {
  _header "Linter  (language-aware)"

  command -v yq &>/dev/null \
    || fail "Linter stage requires 'yq' — install it (snap install yq / brew install yq)."

  local -a langs=()
  local -A lang_images=()

  debug "LINTER_LANGUAGE : ${LINTER_LANGUAGE}"

  if [[ -n "$LINTER_LANGUAGE" ]]; then

    langs=("$LINTER_LANGUAGE")

    local image
    image=$(_image_for_language "$LINTER_LANGUAGE") \
      || { error "Cannot resolve image for '$LINTER_LANGUAGE'"; return 1; }
    lang_images["$LINTER_LANGUAGE"]="$image"
    debug "Language (forced): $LINTER_LANGUAGE  ->  $image"

  else
    debug "Detecting languages via get_lint_image.sh ..."

    local matrix_json
    matrix_json=$(_get_linter_matrix)

    if [[ -z "$matrix_json" ]]; then
      warn "No supported languages detected — skipping linter stage"
      return 0
    fi

    while IFS=$'\t' read -r lang img; do
      langs+=("$lang")
      lang_images["$lang"]="${REGISTRY}/${img}"
    done < <(echo "$matrix_json" | jq -r '.include[] | [.language, .image] | @tsv')
  fi

  if [[ ${#langs[@]} -eq 0 ]]; then
    warn "No supported languages detected — skipping linter stage"
    return 0
  fi

  info "Languages: ${langs[*]}"

  local overall_rc=0
  for lang in "${langs[@]}"; do
    local image="${lang_images[$lang]:-}"
    if [[ -z "$image" ]]; then
      warn "No linter image for '$lang' — skipping"
      continue
    fi


    if [[ "$lang" == "rust" && -z "${CLONE_TOKEN:-}" ]]; then
      error "CLONE_TOKEN is required for the Rust linter but is not set."
      error "  → export CLONE_TOKEN=<your-token>"
      exit 1
    elif [[ "$lang" != "rust" && -z "${CLONE_TOKEN:-}" ]];then
        CLONE_TOKEN=""
    fi

    local ctx stage_report t0 t1 elapsed rc=0
    ctx=$(build_github_context)
    stage_report="${OUTPUT_DIR}/linter-${lang}"
    mkdir -p "$stage_report"

    local platform="linux/amd64"
    if [[ "$lang" == "rust" && "$(uname)" == "Darwin" ]]; then     
      image=${image}${IMG_ARCH_SUFFIX}
      platform="linux/arm64"                                  
    fi 

    echo ""
    debug "  [$lang]  Image : $image"
    debug "  [$lang]  Output: $stage_report"

    t0=$(date +%s)

    debug "docker run --rm \
      --platform "${platform}" \
      --name \"devsecops-linter-${lang}-$$\" \
      -v \"${PROJECT_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}\" \
      -v \"${CI_SCRIPTS_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts\" \
      -v \"${stage_report}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/artifacts\" \
      -e \"GITHUB_WORKSPACE=${WORKSPACE_MOUNT}/${PROJECT_NAME}\" \
      -e \"LOG_LEVEL=${LOG_LEVEL}\" \
      -e \"GIT_BASE_URL=github.com/${ORG}\" \
      -e \"CLONE_TOKEN=$CLONE_TOKEN\" \
      -e \"GITHUB_CONTEXT=${ctx}\" \
      \"$image\" \
      bash \"${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/linter/linter.sh\" \"$lang\" || rc=$?"

    local -a linter_cmd=(
      docker run --rm
      --platform "${platform}"
      --user "$(id -u):$(id -g)"
      --name "devsecops-linter-${lang}-$$"
      -v "${PROJECT_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}"
      -v "${CI_SCRIPTS_DIR}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts"
      -v "${stage_report}:${WORKSPACE_MOUNT}/${PROJECT_NAME}/artifacts"
      -e "GITHUB_WORKSPACE=${WORKSPACE_MOUNT}/${PROJECT_NAME}"
      -e "GIT_BASE_URL=github.com/${ORG}"
      -e "CLONE_TOKEN=$CLONE_TOKEN"
      -e "MAVEN_CONFIG=/tmp/.m2"
      -e "LOG_LEVEL=${LOG_LEVEL}"
      -e "GITHUB_CONTEXT=${ctx}"
      "$image"
      bash "${WORKSPACE_MOUNT}/${PROJECT_NAME}/ci-scripts/scripts/shell/linter/linter.sh" "$lang"
    )

    if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
      "${linter_cmd[@]}" || rc=$?
    else
      "${linter_cmd[@]}" &>/dev/null || rc=$?
    fi

    t1=$(date +%s)
    elapsed=$(( t1 - t0 ))

    if [[ $rc -eq 0 ]]; then
      ok   "Linter '$lang' passed  [${elapsed}s]"
    else
      warn "Linter '$lang' finished with findings  [exit=${rc}, ${elapsed}s]"
      warn "Reports in: $stage_report/"
      overall_rc=$rc
    fi
  done

  return $overall_rc
}

# ── Summary ─────────────────────────────────────────────────────────────────
print_summary() {
  _banner "Summary"
  echo

  local any_fail=0
  local col_stage=20 col_status=12 col_exit=6 col_report=45
  local divider width=$(( col_stage + col_status + col_exit + col_report + 8 ))
  printf -v divider '%*s' "$width" ''
  divider="${divider// /─}"

  printf "  ${BOLD}${CYAN}%-*s  %-*s  %-*s  %-*s${NC}\n" \
    "$col_stage" "STAGE" "$col_status" "STATUS" "$col_exit" "EXIT" "$col_report" "REPORT PATH"
  echo -e "  ${CYAN}${divider}${NC}"

  for stage in "${!STAGE_RESULTS[@]}"; do
    local rc="${STAGE_RESULTS[$stage]}"
    local report_path="${OUTPUT_DIR}/${stage}"
    if [[ "$stage" == "linter" ]]; then
      local -a langs=()
      for d in "${OUTPUT_DIR}"/linter-*/; do
        [[ -d "$d" ]] && langs+=("$(basename "${d%/}" | cut -d- -f2-)")
      done
      if [[ ${#langs[@]} -gt 0 ]]; then
        local full_paths=()
        for lang in "${langs[@]}"; do
          full_paths+=("${OUTPUT_DIR}/linter-${lang}")
        done
        report_path="$(IFS=','; echo "${full_paths[*]}" | sed 's/,/, /g')"
      fi
    fi
    if [[ "$rc" -eq 0 ]]; then
      printf "  ${GREEN}%-*s  %-*s  %-*s  %-*s${NC}\n" \
        "$col_stage" "$stage" "$col_status" "PASSED" "$col_exit" "$rc" "$col_report" "$report_path"
    else
      printf "  ${RED}%-*s  %-*s  %-*s  %-*s${NC}\n" \
        "$col_stage" "$stage" "$col_status" "FINDINGS" "$col_exit" "$rc" "$col_report" "$report_path"
      any_fail=1
    fi
  done

  echo -e "  ${CYAN}${divider}${NC}"
  echo

  if [[ $any_fail -eq 0 ]]; then
    ok "All stages passed with no findings."
  else
    warn "One or more stages reported findings — review the reports above."
  fi
}

main() {
  parse_args "$@"
  _resolve_paths
  [[ -z "$CI_SCRIPTS_DIR" ]] && _clone_ci_scripts
  preflight_check

  for stage in "${STAGES[@]}"; do
    local rc=0
    case "$stage" in
      secret-scan) run_secret_scan || rc=$? ;;
      sca)         run_sca         || rc=$? ;;
      iac)         run_iac         || rc=$? ;;
      linter)      run_linter      || rc=$? ;;
      *)
        error "Unknown stage: '$stage'"
        error "Valid stages: ${ALL_STAGES[*]}  (or: all)"
        exit 1
        ;;
    esac
    STAGE_RESULTS["$stage"]=$rc
  done

  print_summary
}

main "$@"
