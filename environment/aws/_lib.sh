# shellcheck shell=bash
# Shared helpers for the scripts in environment/aws/. Each script sources this file; it is never
# run on its own.
#
# Documented, not run: the workshop authors have not run any of these scripts against AWS.
#
# Conventions every script follows:
#   --dry-run   print every command (aws, docker, ...) and run none of them; lookups print a
#               <placeholder> instead of calling AWS, so a dry run needs no credentials.
#   --yes       answer "yes" to the confirmation prompts of destructive steps.
#   AWS_REGION  required; every aws call gets an explicit --region "$AWS_REGION".
# Commands are printed to stderr with a leading "+", like `set -x`.

DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
# In a dry run, a lookup that asks "does this exist?" answers no (show the create path) unless
# a script sets this to 1 (teardown.sh, to show the delete path).
DRY_RUN_ASSUME_EXISTS="${DRY_RUN_ASSUME_EXISTS:-0}"
# Every resource the scripts create carries this tag; the IAM policies in iam/ rely on it.
PROJECT_TAG_KEY="Project"
PROJECT_TAG_VALUE="${PROJECT_TAG_VALUE:-mktstats-workshop}"

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$AWS_DIR/../.." && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*" >&2; }

# Print the leading comment block of the calling script as its help text.
usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p;}' "$0"; }

# Handle the flags every script accepts. Returns 1 for anything else.
common_flag() {
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes) ASSUME_YES=1 ;;
    -h | --help) usage; exit 0 ;;
    *) return 1 ;;
  esac
}

banner() {
  if [[ "$DRY_RUN" == 1 ]]; then
    note "Dry run: printing commands only; nothing is sent to AWS and nothing is built."
  fi
}

require_region() {
  [[ -n "${AWS_REGION:-}" ]] || die "AWS_REGION is not set. Example: AWS_REGION=eu-west-1 $0 --dry-run"
  [[ "$AWS_REGION" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]+$ ]] || die "AWS_REGION does not look like a Region code: $AWS_REGION"
}

require_var() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "$name is required (see --help)."
}

need_cmd() {
  if ! command -v "$1" > /dev/null 2>&1; then
    if [[ "$DRY_RUN" == 1 ]]; then
      note "warning: '$1' is not installed; a real run needs it."
    else
      die "'$1' is required but not installed."
    fi
  fi
}

# Print a command the way a shell would read it.
show() {
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } >&2
}

# Print a file that a command reads (JSON payloads), so a dry run shows exactly what is sent.
show_file() {
  note "  # contents of $1:"
  sed 's/^/  #   /' "$1" >&2
}

# run CMD...: print the command, then run it unless this is a dry run.
run() {
  show "$@"
  if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
  "$@"
}

# capture PLACEHOLDER CMD...: print the command and run it, printing its output. In a dry run,
# print PLACEHOLDER instead. Use as: x=$(capture "<thing>" aws ...)
capture() {
  local placeholder=$1
  shift
  show "$@"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$placeholder"
  else
    "$@"
  fi
}

# capture_quiet PLACEHOLDER CMD...: like capture, but discards CMD's stderr (lookups that are
# expected to fail when something does not exist yet). Use with `|| true`.
capture_quiet() {
  local placeholder=$1
  shift
  show "$@"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$placeholder"
  else
    "$@" 2> /dev/null
  fi
}

# probe CMD...: succeed if CMD succeeds (output discarded). Used for "does this exist?" checks.
probe() {
  show "$@"
  if [[ "$DRY_RUN" == 1 ]]; then
    [[ "$DRY_RUN_ASSUME_EXISTS" == 1 ]]
    return
  fi
  "$@" > /dev/null 2>&1
}

aws_run() { run aws "$@" --region "$AWS_REGION"; }
aws_capture() {
  local placeholder=$1
  shift
  capture "$placeholder" aws "$@" --region "$AWS_REGION"
}
aws_capture_quiet() {
  local placeholder=$1
  shift
  capture_quiet "$placeholder" aws "$@" --region "$AWS_REGION"
}
aws_probe() { probe aws "$@" --region "$AWS_REGION"; }

# confirm PROMPT: ask before a destructive step. --yes answers for you; a dry run never asks.
confirm() {
  local prompt=$1 answer
  if [[ "$DRY_RUN" == 1 ]]; then
    note "[dry-run] would ask: $prompt [y/N]"
    return 0
  fi
  if [[ "$ASSUME_YES" == 1 ]]; then
    note "$prompt [y/N] y (--yes)"
    return 0
  fi
  [[ -r /dev/tty ]] || die "Cannot ask for confirmation without a terminal. Rerun with --yes if you are sure."
  read -r -p "$prompt [y/N] " answer < /dev/tty
  [[ "$answer" == y || "$answer" == Y || "$answer" == yes ]]
}

# wait_for DESCRIPTION SECONDS INTERVAL CMD...: rerun CMD (which prints a status) until it prints
# a word from $WAIT_OK; fail on a word from $WAIT_FAIL or after SECONDS, printing $WAIT_HINT if
# set. A dry run returns at once.
wait_for() {
  local what=$1 limit=$2 every=$3 status waited=0 ok="${WAIT_OK:-}" fail="${WAIT_FAIL:-}"
  shift 3
  [[ -n "$ok" ]] || die "wait_for needs WAIT_OK"
  show "$@"
  note "  (repeated every ${every}s until one of: ${ok}; fails on: ${fail:-none})"
  if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
  while :; do
    status=$("$@" 2> /dev/null || true)
    if [[ -n "$status" && " $ok " == *" $status "* ]]; then
      note "  $what: $status"
      return 0
    fi
    if [[ -n "$fail" && -n "$status" && " $fail " == *" $status "* ]]; then
      die "$what: $status${WAIT_HINT:+. $WAIT_HINT}"
    fi
    if ((waited >= limit)); then die "$what: still '$status' after ${limit}s${WAIT_HINT:+. $WAIT_HINT}"; fi
    sleep "$every"
    waited=$((waited + every))
  done
}

make_tmpdir() {
  local base="${TMPDIR:-/tmp}"
  TMP_DIR="$(mktemp -d "${base%/}/mktstats-aws.XXXXXX")"
  trap 'rm -rf "$TMP_DIR"' EXIT
}
