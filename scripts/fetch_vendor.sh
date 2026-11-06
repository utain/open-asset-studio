#!/usr/bin/env bash
# Clone the third-party source checkouts the workers need, each at the commit it was tested with.
#   scripts/fetch_vendor.sh                 # everything in vendor/vendor.lock
#   scripts/fetch_vendor.sh seed-vc MMAudio # only entries whose path contains one of these words (case-insensitive)
# Each project keeps its own licence: read it before you use that backend (see THIRD_PARTY_NOTICES.md).
# Model weights are not fetched here: they come from the Hugging Face cache (Models page -> fix command).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
LOCK=vendor/vendor.lock

want() {  # want <path> [filters...]; no filter -> everything
  local path=${1,,}; shift
  [ $# -eq 0 ] && return 0
  for w in "$@"; do [[ $path == *"${w,,}"* ]] && return 0; done
  return 1
}

matched=0
while read -r path url commit; do
  want "$path" "$@" || continue
  matched=$((matched + 1))
  # a checkout's .git is a directory, a submodule's is a file: both mean "already cloned"
  if [ -e "$path/.git" ]; then
    echo "= $path (present)"
  else
    echo "+ $path  <- $url @ ${commit:0:12}"
    mkdir -p "$(dirname "$path")"
    git clone --quiet --filter=blob:none "$url" "$path" </dev/null
  fi
  # the commit may be missing from an existing shallow or stale clone
  if ! git -C "$path" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$path" fetch --quiet origin "$commit" </dev/null
  fi
  git -C "$path" -c advice.detachedHead=false checkout --quiet "$commit" </dev/null
done < <(grep -vE '^\s*(#|$)' "$LOCK")

if [ "$matched" -eq 0 ]; then
  echo "No entry in $LOCK matches: $*" >&2
  exit 1
fi

# Post-steps some backends expect (see docs/research/audio_more/<group>.md for the reasons):
if want vendor/MeloTTS-ttslight "$@" && [ -d vendor/MeloTTS ] && [ ! -d vendor/MeloTTS-ttslight ]; then
  rsync -a --exclude .git vendor/MeloTTS/ vendor/MeloTTS-ttslight/   # ttslight backend: a plain copy, no .git
fi
if want vendor/Kimi-Audio "$@" && [ -d vendor/Kimi-Audio ]; then
  git -C vendor/Kimi-Audio submodule update --init --quiet </dev/null || true
fi

cat <<'EOF'
Done. The music workers link some vendor folders (vendor/ace_project, vendor/heartmula_ckpt,
vendor/magenta_home, YuE's xcodec_mini_infer) to Hugging Face snapshots themselves on first run,
so the model weights must be in the HF cache first. See docs/research/audio_more/music.md.
EOF
