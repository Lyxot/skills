#!/bin/sh
set -eu
set -f

usage() {
  printf '%s\n' 'Usage: download_modelscope_to_hf_cache.sh [--dry-run] [--cache-dir DIR] [--hf-endpoint URL] [--reserve-gib N] [--ms-branch NAME] ORG/REPO' >&2
  exit 2
}

dry_run=0
cache_root=''
hf_endpoint=${HF_ENDPOINT:-https://huggingface.co}
ms_endpoint='https://modelscope.cn'
reserve_gib=20
ms_branch='master'
repo=''

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --cache-dir) [ "$#" -ge 2 ] || usage; cache_root=$2; shift 2 ;;
    --hf-endpoint) [ "$#" -ge 2 ] || usage; hf_endpoint=$2; shift 2 ;;
    --reserve-gib) [ "$#" -ge 2 ] || usage; reserve_gib=$2; shift 2 ;;
    --ms-branch) [ "$#" -ge 2 ] || usage; ms_branch=$2; shift 2 ;;
    --) shift; break ;;
    -*) usage ;;
    *) [ -z "$repo" ] || usage; repo=$1; shift ;;
  esac
done
[ -n "$repo" ] && [ "$#" -eq 0 ] || usage

repo_org=${repo%%/*}
repo_name=${repo#*/}
[ -n "$repo_org" ] && [ -n "$repo_name" ] && [ "$repo_name" != "$repo" ] || {
  printf '%s\n' 'Repository ID must be ORG/REPO.' >&2
  exit 2
}
case "$repo_org" in *[!A-Za-z0-9._-]*) printf '%s\n' 'Invalid repository owner.' >&2; exit 2 ;; esac
case "$repo_name" in *[!A-Za-z0-9._-]*|*/*) printf '%s\n' 'Invalid repository name.' >&2; exit 2 ;; esac
case "$reserve_gib" in ''|*[!0-9]*) printf '%s\n' '--reserve-gib must be a non-negative integer.' >&2; exit 2 ;; esac

if [ -z "$cache_root" ]; then
  if [ -n "${HF_HUB_CACHE:-}" ]; then
    cache_root=$HF_HUB_CACHE
  elif [ -n "${HF_HOME:-}" ]; then
    cache_root=$HF_HOME/hub
  else
    cache_root=$HOME/.cache/huggingface/hub
  fi
fi
[ -n "$cache_root" ] && [ "$cache_root" != / ] || {
  printf '%s\n' 'Unsafe cache directory.' >&2
  exit 2
}
hf_endpoint=${hf_endpoint%/}
reserve_bytes=$(awk -v gib="$reserve_gib" 'BEGIN {printf "%.0f", gib * 1073741824}')

for command_name in curl jq git awk find sed wc stat mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'Missing command: %s\n' "$command_name" >&2
    exit 127
  }
done

file_bytes() {
  if stat -f %z "$1" >/dev/null 2>&1; then
    stat -f %z "$1"
  else
    stat -c %s "$1"
  fi
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

encode_path() {
  ep_input=$1
  ep_old_ifs=$IFS
  IFS=/
  set -- $ep_input
  IFS=$ep_old_ifs
  ep_result=''
  for ep_segment do
    ep_encoded=$(jq -rn --arg value "$ep_segment" '$value | @uri')
    if [ -n "$ep_result" ]; then ep_result=$ep_result/$ep_encoded; else ep_result=$ep_encoded; fi
  done
  printf '%s\n' "$ep_result"
}

tmp_dir=$(mktemp -d)
lock_dir=''
cleanup() {
  if [ -n "$lock_dir" ] && [ -d "$lock_dir" ]; then rmdir "$lock_dir" 2>/dev/null || :; fi
  if [ -d "$tmp_dir" ]; then
    find "$tmp_dir" -type f -exec rm -f {} \;
    find "$tmp_dir" -depth -type d -exec rmdir {} \; 2>/dev/null || :
  fi
}
trap cleanup 0
trap 'exit 130' 2
trap 'exit 143' 15

hf_api=$tmp_dir/hf.json
hf_files=$tmp_dir/hf-files.tsv
ms_files=$tmp_dir/modelscope-files.tsv
if ! curl -fsSL "$hf_endpoint/api/models/$repo?blobs=true" -o "$hf_api"; then
  printf 'The same repository ID is unavailable from %s: %s\n' "$hf_endpoint" "$repo" >&2
  exit 2
fi
hf_sha=$(jq -er '.sha' "$hf_api")
jq -r '.siblings[] | [.rfilename, (.size|tostring), (.lfs.sha256 // "-"), .blobId] | @tsv' "$hf_api" > "$hf_files"

ms_sha=$(git ls-remote "https://www.modelscope.cn/$repo.git" "refs/heads/$ms_branch" | awk 'NR == 1 {print $1}')
[ -n "$ms_sha" ] || {
  printf 'ModelScope branch unavailable: %s@%s\n' "$repo" "$ms_branch" >&2
  exit 2
}

: > "$ms_files"
roots_file=$tmp_dir/modelscope-roots.txt
printf '%s\n' '__ROOT__' > "$roots_file"
root_number=1
while :; do
  root_count=$(wc -l < "$roots_file" | awk '{print $1}')
  [ "$root_number" -le "$root_count" ] || break
  root_marker=$(sed -n "${root_number}p" "$roots_file")
  if [ "$root_marker" = '__ROOT__' ]; then root=''; else root=$root_marker; fi
  root_json=$tmp_dir/root-$root_number.json
  curl -fsSLG "$ms_endpoint/api/v1/models/$repo/repo/files" \
    --data-urlencode "Revision=$ms_sha" --data-urlencode "Root=$root" -o "$root_json"
  [ "$(jq -r '.Code' "$root_json")" = 200 ] || {
    printf 'ModelScope API failed at: %s\n' "$root" >&2
    exit 2
  }
  jq -r '.Data.Files[] | select(.Type == "tree") | .Path' "$root_json" >> "$roots_file"
  jq -r '.Data.Files[] | select(.Type != "tree") | [.Path, (.Size|tostring), .Sha256] | @tsv' "$root_json" >> "$ms_files"
  root_number=$((root_number + 1))
done

hf_bytes=$(jq '[.siblings[].size] | add // 0' "$hf_api")
ms_bytes=$(awk -F '\t' '{sum += $2} END {printf "%.0f", sum}' "$ms_files")
printf 'Repository: %s\n' "$repo"
printf 'Hugging Face commit: %s\n' "$hf_sha"
printf 'ModelScope commit: %s\n' "$ms_sha"
printf 'Hugging Face bytes: %s\n' "$hf_bytes"
printf 'ModelScope bytes: %s\n' "$ms_bytes"
printf 'Cache directory: %s\n' "$cache_root"
[ "$dry_run" -eq 0 ] || exit 0

for command_name in aria2c hf; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'Missing command: %s\n' "$command_name" >&2
    exit 127
  }
done

repo_cache=$cache_root/models--$repo_org--$repo_name
mkdir -p "$repo_cache/blobs" "$repo_cache/refs" "$repo_cache/snapshots/$ms_sha" "$repo_cache/snapshots/$hf_sha"
candidate_lock=$repo_cache/.modelscope-to-hf-cache.lock
if ! mkdir "$candidate_lock" 2>/dev/null; then
  printf 'Another download holds the repository lock: %s\n' "$repo" >&2
  exit 3
fi
lock_dir=$candidate_lock

link_blob() {
  link_snapshot=$1
  link_path=$2
  link_name=$3
  link_destination=$link_snapshot/$link_path
  link_relative=../../blobs/$link_name
  link_depth=$(printf '%s\n' "$link_path" | awk -F/ '{print NF - 1}')
  link_index=0
  while [ "$link_index" -lt "$link_depth" ]; do
    link_relative=../$link_relative
    link_index=$((link_index + 1))
  done
  mkdir -p "$(dirname "$link_destination")"
  ln -sf "$link_relative" "$link_destination"
}

download_file() {
  dl_url=$1
  dl_output=$2
  dl_expected_size=$3
  dl_free_kib=$(df -Pk "$cache_root" | awk 'NR == 2 {print $4}')
  dl_allowed=$(awk -v free_kib="$dl_free_kib" -v expected="$dl_expected_size" -v reserve="$reserve_bytes" \
    'BEGIN {print ((free_kib * 1024 - expected) >= reserve) ? 1 : 0}')
  [ "$dl_allowed" -eq 1 ] || {
    printf 'Download would leave less than %s GiB free.\n' "$reserve_gib" >&2
    exit 3
  }
  aria2c --continue=true --max-connection-per-server=8 --split=8 \
    --min-split-size=8M --file-allocation=none --auto-file-renaming=false \
    --allow-overwrite=true --max-tries=12 --retry-wait=5 \
    --connect-timeout=30 --timeout=60 --summary-interval=0 \
    --show-console-readout=false --console-log-level=warn --download-result=hide \
    --dir="${dl_output%/*}" --out="${dl_output##*/}" "$dl_url"
  [ "$(file_bytes "$dl_output")" = "$dl_expected_size" ] || {
    printf 'Size mismatch: %s\n' "$dl_output" >&2
    exit 4
  }
}

tab=$(printf '\t')
while IFS="$tab" read -r file_path file_size ms_sha256; do
  hf_lfs=$(jq -r --arg file_path "$file_path" \
    '[.siblings[] | select(.rfilename == $file_path) | (.lfs.sha256 // "-")][0] // "-"' "$hf_api")
  blob_name=''
  if [ "$hf_lfs" != '-' ]; then
    blob_name=$ms_sha256
  else
    hf_blob=$(jq -r --arg file_path "$file_path" \
      '[.siblings[] | select(.rfilename == $file_path) | .blobId][0] // "-"' "$hf_api")
    if [ "$hf_blob" != '-' ] && [ -f "$repo_cache/blobs/$hf_blob" ] && \
       [ "$(file_bytes "$repo_cache/blobs/$hf_blob")" = "$file_size" ] && \
       [ "$(sha256_file "$repo_cache/blobs/$hf_blob")" = "$ms_sha256" ]; then
      blob_name=$hf_blob
    fi
  fi

  if [ -n "$blob_name" ] && [ -f "$repo_cache/blobs/$blob_name" ] && [ "$hf_lfs" != '-' ] && \
     [ "$(sha256_file "$repo_cache/blobs/$blob_name")" != "$ms_sha256" ]; then
    blob_name=''
  fi

  if [ -z "$blob_name" ] || [ ! -f "$repo_cache/blobs/$blob_name" ] || \
     [ "$(file_bytes "$repo_cache/blobs/$blob_name" 2>/dev/null || printf 0)" != "$file_size" ]; then
    staging=$repo_cache/blobs/.$ms_sha256.incomplete
    encoded_path=$(encode_path "$file_path")
    download_file "$ms_endpoint/models/$repo/resolve/$ms_sha/$encoded_path" "$staging" "$file_size"
    [ "$(sha256_file "$staging")" = "$ms_sha256" ] || {
      printf 'ModelScope checksum mismatch: %s\n' "$file_path" >&2
      exit 4
    }
    if [ "$hf_lfs" != '-' ]; then blob_name=$ms_sha256; else blob_name=$(git hash-object --no-filters "$staging"); fi
    if [ -f "$repo_cache/blobs/$blob_name" ]; then rm -f "$staging"; else mv "$staging" "$repo_cache/blobs/$blob_name"; fi
  fi
  link_blob "$repo_cache/snapshots/$ms_sha" "$file_path" "$blob_name"
done < "$ms_files"
printf '%s\n' "$ms_sha" > "$repo_cache/refs/modelscope"

while IFS="$tab" read -r file_path file_size hf_lfs hf_blob; do
  if [ "$hf_lfs" != '-' ]; then blob_name=$hf_lfs; else blob_name=$hf_blob; fi
  valid_blob=0
  if [ -f "$repo_cache/blobs/$blob_name" ] && [ "$(file_bytes "$repo_cache/blobs/$blob_name")" = "$file_size" ]; then
    if [ "$hf_lfs" != '-' ]; then
      [ "$(sha256_file "$repo_cache/blobs/$blob_name")" = "$blob_name" ] && valid_blob=1
    else
      [ "$(git hash-object --no-filters "$repo_cache/blobs/$blob_name")" = "$blob_name" ] && valid_blob=1
    fi
  fi
  if [ "$valid_blob" -eq 0 ]; then
    staging=$repo_cache/blobs/.$blob_name.incomplete
    encoded_path=$(encode_path "$file_path")
    download_file "$hf_endpoint/$repo/resolve/$hf_sha/$encoded_path" "$staging" "$file_size"
    if [ "$hf_lfs" != '-' ]; then
      [ "$(sha256_file "$staging")" = "$blob_name" ] || { printf 'Hugging Face checksum mismatch: %s\n' "$file_path" >&2; exit 4; }
    else
      [ "$(git hash-object --no-filters "$staging")" = "$blob_name" ] || { printf 'Hugging Face Git blob mismatch: %s\n' "$file_path" >&2; exit 4; }
    fi
    mv "$staging" "$repo_cache/blobs/$blob_name"
  fi
  link_blob "$repo_cache/snapshots/$hf_sha" "$file_path" "$blob_name"
done < "$hf_files"

printf '%s\n' "$hf_sha" > "$repo_cache/refs/main"
HF_ENDPOINT=$hf_endpoint hf cache verify "$repo" --revision "$hf_sha" \
  --cache-dir "$cache_root" --fail-on-missing-files --fail-on-extra-files
find "$repo_cache/blobs" -maxdepth 1 -type f -name '*.incomplete' -exec rm -f {} \;
printf 'Complete: %s\n' "$repo"
