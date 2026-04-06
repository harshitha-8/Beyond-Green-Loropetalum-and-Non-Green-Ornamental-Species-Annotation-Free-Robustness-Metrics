#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
  echo "Usage: bash scripts/cleanup_macos_artifacts.sh [--dry-run|--apply]"
  exit 1
fi

files=()
while IFS= read -r -d '' file; do
  files+=("$file")
done < <(find . -path "./.git" -prune -o -name "._*" -print0 | sort -z)

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No macOS artifact files found."
  exit 0
fi

echo "Found ${#files[@]} macOS artifact file(s):"
for file in "${files[@]}"; do
  printf '  %s\n' "$file"
done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with --apply to delete these files."
  exit 0
fi

rm -f -- "${files[@]}"
echo "Cleanup complete."
