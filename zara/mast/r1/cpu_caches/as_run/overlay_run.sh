#!/bin/bash
# overlay_run.sh "MYDIR=ZARAPATH;MYDIR2=ZARAPATH2;..." <command...>
# Inside its own mount namespace (call through unshare -rm): Zara's /fsx/zara.ercan and the official data read-only, then each
# MYDIR (writable, ours) mounted over ZARAPATH, so her unchanged scripts read and write our CPU-built caches.
set -uo pipefail
MAP=$1; shift
for d in /fsx/zara.ercan /fsx/hackathon_offsite; do mount --bind $d $d && mount -o remount,ro,bind $d || exit 3; done
IFS=';' read -ra PAIRS <<< "$MAP"
for p in "${PAIRS[@]}"; do [ -z "$p" ] && continue; src=${p%%=*}; dst=${p#*=}; mkdir -p $src; mount --bind $src $dst || { echo "overlay $src -> $dst failed"; exit 5; }; done
touch /fsx/zara.ercan/.rotest 2>/dev/null && { echo "WRITE NOT BLOCKED"; exit 4; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$@"
