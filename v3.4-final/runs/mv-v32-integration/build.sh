#!/bin/sh
set -eu
cd "$(dirname "$0")"
for component in primal levels route ng regions zone maskfleet window screen2; do
    if test -f "$component/build.sh"; then sh "$component/build.sh"; fi
done
