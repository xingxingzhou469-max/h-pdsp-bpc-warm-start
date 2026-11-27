#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
reference=${V33_V32:-../../mv-v32-integration}
mkdir -p build
artifact=build/subsequence.so
scratch="$artifact.$$.tmp"
trap 'rm -f "$scratch"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++20 -O3 -Wall -Wextra -fPIC -shared -I "$reference/levels" search.cpp -o "$scratch"
mv -f "$scratch" "$artifact"
