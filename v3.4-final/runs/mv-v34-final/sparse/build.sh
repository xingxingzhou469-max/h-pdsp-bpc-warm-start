#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
artifact=price.so
scratch="$artifact.$$.tmp"
trap 'rm -f "$scratch"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++20 -O3 -Wall -Wextra -fPIC -shared price.cpp -o "$scratch"
mv -f "$scratch" "$artifact"
