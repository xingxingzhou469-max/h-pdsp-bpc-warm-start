#!/bin/sh
set -eu
cd "$(dirname "$0")"
temporary="mask_price.so.$$.tmp"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared price.cpp -o "$temporary"
mv "$temporary" mask_price.so
