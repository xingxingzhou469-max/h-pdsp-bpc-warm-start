#!/bin/sh
set -eu
cd "$(dirname "$0")"
"${CXX:-c++}" -std=c++17 -O3 -fPIC -shared screen.cpp -o screen_kernel.so
