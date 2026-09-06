#!/usr/bin/env bash
# Patches the vendored MRS headers, builds the generator, writes tests/golden/*.csv.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
mrs=$root/research/code_sources/mrs_multirotor_simulator/include/mrs_multirotor_simulator/uav_system

if [ ! -f "$mrs/multirotor_model.hpp" ]; then
  echo "MRS reference not found at $mrs" >&2
  echo "research/code_sources/ is gitignored; clone ctu-mrs/mrs_multirotor_simulator there." >&2
  exit 1
fi

eigen_inc=${EIGEN_INC:-$(brew --prefix eigen)/include}
boost_inc=${BOOST_INC:-$(brew --prefix boost)/include}

rm -rf "$here/build"
mkdir -p "$here/build" "$root/tests/golden"

# The whole tree is copied so uav_system.hpp picks up the patched model beside it.
cp -R "$mrs" "$here/build/uav_system"
patch -s "$here/build/uav_system/multirotor_model.hpp" < "$here/transpose.diff"

g++ -std=c++17 -O2 \
  -I"$eigen_inc" -I"$boost_inc" -I"$here/build/uav_system" \
  "$here/gen_golden.cpp" -o "$here/build/gen_golden"

"$here/build/gen_golden" "$root/tests/golden"
