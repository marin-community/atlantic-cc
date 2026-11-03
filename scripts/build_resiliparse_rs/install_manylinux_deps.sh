#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Add what build.sh needs to a quay.io/pypa/manylinux_2_28_x86_64 image: git (if absent) and rustup
# with the pinned Rust toolchain. The image already has gcc-toolset-14, curl, tar and CPython 3.12
# (/opt/python/cp312-cp312). Run as root; installs rustup under $HOME (/root).
set -euo pipefail

RUSTUP_VERSION="1.29.0"
RUSTUP_INIT_SHA256="4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
RUST_TOOLCHAIN="1.98.0"

command -v git >/dev/null || dnf install -y --setopt=install_weak_deps=False git

# Not /tmp: it is mounted noexec on some cluster workers.
installer="$HOME/rustup-init"
curl -fsSL --retry 3 -o "$installer" \
  "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/x86_64-unknown-linux-gnu/rustup-init"
echo "$RUSTUP_INIT_SHA256  $installer" | sha256sum -c -
chmod +x "$installer"
"$installer" -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
rm -f "$installer"
