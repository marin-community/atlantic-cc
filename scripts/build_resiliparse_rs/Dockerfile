# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Portable Linux x86_64 build of resiliparse._extract_rs (glibc >= 2.28, manylinux_2_28). See
# docs/extractor.md. From the repository root:
#
#   docker build --platform linux/amd64 -t atlantic-cc-resiliparse-rs scripts/build_resiliparse_rs
#   docker run --rm --platform linux/amd64 -v "$PWD/dist:/out" atlantic-cc-resiliparse-rs
#
# The wheel, the production-layout artifact/ directory and the build logs land in ./dist.
FROM quay.io/pypa/manylinux_2_28_x86_64:2026.10.26-1@sha256:e0adf18521ad35192df733a6c62d35df7801957d24e0470238aacecfab2188a3

COPY install_manylinux_deps.sh /opt/atlantic-cc/
RUN bash /opt/atlantic-cc/install_manylinux_deps.sh
ENV PATH=/root/.cargo/bin:$PATH

COPY build.sh package_wheel.py /opt/atlantic-cc/
ENTRYPOINT ["bash", "/opt/atlantic-cc/build.sh", "--python", "/opt/python/cp312-cp312/bin/python3.12", "--work-dir", "/root/resiliparse_rs_build"]
CMD ["--out-dir", "/out"]
