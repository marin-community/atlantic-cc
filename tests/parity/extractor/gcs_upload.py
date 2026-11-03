# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a directory tree to GCS with only the standard library (for job images without gcsfs).

Uses the GCE metadata server's service-account token, so it only works on Google Cloud VMs.

    python gcs_upload.py LOCAL_DIR gs://bucket/prefix
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={name}"


def access_token() -> str:
    request = urllib.request.Request(TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["access_token"]


def upload_tree(local_dir: Path, gcs_prefix: str) -> list[str]:
    bucket, _, prefix = gcs_prefix.removeprefix("gs://").partition("/")
    token = access_token()
    uploaded = []
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        name = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
        url = UPLOAD_URL.format(bucket=bucket, name=urllib.parse.quote(name, safe=""))
        request = urllib.request.Request(
            url,
            data=path.read_bytes(),
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(request, timeout=300):
            pass
        uploaded.append(f"gs://{bucket}/{name}")
    return uploaded


if __name__ == "__main__":
    for url in upload_tree(Path(sys.argv[1]), sys.argv[2]):
        print("uploaded", url)
