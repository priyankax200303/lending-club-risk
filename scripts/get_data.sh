#!/usr/bin/env bash
# Downloads a pinned public processed copy of Lending Club's resolved loans into data/raw, then checks file hashes.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p data/raw
B=https://raw.githubusercontent.com/YonatanGafri/lending-club-mlops/77fad1952dc9c39f770e48ece451bc325d92ae25/data
curl -fsS -o data/raw/train_fe.parquet "$B/train_fe.parquet"
curl -fsS -o data/raw/test_fe.parquet "$B/test_fe.parquet"
echo "b2b861094ceb82ef5c8f343ea0a4153c38339b78d3a3975323725dda54c7830b  data/raw/train_fe.parquet" | sha256sum -c -
echo "7abd8463f948acab528b194e7911bd6dba5b38e20f7bf006ee32d60220403958  data/raw/test_fe.parquet" | sha256sum -c -
