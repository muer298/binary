#!/usr/bin/env bash
set -euo pipefail
umask 077
key_file=/home/cn/.config/seraph/deepseek_api_key
if [[ ! -f "$key_file" || ! -r "$key_file" ]]; then
  echo 'API_KEY_AVAILABLE=false'
  exit 2
fi
IFS= read -r DEEPSEEK_API_KEY < "$key_file"
if [[ -z "$DEEPSEEK_API_KEY" ]]; then
  echo 'API_KEY_AVAILABLE=false'
  exit 2
fi
export DEEPSEEK_API_KEY
exec python3 /data/chaoni/mjy/resym_exp/packages/seraph_external_ssq_v0_4_0e_order_audit/external_ssq_order_client.py run
