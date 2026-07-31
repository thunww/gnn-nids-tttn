#!/bin/bash
# Kich ban tan cong 1 -- Port scan, chay tren may attacker (Kali). Ghi lai chinh xac gio bat
# dau/ket thuc (bat buoc theo Phase C, doi chieu voi ket qua model sau nay).
#
# Dung: bash run_scenario_portscan.sh <victim_ip>
set -e
VICTIM="${1:-192.168.207.199}"

echo "=== KICH BAN: Port scan toi $VICTIM ==="
echo "Bat dau: $(date '+%Y-%m-%d %H:%M:%S')"
nmap -T5 -p- --min-rate 3000 "$VICTIM"
echo "Ket thuc: $(date '+%Y-%m-%d %H:%M:%S')"
