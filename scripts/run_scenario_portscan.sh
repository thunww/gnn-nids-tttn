#!/bin/bash
# Kich ban tan cong 1 -- Port scan, chay tren may attacker (Kali). Ghi lai chinh xac gio bat
# dau/ket thuc (bat buoc theo Phase C, doi chieu voi ket qua model sau nay).
#
# QUAN TRONG (phat hien 2026-07-31): may monitor (200) chi co 2 CPU / 3.3GB RAM -- toc do scan
# qua nhanh (-T5 --min-rate 3000) lam Zeek qua tai, mat 87-99% goi tin that (xem
# capture_loss.log), khien flow ghi lai rong, model khong co du lieu de phat hien dung. Toc do
# duoi day da giam de Zeek theo kip.
#
# Dung: bash run_scenario_portscan.sh <victim_ip>
set -e
VICTIM="${1:-192.168.207.199}"

echo "=== KICH BAN: Port scan toi $VICTIM ==="
echo "Bat dau: $(date '+%Y-%m-%d %H:%M:%S')"
nmap -T3 -p- --min-rate 50 --max-rate 200 "$VICTIM"
echo "Ket thuc: $(date '+%Y-%m-%d %H:%M:%S')"
