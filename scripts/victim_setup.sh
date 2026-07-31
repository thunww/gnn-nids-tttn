#!/bin/bash
# Cai dat cac dich vu can thiet tren may VICTIM (192.168.207.199) de co du lieu that cho 5
# kich ban tan cong Phase B/C. Xem docs/graphsage/05_demo_realtime_setup.md muc 4.3.
#
# Chay tren chinh may victim:
#   sudo bash scripts/victim_setup.sh
set -e

echo "== Cai dat SSH, Apache, vsftpd =="
sudo apt update
sudo apt install -y openssh-server apache2 vsftpd

echo "== Bat cac dich vu, tu khoi dong cung may =="
sudo systemctl enable --now ssh apache2 vsftpd

echo "== Tao user demo rieng cho kich ban brute-force (mat khau YEU CO Y, chi dung trong lab cach ly) =="
if ! id -u demo_target >/dev/null 2>&1; then
    sudo useradd -m -s /bin/bash demo_target
fi
echo "demo_target:Passw0rd123" | sudo chpasswd

echo "== Xong. Kiem tra: =="
sudo systemctl status ssh apache2 vsftpd --no-pager | grep -E "^\s*Active|●"
echo "Victim san sang: SSH (22), HTTP (80), FTP (21), user brute-force: demo_target / Passw0rd123"
