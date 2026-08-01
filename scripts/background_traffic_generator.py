#!/usr/bin/env python3
"""Sinh traffic nen (nhieu flow ngan, da dang: TCP/UDP/ICMP) giua cac may trong mang ao demo,
dung de lam day nhanh cua so 2000 flow/cua so ma model can de suy luan 1 lan (xem
docs/graphsage/05_demo_realtime_setup.md muc 6.5 -- ly do can script nay).

Chay tren may attacker HOAC victim (khong phai monitor 192.168.207.200), nham vao (các) may
con lai trong mang ao -- moi lan ket noi/gui goi la 1 flow moi (Zeek gan uid rieng).

Vi du chay tren may victim (199), nham vao attacker (194):
    python3 background_traffic_generator.py 192.168.207.194 --rate 10

Chay ~200 giay (~3-4 phut) voi --rate 10 la du 2000 flow cho cua so dau tien.
"""

from __future__ import annotations

import argparse
import random
import socket
import subprocess
import time

COMMON_PORTS = [22, 53, 80, 443, 3306, 8080]


def tcp_probe(host: str, port: int, timeout: float = 0.3) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect_ex((host, port))
    except OSError:
        pass


def udp_probe(host: str, port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"ping", (host, port))
    except OSError:
        pass


def icmp_ping(host: str) -> None:
    subprocess.run(
        ["ping", "-c", "1", "-W", "1", host],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="Danh sach IP may khac trong mang ao demo")
    parser.add_argument("--rate", type=float, default=10, help="So flow/giay muon sinh (mac dinh 10)")
    args = parser.parse_args()

    interval = 1 / args.rate
    sent = 0
    print(f"Bat dau sinh traffic nen toi {args.targets}, ~{args.rate} flow/giay. Ctrl+C de dung.")
    try:
        while True:
            target = random.choice(args.targets)
            kind = random.choice(["tcp", "tcp", "udp", "icmp"])
            if kind == "tcp":
                tcp_probe(target, random.choice(COMMON_PORTS))
            elif kind == "udp":
                udp_probe(target, random.choice(COMMON_PORTS))
            else:
                icmp_ping(target)
            sent += 1
            if sent % 100 == 0:
                print(f"  da sinh ~{sent} flow")
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nDung lai. Tong so flow da sinh: {sent}")


if __name__ == "__main__":
    main()
