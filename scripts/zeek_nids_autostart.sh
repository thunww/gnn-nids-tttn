#!/bin/bash
# Bat promiscuous mode cho card mang giam sat + khoi dong Zeek.
# Chay tu dong luc boot qua systemd unit scripts/zeek-nids-autostart.service.
# Tong quan luong demo: docs/README.md.
set -e

INTERFACE="ens33"
ZEEKCTL="/opt/zeek/bin/zeekctl"

ip link set "$INTERFACE" promisc on
"$ZEEKCTL" deploy
