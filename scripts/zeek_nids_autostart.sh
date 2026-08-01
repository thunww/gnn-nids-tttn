#!/bin/bash
# Bat promiscuous mode cho card mang giam sat + khoi dong Zeek.
# Chay tu dong luc boot qua systemd unit scripts/zeek-nids-autostart.service.
# Xem huong dan cai dat: docs/graphsage/05_demo_realtime_setup.md muc 4.2.
set -e

INTERFACE="ens33"
ZEEKCTL="/opt/zeek/bin/zeekctl"

ip link set "$INTERFACE" promisc on
"$ZEEKCTL" deploy
