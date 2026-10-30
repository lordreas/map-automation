# Projector network setup (2026-10-27)

The replacement 64 GB card uses the official A733 Debian KDE R6 image.
The board hostname is `projector`, advertised by Avahi as `projector.local`.
The laptop's SSH alias `projector` uses that name and the `map` account,
with the laptop's existing Ed25519 public key. Because multicast name lookups
were intermittent on Dufourspitze, the laptop's SSH ProxyCommand first connects
to the verified board's stable IPv6 link-local address on the active Wi-Fi
interface, then falls back to mDNS. OpenSSH still verifies the pinned host key. No Git access is configured.
The old fixed hotspot address is no longer used.

Both NetworkManager profiles use IPv4 DHCP and automatic connection:

| Network | Autoconnect priority |
| --- | ---: |
| Familio | 100 |
| Dufourspitze | 10 |

Credentials are stored only in root-owned NetworkManager keyfiles on the board.
Do not copy them into this repository. Priority chooses among available networks
when connecting; NetworkManager does not preempt a working connection solely
because a higher-priority network appears.

```sh
ssh projector
# The control page follows DHCP changes too:
# http://projector.local:8080
ssh projector 'nmcli -f NAME,TYPE,DEVICE connection show --active; ip -brief address'
```

Android can choose a different hotspot subnet between sessions. A fixed client
address does not solve that. A router DHCP reservation is preferable if a stable
numeric address is needed on Dufourspitze; configure it on the router, not by
guessing an unused address on the board. `projector.local` avoids needing it.
The Radxa WLAN MAC observed during setup is `90:30:d6:03:56:a4`.

mDNS requires that the laptop and board are on the same local network and that
multicast is permitted. If a network blocks mDNS, check its DHCP leases or scan
the current local subnet; don't reuse an address from another hotspot session.

## Replacement card deployment

Runtime files are under `/home/map/map-automation`; updates use rsync and exclude
`.git`, local virtual environments, native binaries, settings and caches.
The service uses the existing `deploy/radxa-a7z.service`, native KMS/DRM on VT1,
and the validated PowerVR libraries. The desktop remains installed. VT2/VT3 are
kept available for local login as `map`. The old `radxa` account is locked,
its login shell disabled, and its original home retained as a backup.

```sh
ssh projector 'sudo systemctl stop map-projection; sudo systemctl start sddm'
ssh projector 'sudo systemctl stop sddm; sudo systemctl start map-projection'
```

Debian 11's original security package URLs returned 404s. The three standard
Bullseye source entries were moved to the official signed Debian archive, with
`check-valid-until=no` scoped to those archived entries. Signature verification
remains enabled; Radxa package sources and GPU drivers are retained. Original
source files have `.before-projector` backups. This does not extend Debian 11's
security-support lifetime.

The latest known physical calibration is recovered from the 2026-10-25 full
pipeline benchmark. It should be rechecked visually after reinstalling.


## Local console toggle

Press and release **Ctrl+left Alt** together, without another key, to stop the
projection service and show a text login on VT2. Press and release them again
to start projection on VT1. Other Ctrl+Alt shortcuts cancel this chord, so
Ctrl+Alt+F2 and Ctrl+Alt+Delete keep their normal meanings. AltGr is excluded.
Keyboard hot-plug is supported. Keystrokes are neither recorded nor grabbed.

The root-owned `map-projection-console.service` runs the deployed
`console_toggle.py`. Entering the console saves current calibration and displayed
time and pauses playback. Returning starts a fresh graphics process so DRM
ownership is cleanly released and reacquired. No desktop session is started.
The existing `map` login uses the previous account's password; no password is
stored in this repository. With no keyboard attached, the listener waits.

Both Familio and Dufourspitze were tested with DHCP. The SSH alias continued to
work after switching between IPv4 subnets. Board Wi-Fi power saving is disabled
in both profiles to improve responsiveness. On 2026-10-27, local transfers over
Familio were substantially faster than over Dufourspitze, so the horizon bank
was restored over Familio.


On the Pixel 7, Chrome initially failed with `ERR_TOO_MANY_RETRIES` and no TCP
connection reached the board. Granting Chrome's Android **Nearby devices**
permission resolved access; no server/firewall change was necessary.
