#!/bin/sh
# Installs hwgate-fan as root, so hwgate can pin the fans for --stable runs. Once per machine:
#   sudo sh install-fan-helper.sh
# then let your account run it without a password: sudo visudo -f /etc/sudoers.d/hwgate-fan
set -eu
[ "$(id -u)" = 0 ] || { echo "run it with sudo" >&2; exit 1; }
here=$(cd "$(dirname "$0")" && pwd)
dir=/usr/local/libexec
dest=$dir/hwgate-fan
# A NOPASSWD binary under a directory anyone but root can write is a root shell for them.
root_only() {
    [ ! -L "$1" ] && [ "$(stat -f %Su "$1")" = root ] && [ $(( 0$(stat -f %Lp "$1") & 022 )) = 0 ] &&
        [ -z "$(ls -led "$1" | sed 1d)" ] ||
        { echo "$1 must be a real directory owned by root, writable only by root, with no ACL" >&2; exit 1; }
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
clang -O2 -Wall -framework IOKit -o "$tmp/hwgate-fan" "$here/hwgate-fan.c"
# Each directory is checked before anything is created or written under it.
for d in / /usr /usr/local; do root_only "$d"; done
[ -e "$dir" ] || [ -L "$dir" ] || mkdir -m 755 "$dir"
root_only "$dir"
rm -f "$dest"
install -o root -g wheel -m 755 "$tmp/hwgate-fan" "$dest"
echo "installed $dest. Add this with: sudo visudo -f /etc/sudoers.d/hwgate-fan"
echo "${SUDO_USER:-$(logname)} ALL=(root) NOPASSWD: $dest status, $dest max, $dest auto"
