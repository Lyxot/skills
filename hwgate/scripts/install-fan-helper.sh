#!/bin/sh
# Installs hwgate-fan as root, so hwgate can pin the fans for --stable runs. Once per machine:
#   sudo sh install-fan-helper.sh [--percent N]
# then let your account run it without a password: sudo visudo -f /etc/sudoers.d/hwgate-fan
# --percent pins the fans at N% (75-100) of their maximum and writes hwgate-fan.conf beside the helper;
# without it an existing setting is kept, and a first install pins at 100%.
set -eu
percent=
case "$#:${1-}" in
    2:--percent) percent=$2 ;;
    0:) ;;
    *) echo "usage: sudo sh $0 [--percent N]" >&2; exit 2 ;;
esac
case "$#:$percent" in
    0:|2:7[5-9]|2:[89][0-9]|2:100) ;;
    *) echo "--percent takes 75 to 100" >&2; exit 2 ;;
esac
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
if [ -n "$percent" ]; then
    echo "percent=$percent" > "$tmp/conf"
    # Replaced in one rename: a pin reading a missing conf would go to 100%.
    install -o root -g wheel -m 644 "$tmp/conf" "$dest.conf.new"
    mv -f "$dest.conf.new" "$dest.conf"
fi
"$dest" status | grep "^configured pin:" || true
echo "installed $dest. Add this with: sudo visudo -f /etc/sudoers.d/hwgate-fan"
echo "${SUDO_USER:-$(logname)} ALL=(root) NOPASSWD: $dest status, $dest max, $dest full, $dest auto"
