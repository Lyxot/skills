#!/bin/sh
# Installs tbshare-mount as root on the Mac that mounts the share, with its config and password. Once per machine:
#   sudo sh install-tbshare-helper.sh --share Share --user account [--interface bridge0] [--mountpoint /Volumes/Share]
# It asks for the account's SMB password itself. Then let your account run the helper without a password:
#   sudo visudo -f /etc/sudoers.d/tbshare
set -eu
[ "$(id -u)" = 0 ] || { echo "run it with sudo" >&2; exit 1; }
share= user= interface=bridge0 mountpoint=
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
    case $1 in
        --share) share=$2 ;;
        --user) user=$2 ;;
        --interface) interface=$2 ;;
        --mountpoint) mountpoint=$2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift 2
done
[ -n "$share" ] && [ -n "$user" ] || { echo "--share and --user are required" >&2; exit 2; }
mountpoint=${mountpoint:-/Volumes/$share}
here=$(cd "$(dirname "$0")" && pwd)
# A NOPASSWD program under a directory anyone but root can write is a root shell for them.
root_only() {
    [ ! -L "$1" ] && [ "$(stat -f %Su "$1")" = root ] && [ $(( 0$(stat -f %Lp "$1") & 022 )) = 0 ] &&
        [ -z "$(ls -led "$1" | sed 1d)" ] ||
        { echo "$1 must be a real directory owned by root, writable only by root, with no ACL" >&2; exit 1; }
}
for d in / /usr /usr/local; do root_only "$d"; done
for d in /usr/local/libexec /usr/local/etc /usr/local/etc/tbshare; do
    [ -e "$d" ] || [ -L "$d" ] || mkdir -m 755 "$d"
    root_only "$d"
done
dest=/usr/local/libexec/tbshare-mount
rm -f "$dest"
install -o root -g wheel -m 755 "$here/tbshare-mount" "$dest"
conf=/usr/local/etc/tbshare/config
umask 022
printf '[tbshare]\nshare = %s\nuser = %s\ninterface = %s\nmountpoint = %s\n' \
    "$share" "$user" "$interface" "$mountpoint" > "$conf"
chown root:wheel "$conf"
chmod 644 "$conf"
printf 'SMB password of %s, for the share %s: ' "$user" "$share"
trap 'stty echo' EXIT INT TERM
stty -echo
IFS= read -r password
stty echo
echo
[ -n "$password" ] || { echo "no password given" >&2; exit 1; }
# The helper types it on a terminal, where control characters edit the line instead.
case $password in *[[:cntrl:]]*) echo "the password has a control character the helper cannot type" >&2; exit 1 ;; esac
secret=/usr/local/etc/tbshare/secret
umask 077
printf '%s\n' "$password" > "$secret"
chown root:wheel "$secret"
chmod 600 "$secret"
echo "installed $dest, $conf and $secret. Add this with: sudo visudo -f /etc/sudoers.d/tbshare"
echo "${SUDO_USER:-$(logname)} ALL=(root) NOPASSWD: $dest \"\""
