---
name: tbshare
description: Mount another Mac's SMB share over the Thunderbolt Bridge, not Wi-Fi or a slower link, and check that it reads at speed. Use on the Mac that mounts the share, before a job loads models or data from it (for example a Hugging Face cache on the other Mac's external disk), or when such a share is slow or not mounted.
---

# tbshare

By name, macOS often reaches an SMB server over Wi-Fi or a slow link instead of the Thunderbolt Bridge. `tbshare` mounts the share at the other Mac's address on the bridge and checks which link really carries it.

```bash
~/.agents/skills/tbshare/scripts/tbshare ensure   # mount over the bridge if needed, then check the speed
~/.agents/skills/tbshare/scripts/tbshare status   # report only
```

From the other Mac: `ssh <client> '~/.agents/skills/tbshare/scripts/tbshare ensure'`.

`ensure` mounts the share if it is not mounted. A share on another link is remounted over the bridge only when the bridge answers and no process has a file open on the share; otherwise it is left as it is. A share already on the bridge is never remounted, even when slow.

| Code | Meaning | What to do |
|---|---|---|
| 0 | read over the bridge at 500 MB/s or more | load from the share |
| 3 | not mounted (`status` only) | run `ensure` |
| 4 | on another link or slow, and left mounted as it is (for example, the bridge does not answer) | copy what the job needs instead, and tell the user what it printed |
| 5 | on another link, with files open on it | wait for those jobs, or copy instead; never unmount it under them |
| 6 | the helper is not set up | tell the user the setup below; agents can't do it |
| 1 | could not mount or measure | copy instead, and tell the user what it printed |

## Setup, by the user, once per mounting Mac

Mounting needs the share's password, which agents must never see. A root helper holds it:

```bash
sudo sh ~/.agents/skills/tbshare/scripts/install-tbshare-helper.sh --share <share> --user <account>
sudo visudo -f /etc/sudoers.d/tbshare   # add the NOPASSWD line it prints
```

The installer asks for the password and keeps it readable only by root. The helper takes no arguments: it mounts that one share at its mount point (`/Volumes/<share>` unless `--mountpoint` says otherwise), as the user who ran it, from the one other host on `bridge0`. `ensure` unmounts nothing until the helper is set up. Use a sharing-only account on the server with read access rather than your login, and read the installer before running it with sudo. Keeping the password from other processes relies on System Integrity Protection being on.

## Rules

- Never read, print, or ask for the share's password, and never run the installer yourself.
- Never unmount a share another job is reading.
