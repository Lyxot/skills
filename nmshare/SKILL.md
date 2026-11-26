---
name: nmshare
description: Give an Unsloth Studio frontend its node_modules without a fresh npm install — one install per package-lock.json, shared across worktrees as APFS copy-on-write clones. Use before building or testing the frontend in any checkout or worktree, when a frontend has no node_modules or a stale one, after changing package.json, or when node_modules disk use across worktrees has grown.
---

# nmshare

A full `npm install` per worktree costs about a gigabyte. `nmshare` keeps one install per `package-lock.json` in `<workspace>/.nm-store/` and gives each frontend a real `node_modules` that is an APFS clone of it, so a second worktree on the same lockfile costs tens of megabytes. Each worktree still gets its own `.vite` cache and `.bin`.

```bash
S=~/.agents/skills/nmshare/scripts
$S/nmshare status                     # what each frontend has, and which lockfile it needs
$S/nmshare sync worktrees/<name>      # clone the right install into one worktree
$S/nmshare sync                       # repair every frontend that already has one
$S/nmshare sync --all                 # also install frontends that have none yet
$S/nmshare gc                         # delete store installs no frontend asks for any more
```

The workspace defaults to `~/Github/unsloth` (`NMSHARE_WORKSPACE` overrides it); the frontends are its `unsloth/studio/frontend` and each `worktrees/*/studio/frontend`. A relative path is taken from the workspace.

- A lockfile with no install yet is installed once with `npm ci` and adopted into the store; every later worktree on it clones.
- After changing `package.json`, run `sync --force` on that worktree so the new lockfile gets its own store install instead of building against the old one.

## Rules

- Never run `npm install` or `npm ci` yourself to populate a frontend's `node_modules`; use `nmshare sync`.
- Never symlink one worktree's `node_modules` at another's: the lockfiles drift apart and the build resolves the wrong dependency set. `status` flags any such symlink, and `sync` replaces it.
- `gc` deletes store installs for lockfiles no current frontend uses; run it only after the worktrees that needed them are gone.
