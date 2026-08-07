#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$workspace/project2-coding-agent-rl/.venvs/swe-tools/bin/python"

current_user="$(id -un)"
docker_members="$(getent group docker | cut -d: -f4)"
if [[ ",$docker_members," != *",$current_user,"* ]]; then
  echo "REFUSED: user $current_user is not listed in the docker group." >&2
  exit 2
fi

sg docker -c 'docker info --format "Docker server {{.ServerVersion}}, driver {{.Driver}}"'
sg docker -c "'$python_bin' -c 'import docker; c=docker.from_env(); assert c.ping(); print(\"Docker SDK ping OK\")'"
