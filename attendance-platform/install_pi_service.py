"""Install the attendance platform as the current Pi user's systemd service."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


def unit_quote(value, specifiers=True):
    value = str(value)
    if any(char in value for char in "\r\n\0"):
        raise ValueError("Invalid newline in service value")
    value = value.replace('\\', '\\\\').replace('"', '\\"')
    return '"' + (value.replace('%', '%%') if specifiers else value) + '"'


def service_unit(root, username, environment_file, port, groups=()):
    return "\n".join([
        "[Unit]", "Description=Face Track attendance platform",
        "Wants=network-online.target", "After=network-online.target", "StartLimitIntervalSec=0", "",
        "[Service]", "Type=simple", f"User={username}",
        *( ["SupplementaryGroups=" + " ".join(groups)] if groups else [] ),
        "WorkingDirectory=" + unit_quote(root), "EnvironmentFile=" + unit_quote(environment_file),
        "Environment=PYTHONUNBUFFERED=1",
        "ExecStart=" + unit_quote(root / ".venv/bin/python") + " -u " + unit_quote(root / "app.py")
        + f" --host 0.0.0.0 --port {port}",
        "Restart=always", "RestartSec=5", "TimeoutStopSec=20", "KillMode=control-group",
        "UMask=0077", "NoNewPrivileges=true", "", "[Install]", "WantedBy=multi-user.target", ""
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--device", default="classroom-pi")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This installer is for the Raspberry Pi's Linux system")
    import grp
    import pwd
    if os.geteuid() == 0:
        parser.error("Run as the Pi login user, without sudo. Only service installation will request sudo.")
    if not 1 <= args.port <= 65535 or args.device != "classroom-pi":
        parser.error("Use a valid port and classroom-pi (the deployed teacher site's device ID)")
    root = Path(__file__).resolve().parent
    credentials = args.credentials.expanduser().resolve(strict=True)
    if credentials.is_relative_to(root.parent):
        parser.error("Keep the private credentials file outside the Git repository")
    details = json.loads(credentials.read_text(encoding="utf-8-sig"))
    if details.get("type") != "service_account" or details.get("project_id") != "bio-attendance-9zeket" or not details.get("private_key"):
        parser.error("Expected a private service account for bio-attendance-9zeket")
    if not (root / ".venv/bin/python").is_file():
        parser.error("Create .venv with system Python and --system-site-packages first")
    identity = pwd.getpwuid(os.getuid())
    config = Path(identity.pw_dir) / ".config/face-track"
    environment_file = config / "service.env"
    available_groups = {group.gr_name for group in grp.getgrall()}
    groups = [name for name in ("video", "render") if name in available_groups]
    unit = service_unit(root, identity.pw_name, environment_file, args.port, groups)
    if args.dry_run:
        print(unit)
        return
    credentials.chmod(0o600)
    config.mkdir(mode=0o700, parents=True, exist_ok=True)
    config.chmod(0o700)
    # Preserve the generated local password when reinstalling the service.
    admin_password = secrets.token_urlsafe(24)
    if environment_file.exists():
        for line in environment_file.read_text().splitlines():
            if line.startswith("FACETRACK_ADMIN_PASSWORD="):
                admin_password = line.split("=", 1)[1].strip('"')
    environment_file.touch(mode=0o600, exist_ok=True)
    environment_file.chmod(0o600)
    environment_file.write_text("GOOGLE_APPLICATION_CREDENTIALS=" + unit_quote(credentials, False)
                                + "\nFACETRACK_DEVICE_ID=classroom-pi\nFACETRACK_ADMIN_PASSWORD="
                                + unit_quote(admin_password, False) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as folder:
        unit_file = Path(folder) / "face-track.service"
        unit_file.write_text(unit, encoding="utf-8")
        subprocess.run(["sudo", "install", "-m", "644", str(unit_file),
                        "/etc/systemd/system/face-track.service"], check=True)
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", "face-track.service"], check=True)
    subprocess.run(["sudo", "systemctl", "restart", "face-track.service"], check=True)
    print("Service installed, enabled at boot and started. Check: systemctl status face-track --no-pager")
    print("Logs: journalctl -u face-track -n 80 --no-pager")
    print("Teacher page: https://face-track-bio-attendance.web.app/")


if __name__ == "__main__":
    main()
