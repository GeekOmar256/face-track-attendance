import argparse
import getpass

import firebase_admin
from firebase_admin import auth, credentials


def main():
    parser = argparse.ArgumentParser(description="Provision a device-scoped Firebase teacher account")
    parser.add_argument("--email", required=True)
    parser.add_argument("--device", default="classroom-pi")
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--reset-password", action="store_true")
    args = parser.parse_args()
    application = firebase_admin.initialize_app(credentials.Certificate(args.credentials))
    try:
        user = auth.get_user_by_email(args.email, app=application)
    except auth.UserNotFoundError:
        password = getpass.getpass("New teacher password: ")
        user = auth.create_user(email=args.email, password=password, display_name="Teacher", app=application)
    else:
        if args.reset_password:
            password = getpass.getpass("Replacement teacher password: ")
            user = auth.update_user(user.uid, password=password, app=application)
        if user.disabled:
            raise ValueError("This account is disabled; review it before granting access")
    claims = {**(user.custom_claims or {}), "face_track_admin": True, "face_track_device": args.device}
    auth.set_custom_user_claims(user.uid, claims, app=application)
    print(f"Teacher account ready: {args.email}; device: {args.device}. Sign in again to refresh access.")


if __name__ == "__main__":
    main()
