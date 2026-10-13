import argparse
import firebase_admin
from firebase_admin import auth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grant an existing Firebase Auth account access to one device")
    parser.add_argument("uid")
    parser.add_argument("device")
    args = parser.parse_args()
    firebase_admin.initialize_app()
    user = auth.get_user(args.uid)
    auth.set_custom_user_claims(args.uid, {**(user.custom_claims or {}), "face_track_admin": True, "face_track_device": args.device})
    print("Admin granted. Sign out and sign in again on the mobile dashboard.")
