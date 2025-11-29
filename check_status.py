#!/usr/bin/env python3
"""
Status Checker - Verifies the F1 Telemetry application is ready to receive data
"""

import socket
import sys

def check_port_listening(port=20777):
    """Check if the application is listening on the telemetry port"""
    print(f"🔍 Checking if port {port} is listening...")

    try:
        # Try to bind to the port - if we can, nothing is listening
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_sock.bind(('', port))
        test_sock.close()

        print(f"❌ Port {port} is NOT in use")
        print(f"   → The Telemetry application might not be running")
        return False

    except OSError:
        print(f"✅ Port {port} IS in use (application is running)")
        return True

def test_send_packet(port=20777):
    """Send a test UDP packet"""
    print(f"\n📤 Sending test UDP packet to localhost:{port}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_message = b"TEST"
        sock.sendto(test_message, ('127.0.0.1', port))
        sock.close()

        print(f"✅ Test packet sent successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to send packet: {e}")
        return False

def main():
    print("=" * 60)
    print("F1 TELEMETRY APPLICATION - STATUS CHECK")
    print("=" * 60)

    # Check if port is listening
    is_listening = check_port_listening(20777)

    if is_listening:
        print("\n✅ APPLICATION STATUS: READY")
        print("\n📋 Next steps:")
        print("   1. Launch F1 25 (or F1 22/23/24)")
        print("   2. Go to: Settings → Telemetry Settings")
        print("   3. Configure:")
        print("      • UDP Telemetry: ON")
        print("      • Port: 20777")
        print("      • UDP Format: 2025 (or your game version)")
        print("      • UDP Broadcast Mode: ON")
        print("   4. Start any session (Practice/Qualifying/Race)")
        print("   5. Data will appear in the application!")

        # Try sending a test packet
        test_send_packet(20777)

    else:
        print("\n❌ APPLICATION STATUS: NOT RUNNING")
        print("\n📋 To start the application:")
        print("   python Telemetry.py")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
