#!/usr/bin/env python3
"""
Simple Telemetry Player - Replays recorded UDP packets

Usage:
    python telemetry_player.py my_race.bin

Start Telemetry.py first, then run this to replay the recording!
"""

import socket
import time
import argparse
import struct
import sys

class TelemetryPlayer:
    def __init__(self, input_file, port=20777, host='127.0.0.1', speed=1.0, loop=False):
        self.input_file = input_file
        self.port = port
        self.host = host
        self.speed = speed
        self.loop = loop
        self.sock = None

    def play(self):
        """Play back the recording"""
        print("=" * 60)
        print("▶️  F1 TELEMETRY PLAYER")
        print("=" * 60)
        print(f"📁 Input file: {self.input_file}")
        print(f"📡 Sending to: {self.host}:{self.port}")
        print(f"⚡ Playback speed: {self.speed}x")
        print(f"🔁 Loop: {'Yes' if self.loop else 'No'}")
        print("=" * 60)

        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            target = (self.host, self.port)

            play_count = 0
            while True:
                play_count += 1
                if self.loop and play_count > 1:
                    print(f"\n🔁 Loop iteration {play_count}")

                packets_sent = self.play_once(target)

                if not self.loop:
                    break

                print(f"\n✅ Completed playback of {packets_sent} packets")
                time.sleep(1)  # Brief pause between loops

        except KeyboardInterrupt:
            print("\n\n🛑 Playback stopped by user")

        except FileNotFoundError:
            print(f"\n❌ Error: File not found: {self.input_file}")
            sys.exit(1)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

        finally:
            if self.sock:
                self.sock.close()

    def play_once(self, target):
        """Play the recording once"""
        packets_sent = 0
        last_timestamp = 0

        with open(self.input_file, 'rb') as f:
            print(f"\n▶️  Starting playback...\n")
            start_time = time.time()

            while True:
                # Read timestamp (8 bytes double)
                timestamp_bytes = f.read(8)
                if len(timestamp_bytes) < 8:
                    break  # End of file

                timestamp = struct.unpack('<d', timestamp_bytes)[0]

                # Read packet length (4 bytes unsigned int)
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break

                packet_length = struct.unpack('<I', length_bytes)[0]

                # Read packet data
                packet_data = f.read(packet_length)
                if len(packet_data) < packet_length:
                    break

                # Wait for the appropriate time (adjusted by speed multiplier)
                if packets_sent > 0:
                    delay = (timestamp - last_timestamp) / self.speed
                    if delay > 0:
                        time.sleep(delay)

                # Send the packet
                self.sock.sendto(packet_data, target)
                packets_sent += 1
                last_timestamp = timestamp

                # Progress indicator
                if packets_sent % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"📤 Sent {packets_sent} packets "
                          f"(@ {timestamp:.1f}s in recording, "
                          f"{elapsed:.1f}s real time)")

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 PLAYBACK SUMMARY")
        print("=" * 60)
        print(f"📦 Total packets sent: {packets_sent}")
        print(f"⏱️  Recording duration: {last_timestamp:.1f}s")
        print(f"⏱️  Playback duration: {elapsed:.1f}s")
        print(f"⚡ Actual speed: {last_timestamp/max(elapsed,1):.2f}x")
        print("=" * 60)

        return packets_sent

def main():
    parser = argparse.ArgumentParser(
        description='Play back recorded F1 telemetry'
    )
    parser.add_argument(
        'input_file',
        help='Input recording file (.bin)'
    )
    parser.add_argument(
        '-H', '--host',
        default='127.0.0.1',
        help='Target host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=20777,
        help='Target UDP port (default: 20777)'
    )
    parser.add_argument(
        '-s', '--speed',
        type=float,
        default=1.0,
        help='Playback speed multiplier (default: 1.0)'
    )
    parser.add_argument(
        '-l', '--loop',
        action='store_true',
        help='Loop playback continuously'
    )

    args = parser.parse_args()

    print("\n🎬 Make sure Telemetry.py is running before starting playback!\n")
    time.sleep(2)

    player = TelemetryPlayer(
        input_file=args.input_file,
        port=args.port,
        host=args.host,
        speed=args.speed,
        loop=args.loop
    )
    player.play()

    print("\n✅ Playback complete!")

if __name__ == "__main__":
    main()
