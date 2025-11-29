#!/usr/bin/env python3
"""
Simple Telemetry Recorder - Records UDP packets from F1 game to a binary file

Usage:
    python telemetry_recorder.py -o my_race.bin

Then play the game with UDP telemetry enabled (port 20777)
The packets will be recorded to the file.

To replay later, use f1-cli:
    npm install -g @racehub/f1-cli
    f1-cli play my_race.bin --port 20777
"""

import socket
import time
import argparse
import struct
from datetime import datetime

class TelemetryRecorder:
    def __init__(self, port=20777, output_file="recording.bin"):
        self.port = port
        self.output_file = output_file
        self.sock = None
        self.file = None
        self.packet_count = 0
        self.start_time = None

    def start(self):
        """Start recording telemetry"""
        print("=" * 60)
        print("📹 F1 TELEMETRY RECORDER")
        print("=" * 60)
        print(f"📡 Listening on port: {self.port}")
        print(f"💾 Output file: {self.output_file}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print("\n✅ Recorder is running!")
        print("🎮 Start the F1 game with UDP telemetry enabled")
        print("🛑 Press Ctrl+C to stop recording\n")

        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('', self.port))
            self.sock.settimeout(1.0)  # 1 second timeout for Ctrl+C responsiveness

            # Open output file
            self.file = open(self.output_file, 'wb')
            self.start_time = time.time()

            # Record loop
            while True:
                try:
                    data, addr = self.sock.recvfrom(2048)

                    # Write timestamp + packet length + packet data
                    timestamp = time.time() - self.start_time
                    packet_length = len(data)

                    # Format: [timestamp (8 bytes)] [length (4 bytes)] [data]
                    self.file.write(struct.pack('<d', timestamp))  # double
                    self.file.write(struct.pack('<I', packet_length))  # unsigned int
                    self.file.write(data)

                    self.packet_count += 1

                    # Progress indicator
                    if self.packet_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        print(f"📦 Recorded {self.packet_count} packets "
                              f"({elapsed:.1f}s elapsed, "
                              f"{self.packet_count/elapsed:.1f} packets/sec)")

                except socket.timeout:
                    # Just continue - this allows Ctrl+C to work
                    pass

        except KeyboardInterrupt:
            print("\n\n🛑 Recording stopped by user")

        finally:
            self.stop()

    def stop(self):
        """Stop recording and cleanup"""
        if self.sock:
            self.sock.close()

        if self.file:
            self.file.close()

        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 60)
        print("📊 RECORDING SUMMARY")
        print("=" * 60)
        print(f"📦 Total packets: {self.packet_count}")
        print(f"⏱️  Duration: {elapsed:.1f} seconds")
        print(f"📈 Average rate: {self.packet_count/max(elapsed, 1):.1f} packets/sec")
        print(f"💾 Saved to: {self.output_file}")
        print("=" * 60)

        if self.packet_count > 0:
            print("\n✅ Recording saved successfully!")
            print("\n🎬 To replay this recording:")
            print(f"   npm install -g @racehub/f1-cli")
            print(f"   f1-cli play {self.output_file} --port 20777")
            print("\n   (Or use a custom player script)")
        else:
            print("\n⚠️  No packets received!")
            print("   Make sure the F1 game is running with:")
            print("   • UDP Telemetry: ON")
            print("   • Port: 20777")
            print("   • UDP Broadcast: ON")

def main():
    parser = argparse.ArgumentParser(
        description='Record F1 game telemetry to a binary file'
    )
    parser.add_argument(
        '-o', '--output',
        default=f'f1_recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin',
        help='Output filename (default: f1_recording_TIMESTAMP.bin)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=20777,
        help='UDP port to listen on (default: 20777)'
    )

    args = parser.parse_args()

    recorder = TelemetryRecorder(port=args.port, output_file=args.output)
    recorder.start()

if __name__ == "__main__":
    main()
