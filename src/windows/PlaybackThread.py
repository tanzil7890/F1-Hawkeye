"""
Playback Thread - Plays back recorded telemetry in background
"""

from PySide6.QtCore import QThread, Signal
import socket
import time
import struct


class PlaybackThread(QThread):
    """Thread to playback recorded telemetry data"""

    playback_started = Signal()
    playback_progress = Signal(int, float)  # packets_sent, timestamp
    playback_finished = Signal(int)  # total_packets
    playback_error = Signal(str)  # error_message

    def __init__(self, file_path, port=20777, host='127.0.0.1', speed=1.0, loop=True):
        super().__init__()
        self.file_path = file_path
        self.port = port
        self.host = host
        self.speed = speed
        self.loop = loop
        self.running = True
        self.paused = False
        self.sock = None

    def run(self):
        """Main playback loop"""
        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            target = (self.host, self.port)

            self.playback_started.emit()

            play_count = 0
            while self.running and (self.loop or play_count == 0):
                play_count += 1
                packets_sent = self.play_once(target)

                if packets_sent > 0:
                    self.playback_finished.emit(packets_sent)

                if not self.loop:
                    break

                # Brief pause between loops
                if self.running and self.loop:
                    time.sleep(1)

        except FileNotFoundError:
            self.playback_error.emit(f"File not found: {self.file_path}")

        except Exception as e:
            self.playback_error.emit(f"Playback error: {str(e)}")

        finally:
            if self.sock:
                self.sock.close()

    def play_once(self, target):
        """Play the recording once"""
        packets_sent = 0
        last_timestamp = 0

        try:
            with open(self.file_path, 'rb') as f:
                while self.running:
                    # Check if paused
                    while self.paused and self.running:
                        time.sleep(0.1)

                    if not self.running:
                        break

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
                    if self.running:
                        self.sock.sendto(packet_data, target)
                        packets_sent += 1
                        last_timestamp = timestamp

                        # Emit progress every 50 packets
                        if packets_sent % 50 == 0:
                            self.playback_progress.emit(packets_sent, timestamp)

        except Exception as e:
            self.playback_error.emit(f"Error reading file: {str(e)}")

        return packets_sent

    def pause(self):
        """Pause playback"""
        self.paused = True

    def resume(self):
        """Resume playback"""
        self.paused = False

    def stop(self):
        """Stop playback"""
        self.running = False
        self.quit()
        self.wait()
