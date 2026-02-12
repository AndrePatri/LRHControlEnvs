#!/usr/bin/env python3
# Minimal joystick GUI subscriber over ZeroMQ: listens to a lightweight payload
# carrying linear x/y (left pad) and yaw rate (right slider) and maps them into
# the same attribute names used by the downstream RefsFromJoy logic.

import argparse
import zmq
import json
import time
import threading
import numpy as np
from typing import Optional, Callable


class JoyListenerXbot2ZMQ:

    def __init__(
        self,
        bind: str = "0.0.0.0:6666",  # bind address (GUI publisher connects here)
        topic: str = "joy",
        poll_interval: float = 0.01,
        on_message: Optional[Callable[[dict], None]] = None,
        debug: bool = False
    ):
        self.debug=debug

        self.bind = bind
        self.topic = topic
        self.poll_interval = float(poll_interval)
        self.on_message = on_message

        # thread control
        self.done = False
        self.listener_thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.connect_addr = f"tcp://{self.bind}"
        print("[JoyListenerZMQ]: Binding to", self.connect_addr)
        # bind so the GUI (publisher) can connect
        self.sock.bind(self.connect_addr)

        # subscribe to all (GUI sends single-frame JSON without topic frame)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")

        # state holders (default neutral)
        self.sticks = np.zeros(4, dtype=np.float32)  # left_x,left_y,right_x,right_y
        self.stick_press = np.zeros(2, dtype=bool) # left, right
        self.triggers = np.zeros(2, dtype=np.float32)  # left, right
        self.bumpers = np.zeros(2, dtype=bool)  # left, right
        self.face = np.zeros(4, dtype=bool)  # X, B, A, Y (keeps the size)
        self.back_start_home = np.zeros(3, dtype=bool)  # back, start, home
        self.hat = np.array([0, 0], dtype=int)  # (x, y) values from hat; -1/0/1

        # raw last payload storage
        self.seq = None
        self.ts = None
        self.name = "<unknown>"
        self.axes = []
        self.buttons = []
        self.hats = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.listener_thread and self.listener_thread.is_alive():
            return
        self.listener_thread = threading.Thread(target=self._poll_joy, name="JoyListenerZMQ")
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _poll_joy(self):
        """
        Poll ZMQ socket with a Poller (no busy loop). When a message arrives,
        decode JSON and store the data. Calls on_message(payload) if provided.
        """
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)

        try:
            while not self.done:
                # poller timeout in milliseconds
                events = dict(poller.poll(int(self.poll_interval * 1000)))
                if self.sock in events:
                    try:
                        msg_str = self.sock.recv_string(flags=0)
                    except zmq.ZMQError as e:
                        # interrupted or socket closed
                        if self.done:
                            break
                        print("[JoyListenerZMQ] ZMQ recv error:", e)
                        continue

                    try:
                        payload = json.loads(msg_str)
                    except Exception as e:
                        print("JSON decode error:", e)
                        continue

                    # store internal arrays
                    self._store_payload_data(payload)

                    # call optional callback
                    if self.on_message:
                        try:
                            self.on_message(payload)
                        except Exception as e:
                            print("on_message callback error:", e)
                # else: no event, just loop again (non-busy thanks to poll timeout)
        except KeyboardInterrupt:
            print("Subscriber stopped by user")
        finally:
            try:
                self.sock.close(linger=0)
            except Exception:
                pass
            try:
                self.ctx.term()
            except Exception:
                pass

    def _store_payload_data(self, payload: dict):
        """
        Parse incoming payloads.
        - If payload carries `vref` (twist array), map to sticks/triggers directly.
        - Otherwise fall back to minimal GUI payload (linear x/y + yaw).
        All other controls are forced to neutral.
        """
        self.seq = payload.get("seq")
        self.ts = payload.get("timestamp", time.time())

        self._store_vref_payload(payload)
            
    def _store_vref_payload(self, payload: dict):
        """
        Parse a velocity_command payload with vref = [vx, vy, vz, wx, wy, wz].
        """
        vref = payload.get("vref", []) or []
        # Fill missing entries with zeros
        vx, vy, vz, wx, wy, wz = vref[:6]
        # Clamp to joystick-like normalized range for downstream expectations
        vx = float(np.clip(vx, -1.0, 1.0))
        vy = float(np.clip(-vy, -1.0, 1.0))
        wz = float(np.clip(-wz, -1.0, 1.0))

        # reset to neutral
        self.sticks[:] = 0.0
        self.stick_press[:] = False
        self.triggers[:] = 0.0
        self.bumpers[:] = False
        self.face[:] = False
        self.back_start_home[:] = False
        self.hat[:] = 0

        # linear -> right stick slots (same as minimal GUI mapping)
        self.sticks[3] = vx
        self.sticks[2] = vy

        # yaw rate -> triggers (positive yaw -> left trigger)
        self.triggers[0] = max(-wz, 0.0)
        self.triggers[1] = max(wz, 0.0)

        # raw copies
        self.axes = [vx, vy, vz, wx, wy, wz]
        self.buttons = []
        self.hats = []

        self.name = payload.get("task_name", payload.get("type", "vref_cmd"))
        self.info_str = (
            f"[{time.strftime('%H:%M:%S', time.localtime(self.ts))}] "
            f"vref lin=({vx:.3f},{vy:.3f}) yaw={wz:.3f} task='{self.name}'"
        )

    def stop(self):
        """
        Request the listener to stop and join the thread briefly.
        """
        if not self.done:
            self.done = True
            # closing socket will interrupt poll/recv
            try:
                self.sock.close(linger=0)
            except Exception:
                pass
            try:
                self.ctx.term()
            except Exception:
                pass

            # join thread
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1.0)

    def pretty_print_payload(self, payload: dict = None):
        """
        Print the latest state from the listener's numpy arrays (thread-safe-ish).
        If payload is provided and arrays are missing, falls back to printing payload.
        """
        try:
            # Info/header (use stored info_str if available)
            header = getattr(self, "info_str", None)
            if header:
                print(header)
            else:
                if payload:
                    ts = payload.get("timestamp")
                    seq = payload.get("seq")
                    name = payload.get("state", {}).get("name", "<unknown>")
                    print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq} device='{name}'")

            # Copy arrays so we don't print while they are being updated
            sticks = self.sticks.copy()       # left_x, left_y, right_x, right_y
            sticks_press = self.stick_press.copy() # left, right
            triggers = self.triggers.copy()   # left, right
            bumpers = self.bumpers.copy()     # left, right
            face = self.face.copy()           # X, B, A, Y (kept as in class)
            back = self.back_start_home.copy()# back, start, home
            hat = self.hat.copy()             # (x, y)
            raw_axes = list(self.axes) if hasattr(self, "axes") else []
            raw_buttons = list(self.buttons) if hasattr(self, "buttons") else []
            raw_hats = list(self.hats) if hasattr(self, "hats") else []

            # Format and print
            # Round floats for readability
            def r(x): 
                try:
                    return round(float(x), 3)
                except Exception:
                    return x

            print("  sticks (left_x,left_y,right_x,right_y):", [r(v) for v in sticks])
            print("  sticks press (LB,RB):", [bool(x) for x in sticks_press])
            print("  triggers (L,R):", [r(v) for v in triggers])
            print("  bumpers (LB,RB):", [bool(x) for x in bumpers])
            print("  face (X,B,A,Y):", [bool(x) for x in face])
            print("  back/start/home:", [bool(x) for x in back])
            print("  hat (x,y):", (int(hat[0]), int(hat[1])))

            # Also show raw lists for debugging (truncated)
            if self.debug:
                print("  raw axes (first 8):", [r(a) for a in raw_axes[:8]])
                print("  raw buttons (first 16):", raw_buttons[:16])
                print("  raw hats:", raw_hats)
                print("-" * 50)

        except Exception as e:
            # Don't crash the whole program if printing fails
            print("pretty_print_payload error:", e)
            # fallback: if payload present, print minimal info from it
            if payload:
                seq = payload.get("seq")
                ts = payload.get("timestamp")
                state = payload.get("state", {})
                name = state.get("name", "<unknown>")
                axes = state.get("axes", [])
                buttons = state.get("buttons", [])
                hats = state.get("hats", [])
                print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq} device='{name}' axes={len(axes)} buttons={len(buttons)} hats={len(hats)}")

def main():
    parser = argparse.ArgumentParser(description="ZeroMQ joystick subscriber (threaded, non-busy).")
    parser.add_argument("--bind", default="0.0.0.0:6666", help="Bind address for the subscriber (host:port). Default 0.0.0.0:6666")
    parser.add_argument("--topic", default="joy", help="Topic to subscribe to (default 'joy')")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Poll interval seconds (default 0.01)")
    args = parser.parse_args()

    # A simple on_message callback that prints the payload summary
    def on_message(payload):
        # Print the payload via the object's pretty printer — but we need access to listener.
        # We'll capture listener from outer scope by setting it after creation. Use fallback print.
        print("AAAAAAAAAAA")
        print(payload)
        if hasattr(listener, "pretty_print_payload"):
            listener.pretty_print_payload(payload)
        else:
            # fallback minimal print
            seq = payload.get("seq")
            ts = payload.get("timestamp")
            print(f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] seq={seq}")

    # create listener and run until Ctrl-C
    listener = JoyListenerXbot2ZMQ(bind=args.bind, topic=args.topic, poll_interval=args.poll_interval, on_message=on_message)

    # start listening using context manager (optional)
    listener.start()
    print("Listener started. Press Ctrl-C to exit.")
    try:
        while not listener.done:
            # main loop can do other work; here we sleep to be idle but responsive
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
