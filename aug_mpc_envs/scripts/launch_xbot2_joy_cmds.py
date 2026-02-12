#!/usr/bin/env python3
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run keyboard or joystick refs writer.")
    parser.add_argument('--ns', type=str, required=True, help='Namespace to be used for shared memory')
    parser.add_argument('--env_idx', type=int, default=0)
    parser.add_argument('--bind', type=str, default='0.0.0.0:6666', help='JoyListenerZMQ bind address (host:port)')
    parser.add_argument('--topic', type=str, default='joy', help='ZeroMQ topic for joystick messages')
    parser.add_argument('--poll-interval', type=float, default=0.01, help='Joy listener poll interval (s)')
    parser.add_argument('--hold-time', type=float, default=0.15, help='Hold time for toggles (seconds) when using joystick')

    args = parser.parse_args()

    from mpc_hive.utilities.shared_data.rhc_data import RhcRefs
    from EigenIPC.PyEigenIPC import VLevel

    # create shared refs (same for both modes)
    shared_refs = RhcRefs(namespace=args.ns,
                          is_server=False,
                          safe=False,
                          verbose=True,
                          vlevel=VLevel.V2)

    # import both classes (assumes they are available at these paths)
    from aug_mpc_envs.utils.xbot2.joy_cmds import RefsFromJoy 

    # joystick-driven
    joy_cmds = RefsFromJoy(namespace=args.ns,
                            shared_refs=shared_refs,
                            verbose=True,
                            agent_refs_world=False,
                            env_idx=args.env_idx,
                            hold_time=args.hold_time)

    # No remote-exit callback; pass callback_arg=None
    joy_cmds.run(bind=args.bind, topic=args.topic, poll_interval=args.poll_interval,
                    callback=None, callback_arg=None)
