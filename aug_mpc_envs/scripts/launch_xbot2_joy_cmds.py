#!/usr/bin/env python3
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run xbot2 joystick refs writer.")
    parser.add_argument('--ns', type=str, required=True, help='Namespace to be used for shared memory')
    parser.add_argument('--env_idx', type=int, default=0)
    parser.add_argument('--bind', type=str, default='0.0.0.0:6666', help='JoyListenerXbot2ZMQ bind address (host:port)')
    parser.add_argument('--topic', type=str, default='joy', help='ZeroMQ topic for joystick messages')
    parser.add_argument('--poll-interval', type=float, default=0.01, help='Joy listener poll interval (s)')
    parser.add_argument('--mode', choices=['linvel', 'pos'], default='linvel',
                        help='Static high-level mode for xbot input.')
    parser.add_argument('--agent_refs', action='store_true',
                        help='If set, write AgentRefs via AgentRefsFromJoy instead of MPC refs.')

    args = parser.parse_args()

    from aug_mpc_envs.utils.xbot2.listener_xbot_zmq import JoyListenerXbot2ZMQ

    if args.agent_refs:
        from aug_mpc.utils.joy_cmds import AgentRefsFromJoy

        joy_cmds = AgentRefsFromJoy(namespace=args.ns,
                                    verbose=True,
                                    agent_refs_world=False,
                                    env_idx=args.env_idx,
                                    hold_time=0.0,
                                    listener_factory=JoyListenerXbot2ZMQ,
                                    listener_endpoint_mode="bind",
                                    fixed_motion_mode=args.mode,
                                    force_omega=True)
    else:
        from mpc_hive.utilities.shared_data.rhc_data import RhcRefs
        from EigenIPC.PyEigenIPC import VLevel
        from mpc_hive.utilities.joy_cmds import RefsFromJoy

        # create shared refs for MPC-ref writes
        shared_refs = RhcRefs(namespace=args.ns,
                              is_server=False,
                              safe=False,
                              verbose=True,
                              vlevel=VLevel.V2)

        joy_cmds = RefsFromJoy(namespace=args.ns,
                                shared_refs=shared_refs,
                                verbose=True,
                                agent_refs_world=False,
                                env_idx=args.env_idx,
                                hold_time=0.0,
                                listener_factory=JoyListenerXbot2ZMQ,
                                listener_endpoint_mode="bind",
                                fixed_motion_mode=args.mode,
                                force_omega=True)

    # No remote-exit callback; pass callback_arg=None
    joy_cmds.run(connect=args.bind, topic=args.topic, poll_interval=args.poll_interval,
                    callback=None, callback_arg=None)
