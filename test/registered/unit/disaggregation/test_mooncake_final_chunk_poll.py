import threading
import unittest
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import (  # noqa: E402
    MooncakeKVManager,
    MooncakeKVSender,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeDrainedAbortStatus(CustomTestCase):
    @staticmethod
    def make_manager(status=KVPoll.WaitingForInput, expected=3):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: status}
        manager.prefill_response_tracker = defaultdict(set)
        manager.required_prefill_response_num_table = {7: expected}
        manager.prefill_drained_abort_rooms = set()
        manager._drained_abort_rooms = OrderedDict()
        manager._drained_abort_lock = threading.Lock()
        manager.transfer_infos = {}
        manager.req_to_decode_prefix_len = {}
        manager.bootstrap_timeout = manager.waiting_timeout = 300
        manager.deferred_bootstrap = None
        manager._staging_ctx = None
        manager._maybe_ack_drained_abort = Mock()
        manager.source_transfers_drained = Mock(return_value=True)
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()
        manager.enable_staging = False
        manager.record_failure = lambda room, reason: (
            manager.failure_records.__setitem__(room, reason)
        )
        return manager

    @staticmethod
    def status_msg(room, status, rank, drained_abort=False):
        msg = [str(room).encode(), str(status).encode(), str(rank).encode()]
        if drained_abort:
            msg.append(b"DRAINED_ABORT")
        return msg

    def test_drained_abort_waits_for_every_unique_prefill_rank(self):
        manager = self.make_manager()
        duplicate_abort = self.status_msg(7, KVPoll.Failed, 0, drained_abort=True)
        manager._handle_prefill_status_message(duplicate_abort)
        manager._handle_prefill_status_message(duplicate_abort)
        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)
        self.assertEqual(len(manager.prefill_response_tracker[7]), 1)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 1))
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 2))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)

        for order in (
            ("abort", "success", "success"),
            ("success", "success", "abort"),
        ):
            with self.subTest(order=order):
                manager = self.make_manager()
                rank_for_kind = iter((0, 1, 2))
                for kind in order:
                    rank = next(rank_for_kind)
                    is_abort = kind == "abort"
                    status = KVPoll.Failed if is_abort else KVPoll.Success
                    manager._handle_prefill_status_message(
                        self.status_msg(7, status, rank, is_abort)
                    )
                    if len(manager.prefill_response_tracker[7]) < 3:
                        self.assertEqual(
                            manager.request_status[7], KVPoll.WaitingForInput
                        )
                self.assertEqual(manager.request_status[7], KVPoll.Failed)
                self.assertIn(7, manager.prefill_drained_abort_rooms)
                self.assertIn(7, manager.failure_records)
                response_count = len(manager.prefill_response_tracker[7])
                manager._handle_prefill_status_message(
                    self.status_msg(7, KVPoll.Success, 99)
                )
                self.assertEqual(manager.request_status[7], KVPoll.Failed)
                self.assertEqual(
                    len(manager.prefill_response_tracker[7]), response_count
                )

                manager.request_status.clear()
                manager._handle_prefill_status_message(
                    self.status_msg(7, KVPoll.Failed, 99, drained_abort=True)
                )
                self.assertNotIn(7, manager.request_status)

    def test_success_and_legacy_failure_keep_existing_terminal_behavior(self):
        manager = self.make_manager()
        for rank in range(3):
            manager._handle_prefill_status_message(
                self.status_msg(7, KVPoll.Success, rank)
            )
        self.assertEqual(manager.request_status[7], KVPoll.Success)
        # A duplicate terminal response neither changes state nor re-runs the
        # final scatter path.
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 2))
        self.assertEqual(len(manager.prefill_response_tracker[7]), 3)
        manager._handle_prefill_status_message(
            self.status_msg(7, KVPoll.Failed, 0, drained_abort=True)
        )
        self.assertEqual(manager.request_status[7], KVPoll.Success)

        manager = self.make_manager(expected=4)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 0))
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Failed, 1))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertNotIn(7, manager.prefill_drained_abort_rooms)

        manager = self.make_manager(expected=1)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 0))
        self.assertEqual(manager.request_status[7], KVPoll.Success)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Failed, 1))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)

    def test_late_drained_abort_target_and_saved_session_targets_are_covered(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.transfer_infos = {
            7: {
                "session-1": SimpleNamespace(
                    endpoint="decode-a", dst_port=100, is_dummy=False
                ),
                "session-2": SimpleNamespace(
                    endpoint="decode-a", dst_port=100, is_dummy=False
                ),
                "session-3": SimpleNamespace(
                    endpoint="decode-b", dst_port=200, is_dummy=False
                ),
                "dummy": SimpleNamespace(endpoint="", dst_port=0, is_dummy=True),
            }
        }
        manager._source_drain = SimpleNamespace(lock=threading.Lock())
        manager.record_failure = Mock()
        manager.update_status = Mock()
        manager.sync_status_to_decode_endpoint = Mock()
        manager._prefill_unique_rank = lambda: 4
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.conclude_state = None
        sender.trace_ctx = Mock()
        sender._drained_abort_targets = set()
        sender._drained_abort_notified_targets = set()

        sender.abort()
        self.assertEqual(
            sender._drained_abort_targets,
            {("decode-a", 100), ("decode-b", 200)},
        )
        manager.sync_status_to_decode_endpoint.side_effect = RuntimeError("send failed")
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            sender.notify_drained_abort()
        self.assertNotIn(7, manager.transfer_infos)
        self.assertIn(7, manager._drained_abort_rooms)
        self.assertEqual(sender._drained_abort_notified_targets, set())
        manager.sync_status_to_decode_endpoint.reset_mock(side_effect=True)
        sender.notify_drained_abort()
        sender.notify_drained_abort()
        self.assertEqual(
            {
                (call.args[0], call.args[1])
                for call in manager.sync_status_to_decode_endpoint.call_args_list
            },
            {("decode-a", 100), ("decode-b", 200)},
        )
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 2)
        for call in manager.sync_status_to_decode_endpoint.call_args_list:
            self.assertEqual(call.args[2:5], (7, KVPoll.Failed, 4))
            self.assertTrue(call.kwargs["is_drained_abort"])

        # After live state was cleared, late bootstrap metadata is handled
        # by the terminal marker rather than resurrecting transfer state.
        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.transfer_infos)
        target = SimpleNamespace(endpoint="decode-c", dst_port=300, is_dummy=False)
        self.assertTrue(manager._notify_drained_abort_target_if_needed(7, target))
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 3)
        self.assertEqual(
            manager.sync_status_to_decode_endpoint.call_args.args[:2],
            ("decode-c", 300),
        )

    def test_blocked_bootstrap_send_does_not_block_terminal_cleanup(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                manager = self.make_manager(status=KVPoll.Failed)
                manager.transfer_infos = {}
                manager.resolve_kv_replica_factor = Mock()
                manager._prefill_unique_rank = lambda: 0
                if terminal:
                    manager.finish_drained_abort(7)
                entered, release, cleaned = (threading.Event() for _ in range(3))
                errors = []

                def blocked_send(*args, **kwargs):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release send")

                def register():
                    try:
                        target = SimpleNamespace(
                            endpoint="decode-blocked",
                            dst_port=123,
                            is_dummy=False,
                            decode_prefix_len=0,
                            spec_metadata=None,
                        )
                        manager._register_bootstrap_transfer(7, "session", target, 1)
                    except BaseException as error:
                        errors.append(error)

                def cleanup():
                    try:
                        manager.finish_drained_abort(8)
                        cleaned.set()
                    except BaseException as error:
                        errors.append(error)

                manager.sync_status_to_decode_endpoint = Mock(side_effect=blocked_send)
                notify = threading.Thread(target=register)
                cleanup_thread = threading.Thread(target=cleanup)
                notify.start()
                started = entered.wait(1)
                cleanup_thread.start()
                responsive = cleaned.wait(0.5)
                release.set()
                notify.join(5)
                cleanup_thread.join(5)
                self.assertTrue(started)
                self.assertTrue(
                    responsive, "bootstrap send held the scheduler cleanup lock"
                )
                self.assertFalse(notify.is_alive())
                self.assertFalse(cleanup_thread.is_alive())
                self.assertEqual(errors, [])
                manager.sync_status_to_decode_endpoint.assert_called_once()
                if terminal:
                    self.assertNotIn(7, manager.transfer_infos)

    def test_bootstrap_target_is_not_notified_until_source_drain(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.update_status(7, KVPoll.WaitingForInput)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        manager.source_transfers_drained = Mock(return_value=False)
        manager.sync_status_to_decode_endpoint = Mock()
        manager._prefill_unique_rank = lambda: 2
        target = SimpleNamespace(endpoint="decode-late", dst_port=456, is_dummy=False)
        manager._notify_drained_abort_target_if_needed(7, target)
        manager.sync_status_to_decode_endpoint.assert_not_called()

        manager.source_transfers_drained.return_value = True
        manager._notify_drained_abort_target_if_needed(7, target)
        manager.sync_status_to_decode_endpoint.assert_called_once_with(
            "decode-late",
            456,
            7,
            KVPoll.Failed,
            2,
            is_drained_abort=True,
        )

    def test_late_bootstrap_notification_error_does_not_escape_thread_hook(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.source_transfers_drained = Mock(return_value=True)
        manager.sync_status_to_decode_endpoint = Mock(
            side_effect=RuntimeError("temporary send failure")
        )
        manager._prefill_unique_rank = lambda: 2
        target = SimpleNamespace(endpoint="decode-late", dst_port=456, is_dummy=False)

        with self.assertLogs(
            "sglang.srt.disaggregation.mooncake.conn", level="ERROR"
        ) as logs:
            manager._notify_drained_abort_target_if_needed(7, target)

        self.assertIn("temporary send failure", logs.output[0])
        manager.sync_status_to_decode_endpoint.assert_called_once()

    def test_wire_marker_is_only_added_for_drained_abort(self):
        manager = object.__new__(MooncakeKVManager)
        manager._send_multipart_locked = Mock()
        manager.sync_status_to_decode_endpoint("127.0.0.1", 12345, 7, KVPoll.Failed, 2)
        manager.sync_status_to_decode_endpoint(
            "127.0.0.1", 12345, 7, KVPoll.Failed, 2, is_drained_abort=True
        )
        self.assertEqual(
            manager._send_multipart_locked.call_args_list[0].args[1],
            [b"7", str(KVPoll.Failed).encode(), b"2"],
        )
        self.assertEqual(
            manager._send_multipart_locked.call_args_list[1].args[1],
            [b"7", str(KVPoll.Failed).encode(), b"2", b"DRAINED_ABORT"],
        )


if __name__ == "__main__":
    unittest.main()
