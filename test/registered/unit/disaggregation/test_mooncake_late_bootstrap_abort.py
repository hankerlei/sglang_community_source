"""Late decode registration must not revive a timed-out prefill room."""

import threading
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager():
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._terminal_bootstrap_lock = threading.Lock()
    manager._terminal_bootstrap_rooms = OrderedDict()
    manager._terminal_bootstrap_targets = {}
    manager._terminal_bootstrap_notified = {}
    manager._staging_outstanding = defaultdict(int)
    manager.bootstrap_timeout = 60
    manager.waiting_timeout = 120
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.request_status = {7: KVPoll.Failed}
    manager.send_kv_status_message = Mock()
    manager.resolve_kv_replica_factor = Mock()
    manager.update_status = Mock()
    return manager


def _info():
    return SimpleNamespace(
        endpoint="127.0.0.1", dst_port=5000, is_dummy=False, decode_prefix_len=0
    )


def test_late_registration_stays_terminal_and_notifies_decode_once():
    manager = _manager()
    manager._mark_failed_bootstrap_room(7)

    manager._register_bootstrap_transfer(7, "decode", _info(), 1)
    manager._register_bootstrap_transfer(7, "decode", _info(), 1)

    assert manager.request_status[7] == KVPoll.Failed
    assert 7 not in manager.transfer_infos
    manager.update_status.assert_not_called()
    manager.send_kv_status_message.assert_called_once_with(
        targets=[("127.0.0.1", 5000)],
        bootstrap_room=7,
        status=KVPoll.Failed,
        failure_reason="Prefill bootstrap or transfer aborted",
    )


def test_failure_notice_waits_for_inflight_source_write():
    manager = _manager()
    manager._staging_outstanding[7] = 1
    manager._mark_failed_bootstrap_room(7)
    manager._register_bootstrap_transfer(7, "decode", _info(), 1)
    manager.send_kv_status_message.assert_not_called()

    manager._staging_outstanding[7] = 0
    manager._notify_failed_bootstrap_room_if_drained(7)
    manager.send_kv_status_message.assert_called_once()


def test_live_registration_keeps_the_normal_bootstrap_path():
    manager = _manager()
    manager.request_status[7] = KVPoll.Bootstrapping

    manager._register_bootstrap_transfer(7, "decode", _info(), 1)

    assert manager.transfer_infos[7]["decode"].endpoint == "127.0.0.1"
    manager.resolve_kv_replica_factor.assert_called_once()
    manager.update_status.assert_called_once_with(7, KVPoll.WaitingForInput)
    manager.send_kv_status_message.assert_not_called()


def test_terminal_notice_does_not_hold_bootstrap_lock():
    manager = _manager()
    manager._mark_failed_bootstrap_room(7)

    def send(**_kwargs):
        assert manager._terminal_bootstrap_lock.acquire(blocking=False)
        manager._terminal_bootstrap_lock.release()

    manager.send_kv_status_message.side_effect = send
    manager._register_bootstrap_transfer(7, "decode", _info(), 1)
    manager.send_kv_status_message.assert_called_once()


def test_abort_marks_room_before_status_publication():
    manager = _manager()
    manager.request_status[7] = KVPoll.Bootstrapping
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = 7

    def publish_failure(_sender):
        assert 7 in manager._terminal_bootstrap_rooms
        manager.request_status[7] = KVPoll.Failed

    with patch.object(CommonKVSender, "abort", autospec=True, side_effect=publish_failure):
        sender.abort()

    assert manager.request_status[7] == KVPoll.Failed
