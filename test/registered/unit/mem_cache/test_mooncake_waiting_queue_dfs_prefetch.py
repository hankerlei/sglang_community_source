import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ContendedLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if self._lock.locked():
            self.contended.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


def _linker():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_enabled = True
    linker.session_lock = threading.Lock()
    linker.host_prefetch_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lease_base = {}
    linker.host_prefetch_entries = {}
    linker.storage = SimpleNamespace(store=MagicMock())
    return linker


class TestWaitingQueueDfsPrefetch(unittest.TestCase):
    def test_claim_moves_session_atomically(self):
        linker = _linker()
        linker.prepared_load_sessions["prefetch"] = ["a", "b"]
        linker.host_prefetch_entries["r"] = {
            "state": "dfs_prefetched",
            "session_rid": "prefetch",
        }

        self.assertTrue(linker.claim_ready_host_prefetch("r"))
        self.assertEqual(linker.prepared_load_sessions, {"r": ["a", "b"]})
        self.assertEqual(linker.host_prefetch_entries["r"]["state"], "claimed")

    def test_claim_wait_does_not_hold_host_prefetch_lock(self):
        linker = _linker()
        session_lock = _ContendedLock()
        linker.session_lock = session_lock
        linker.prepared_load_sessions["prefetch"] = ["a"]
        linker.host_prefetch_entries.update(
            {
                "r": {"state": "dfs_prefetched", "session_rid": "prefetch"},
                "worker": {"state": "reading"},
            }
        )
        claim_result = {}
        update_done = threading.Event()

        def claim():
            claim_result["value"] = linker.claim_ready_host_prefetch("r")

        def publish():
            with linker.host_prefetch_lock:
                linker.host_prefetch_entries["worker"]["state"] = "dfs_prefetched"
            update_done.set()

        with session_lock:
            claimant = threading.Thread(target=claim)
            claimant.start()
            self.assertTrue(session_lock.contended.wait(2))
            publisher = threading.Thread(target=publish)
            publisher.start()
            self.assertTrue(update_done.wait(2))

        claimant.join(2)
        publisher.join(2)
        self.assertFalse(claimant.is_alive())
        self.assertFalse(publisher.is_alive())
        self.assertTrue(claim_result["value"])

    def test_partial_session_cleanup_ends_successful_keys(self):
        linker = _linker()
        linker.session_refcounts = {"a": 1, "b": 1}
        linker.session_sources = {"a": "dfs", "b": "dfs"}
        linker.session_lease_base = {"a": 1.0, "b": 1.0}

        linker._rollback_session_refs_locked(["a", "b"])

        linker.storage.store.batch_get_session_end.assert_called_once_with(["a", "b"])
        self.assertEqual(linker.session_refcounts, {})

    def test_disable_releases_native_arena(self):
        linker = _linker()
        linker.storage.store.disable_dfs_prefetch.return_value = True

        linker.disable_waiting_queue_prefetch()

        linker.storage.store.disable_dfs_prefetch.assert_called_once_with()
        self.assertFalse(linker.host_prefetch_enabled)

    def test_load_revalidation_rejects_evicted_object(self):
        linker = _linker()
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.storage._get_hybrid_page_component_keys = (
            lambda keys, transfer: ([f"{key}_component" for key in keys], 1)
        )
        linker.storage._tag_keys = lambda keys: [f"tenant/{key}" for key in keys]
        linker.storage._batch_exist = lambda keys: [1, 0]
        linker.storage.pp_size = 1
        linker.storage.failed_get_cache = SimpleNamespace(update_batch=MagicMock())

        valid = linker.revalidate_load(
            [PoolTransfer(name=PoolName.KV, keys=["a", "b"])]
        )

        self.assertFalse(valid)
        linker.storage.failed_get_cache.update_batch.assert_called_once_with(
            [], ["tenant/b_component"]
        )

    def test_lookup_requires_every_pp_shard(self):
        store = MooncakeStore.__new__(MooncakeStore)
        store.mem_pool_host = SimpleNamespace(kv_buffer=None)
        store.pp_rank = 0
        store.pp_size = 2
        store._get_hybrid_page_component_keys = (
            lambda keys, transfer: (
                [f"{key}_cp0_pp0_{transfer.name}" for key in keys],
                1,
            )
        )
        store._tag_keys = lambda keys: keys
        store._batch_exist = lambda keys: [
            int(key != f"b_cp0_pp1_{PoolName.KV}") for key in keys
        ]

        result = store.batch_exists_v2(
            ["a", "b"],
            [PoolTransfer(name=PoolName.KV, keys=["a", "b"])],
            query_all_pp=True,
        )

        self.assertEqual(result.restorable_prefix_pages, [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
