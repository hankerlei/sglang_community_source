"""CPU-only checks for Mooncake direct-linker I/O completion semantics."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    _get_mooncake_client_http_port,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeDirectLinkerIO(unittest.TestCase):
    def test_page_wise_load_completes_only_after_all_ranges(self):
        linker = object.__new__(MooncakeDirectLinker)
        linker.num_layers = 2
        linker.page_wise_load_batch_size = 2
        linker.layer_done_counter = Mock()
        pool = Mock()
        pool.get_prepared_layer_range_meta.side_effect = [
            ([[10], [20]], [[4], [4]], [[0], [0]]),
            ([[30], [40]], [[8], [8]], [[4], [4]]),
        ]
        linker.pools = {PoolName.KV: pool}
        store = Mock()
        store.batch_get_into_multi_buffer_ranges.return_value = [12, 12]
        linker.storage = Mock(store=store)
        keys = ["a", "b"]
        started = [keys]

        linker._load_page_wise(7, {PoolName.KV: (keys, [1, 2])}, started)

        store.batch_get_into_multi_buffer_ranges.assert_called_once_with(
            keys, [[10, 30], [20, 40]], [[4, 8], [4, 8]], [[0, 4], [0, 4]]
        )
        store.batch_get_session_end.assert_called_once_with(keys)
        self.assertEqual(started, [])
        self.assertEqual(linker.layer_done_counter.complete.call_count, 2)

    def test_page_wise_scalar_error_never_completes_a_layer(self):
        linker = object.__new__(MooncakeDirectLinker)
        linker.num_layers = 1
        linker.page_wise_load_batch_size = 1
        linker.layer_done_counter = Mock()
        pool = Mock()
        pool.get_prepared_layer_range_meta.return_value = ([[10]], [[4]], [[0]])
        linker.pools = {PoolName.KV: pool}
        store = Mock()
        store.batch_get_into_multi_buffer_ranges.return_value = 707
        linker.storage = Mock(store=store)
        keys = ["a"]
        started = [keys]

        with self.assertRaisesRegex(RuntimeError, "transferred=707"):
            linker._load_page_wise(7, {PoolName.KV: (keys, [1])}, started)

        linker.layer_done_counter.complete.assert_not_called()
        self.assertEqual(started, [keys])

    def test_read_plan_failure_is_reported_after_layer_wait(self):
        counter = ReadPlanLoadCounter(2)
        index = counter.update_producer()
        plan = Mock()
        plan.wait.side_effect = RuntimeError("range read failed")
        counter.bind(index, plan)
        counter.set_consumer(index)

        counter.wait_until(0)
        self.assertFalse(counter.finish(index))
        self.assertEqual(counter.plans, {})

    def test_metrics_port_uses_dp_rank_without_initialized_distributed(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            base = _get_mooncake_client_http_port()
            self.assertEqual(_get_mooncake_client_http_port(2), base + 2)


if __name__ == "__main__":
    unittest.main()
