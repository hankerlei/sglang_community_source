from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import Future
from copy import deepcopy
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()
    return getattr(params, "dp_rank", None) or 0


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class ReadPlanLoadCounter:
    """Expose native ReadPlan layer waits through the linker's completion API."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = self.consumer_index = -1
        self.plans: dict[int, Future] = {}
        self.errors: dict[int, BaseException] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.plans[self.producer_index] = Future()
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def bind(self, index: int, plan) -> None:
        self.plans[index].set_result(plan)

    def fail(self, index: int, error: BaseException) -> None:
        future = self.plans.get(index)
        if future is not None and not future.done():
            future.set_exception(error)
        self.errors[index] = error

    def complete(self, index: int, layer: int) -> None:
        # Non-source MLA peers use the normal layer counter instead.
        raise RuntimeError("ReadPlan does not support synthetic layer completion")

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        if index in self.errors:
            return
        future = self.plans.get(index)
        if future is None:
            return
        try:
            future.result().wait(threshold)
        except BaseException as error:
            # Finish the model forward before reporting the error. Raising here
            # can leave peer TP ranks blocked in a later model collective.
            self.errors[index] = error
            logger.error("Mooncake ReadPlan layer %d failed: %s", threshold, error)

    def finish(self, index: int) -> bool:
        # The forward may skip sparse layers; still wait for the whole plan
        # before the scheduler decides whether its result can be used.
        self.consumer_index = index
        self.wait_until(self.num_layers - 1)
        self.plans.pop(index, None)
        return self.errors.pop(index, None) is None

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.plans.clear()
        self.errors.clear()


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        extra_config=None,
        storage=None,
        forward_stream=None,
    ):
        self.forward_stream = forward_stream
        self.page_size = params.page_size
        self.page_wise_load_threshold = getattr(
            server_args, "mooncake_page_wise_load_threshold", 10
        )
        self.page_wise_load_batch_size = getattr(
            server_args, "mooncake_page_wise_load_batch_size", 128
        )
        self.enable_page_wise_load = getattr(
            server_args, "mooncake_enable_page_wise_load", False
        )
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
            )
        if self.page_wise_load_batch_size <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-batch-size must be positive, got "
                f"{self.page_wise_load_batch_size}."
            )
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config = deepcopy(extra_config or {})
        # TODO: unlike `HiCacheController._generate_storage_config`, this sets no
        # `dp_rank`. Aligning the storage-config producers changes persisted
        # storage keys, so it waits for a storage-key format change.
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
            dp_rank=getattr(params, "dp_rank", None) or 0,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                enable_client_http_server=params.enable_metrics,
            )
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(server_args, params),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if getattr(server_args, "extra_metric_labels", None):
                labels.update(server_args.extra_metric_labels)
            collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        self.read_plan_enabled = os.environ.get("SGLANG_MOONCAKE_READ_PLAN") == "1"
        self.read_plan_reuse_ranges = (
            os.environ.get("SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES") == "1"
        )
        if self.read_plan_reuse_ranges and not self.read_plan_enabled:
            raise ValueError(
                "ReadPlan range reuse requires SGLANG_MOONCAKE_READ_PLAN=1"
            )
        if self.read_plan_reuse_ranges and self.enable_page_wise_load:
            raise ValueError(
                "ReadPlan range reuse is incompatible with page-wise loading"
            )
        if self.read_plan_enabled and not callable(
            getattr(self.storage.store, "create_read_plan", None)
        ):
            raise RuntimeError(
                "SGLANG_MOONCAKE_READ_PLAN=1 requires a Mooncake client with create_read_plan"
            )
        if (
            self.read_plan_enabled
            and server_args.enable_linker_mla_dedup
            and rank_replicated
            and tp_size > 1
        ):
            raise ValueError(
                "Mooncake ReadPlan is incompatible with --enable-linker-mla-dedup"
            )
        self.register_buffers()
        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        if server_args.enable_linker_mla_dedup and rank_replicated and tp_size > 1:
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
            logger.info(
                "MLA linker rank-0 loading enabled: tp_rank=%d/%d", tp_rank, tp_size
            )
        self.layer_done_counter = (
            ReadPlanLoadCounter(self.num_layers)
            if self.read_plan_enabled
            else LayerWiseLoadCounter(
                self.num_layers,
                self._broadcast_loaded_layers if self.mla_broadcaster else None,
            )
        )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.pending_load_tokens: dict[str, int] = {}
        self.load_batch_rids: dict[int, list[str]] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, float, object, str | None] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        logical_pages = {
            page_key for transfer in expanded for page_key in transfer.keys
        }
        pending_load_tokens = getattr(self, "pending_load_tokens", None)
        if pending_load_tokens is None:
            pending_load_tokens = self.pending_load_tokens = {}
        pending_load_tokens[rid] = len(logical_pages) * self.page_size
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        getattr(self, "pending_load_tokens", {}).pop(rid, None)
        return self.pending_loads.pop(rid, None) is not None

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        # An MLA dedup peer's layers complete at start; its data arrives through
        # the broadcast on the forward stream, ordered before any later reuse.
        success = self.layer_done_counter.finish(counter_index)
        if self.read_plan_enabled:
            self._finish_prefetch_metrics(
                self.load_batch_rids.pop(counter_index, []), success
            )
        return success

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        broadcaster = self.mla_broadcaster
        if broadcaster is not None:
            # Compare the logical load order (not local slots), including empty
            # batches, before deciding collectively whether this batch can
            # broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = hashlib.sha256(repr(plan).encode()).digest()
            digests = [None] * torch.distributed.get_world_size(self.tp_group)
            torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
            if any(other != digest for other in digests):
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        if self.read_plan_enabled:
            self.load_batch_rids[counter_index] = list(pending)
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready = self.record_transfer_ready()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (0, prepared, ready)
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((counter_index, pending, ready))
        self.stats["load"] += len(pending)
        return counter_index

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, ready = batch
        if first == 0:
            ready.wait(device_module.current_stream())
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                ready,
            )

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready = task
                success = False
                try:
                    ready.synchronize()
                    success = self.load_layer_wise(
                        counter_index, list(pending.values())
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    if not self.read_plan_enabled:
                        self._finish_prefetch_metrics(list(pending), success)
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> bool:
        started = []
        try:
            if getattr(self, "read_plan_enabled", False):
                self.load_with_read_plan(counter_index, request_transfers)
                return True
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            for keys, _ in batches.values():
                result = self.storage.store.batch_get_session_start(keys)
                if list(result) != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )
                started.append(keys)

            if self.enable_page_wise_load and any(
                len(keys) >= self.page_wise_load_threshold
                for keys, _ in batches.values()
            ):
                self._load_page_wise(counter_index, batches, started)
                return True

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={expected}"
                        )
                self.layer_done_counter.complete(counter_index, layer)
            return True
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
            return False
        finally:
            cleanup_error = None
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")
                    cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error

    def _finish_prefetch_metrics(self, rids: list[str], success: bool) -> None:
        pending_load_tokens = getattr(self, "pending_load_tokens", {})
        tokens = sum(pending_load_tokens.pop(rid, 0) for rid in rids)
        if not success or tokens <= 0:
            return
        self._log_prefetched_tokens(tokens)
        recorder = getattr(
            getattr(self.storage, "store", None), "record_prefetched_tokens", None
        )
        if recorder is not None:
            try:
                recorder(tokens)
            except BaseException:
                logger.warning(
                    "Failed to record Mooncake prefetched token metric.",
                    exc_info=True,
                )

    def _log_prefetched_tokens(self, tokens: int) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None or tokens <= 0:
            return
        try:
            collector.log_prefetched_tokens(tokens)
        except BaseException:
            logger.warning(
                "Failed to record SGLang direct-storage prefetch metrics.",
                exc_info=True,
            )

    def _load_page_wise(self, counter_index: int, batches, started) -> None:
        """Load all layer ranges for each page before exposing the data."""
        for name, (keys, locations) in batches.items():
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    raise ValueError(
                        f"Mooncake pool={name} layer={layer} produced "
                        f"{len(layer_ptrs)} range entries for {len(keys)} keys."
                    )
                for index in range(len(keys)):
                    ptrs[index].extend(layer_ptrs[index])
                    sizes[index].extend(layer_sizes[index])
                    offsets[index].extend(layer_offsets[index])

            for start in range(0, len(keys), self.page_wise_load_batch_size):
                end = start + self.page_wise_load_batch_size
                chunk_keys = keys[start:end]
                chunk_sizes = sizes[start:end]
                result = self.storage.store.batch_get_into_multi_buffer_ranges(
                    chunk_keys,
                    ptrs[start:end],
                    chunk_sizes,
                    offsets[start:end],
                )
                expected = [sum(item) for item in chunk_sizes]
                if (
                    result is None
                    or isinstance(result, int)
                    or list(result) != expected
                ):
                    raise RuntimeError(
                        f"Mooncake range get failed for pool={name}, "
                        f"complete_page: transferred={result}, expected={expected}"
                    )

        # Page-wise loading gives up layer overlap. Release the read sessions
        # only after every complete page is loaded, and before any layer becomes
        # visible to the model. Remove successful releases so the caller's
        # finally block only retries a session whose cleanup raised.
        for keys in list(started):
            self.storage.store.batch_get_session_end(keys)
            started.remove(keys)
        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def _prepare_read_plan_layouts(self, request_transfers: list[list[PoolTransfer]]):
        # Consolidate index copies once per pool, preserving request/key order.
        # Each component describes (base, row stride, byte count, source offset).
        # Locations may be non-contiguous; Mooncake expands addresses in C++.
        batches = {}
        for transfers in request_transfers:
            for transfer in transfers:
                keys, indices = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                indices.append(transfer.host_indices)

        layouts = []
        for name, (keys, indices) in batches.items():
            pool = self.pools[name]
            joined = indices[0] if len(indices) == 1 else torch.cat(indices)
            locations = pool.prepare_locations(joined)
            layout = []
            for layer in range(self.num_layers):
                index = pool.layer_mapping.get(layer)
                buffer_indices = (index,) if isinstance(index, int) else index
                layout.append(
                    []
                    if index is None
                    else [
                        (*component[buffer_index], offsets[buffer_index])
                        for component, offsets in zip(
                            pool.buffer_meta, pool._component_offsets
                        )
                        for buffer_index in buffer_indices
                    ]
                )
            layouts.append((keys, locations, pool.packed, layout))

        return layouts

    def load_with_read_plan(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
    ) -> None:
        layouts = self._prepare_read_plan_layouts(request_transfers)
        plan = self.storage.store.create_read_plan(
            layouts,
            self.num_layers,
            reuse_ranges=self.read_plan_reuse_ranges,
            page_wise=self.enable_page_wise_load,
            buffer_owners=self.pools,
        )
        self.layer_done_counter.bind(counter_index, plan)
        # run() and wait() release the GIL. Each layer becomes visible only after
        # every pool's bytes have been checked; the last wait includes cleanup.
        # In page-wise mode a single batch_get carries all groups per key, so the
        # first wait(0) blocks until every page is complete and later waits are
        # no-ops, matching _load_page_wise's all-or-nothing release.
        plan.run()

    def offload(
        self, transfers: list[PoolTransfer], session_id: str | None = None
    ) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready = self.record_transfer_ready()
        self.offload_queue.put(
            (expanded, tokens, time.perf_counter(), ready, session_id)
        )
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, started, ready, session_id = task
                ready.synchronize()
                extra_info = (
                    HiCacheStorageExtraInfo(extra_info={"session_id": session_id})
                    if session_id
                    else None
                )
                results = self.storage.batch_set_v2(expanded, extra_info)
                success = all(all(pool_results) for pool_results in results.values())
                self._log_l4_backup_metric(
                    tokens, time.perf_counter() - started, success
                )
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, started, _, _ = task
                    self._log_l4_backup_metric(
                        tokens, time.perf_counter() - started, False
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def _log_l4_backup_metric(
        self, tokens: int, duration: float, success: bool
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            # This branch intentionally does not guess DFS versus local disk.
            # Its Mooncake configuration owns that routing decision.
            collector.log_l4_backup("mooncake", tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 backup metrics.",
                exc_info=True,
            )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        getattr(self, "pending_load_tokens", {}).clear()
        self.load_batch_rids.clear()
        self.load_queue.join()
        self.offload_queue.join()
        self.broadcast_loads.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
