from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
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
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _session_refresh_age_config() -> float:
    age_s = float(os.environ.get("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", "0") or 0)
    lease_ttl_s = float(os.environ.get("SGLANG_MOONCAKE_LEASE_TTL_S", "10") or 10)
    if not math.isfinite(age_s) or age_s < 0:
        raise ValueError(
            "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S must be finite and >= 0"
        )
    if not math.isfinite(lease_ttl_s) or lease_ttl_s <= 0:
        raise ValueError("SGLANG_MOONCAKE_LEASE_TTL_S must be finite and > 0")
    if age_s >= lease_ttl_s:
        logger.warning(
            "Mooncake session refresh age %.3fs is not below lease TTL %.3fs; "
            "disabling refresh skipping",
            age_s,
            lease_ttl_s,
        )
        return 0.0
    return age_s


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class MooncakeDirectLinker(UnifiedCacheLinker):
    # Unit-test fixtures may construct an instance without running __init__.
    session_refresh_age_s = 0.0

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
        self.host_prefetch_enabled = bool(
            getattr(server_args, "mooncake_enable_waiting_queue_dfs_prefetch", False)
        )
        if self.host_prefetch_enabled and (
            getattr(server_args, "schedule_policy", "fcfs") != "fcfs"
            or getattr(server_args, "disaggregation_mode", "null")
            not in ("null", "prefill")
            or params.pp_size != 1
        ):
            self.host_prefetch_enabled = False
            logger.warning(
                "Waiting-queue DFS prefetch disabled: requires FCFS scheduling, "
                "a prefill or non-disaggregated role, and pp_size=1."
            )
        self.page_size = params.page_size
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
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                enable_dfs_prefetch=self.host_prefetch_enabled,
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
        prefetch_enabled = self._resolve_host_prefetch_enabled(
            params, requested=self.host_prefetch_enabled
        )
        if self.host_prefetch_enabled and not prefetch_enabled:
            self.disable_waiting_queue_prefetch()
        self.host_prefetch_enabled = prefetch_enabled
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
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
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.mla_broadcaster else None,
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.prepared_load_sessions: dict[str, list[str]] = {}
        self.session_refcounts: dict[str, int] = {}
        self.session_sources: dict[str, str] = {}
        self.session_lease_base: dict[str, float] = {}
        self.session_refresh_age_s = (
            _session_refresh_age_config() if self.host_prefetch_enabled else 0.0
        )
        self._disable_refresh_skip_for_groups()
        self.session_lock = threading.Lock()
        self.host_prefetch_lock = threading.Lock()
        self.host_prefetch_entries: dict[str, dict[str, object]] = {}
        self.host_prefetch_generation = 0
        self.host_prefetch_limit = max(
            1,
            int(
                getattr(
                    server_args,
                    "mooncake_waiting_queue_dfs_prefetch_max_requests",
                    8,
                )
            ),
        )
        worker_count = max(
            1,
            int(getattr(server_args, "mooncake_waiting_queue_dfs_prefetch_workers", 2)),
        )
        self.host_prefetch_worker_count = min(worker_count, self.host_prefetch_limit)
        self.host_prefetch_queue: Queue[tuple[str, list[PoolTransfer]] | None] = Queue()
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, list[object], str | None] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.host_prefetch_threads: list[threading.Thread] = []
        if self.host_prefetch_enabled:
            for worker_id in range(self.host_prefetch_worker_count):
                thread = threading.Thread(
                    target=self.host_prefetch_thread_func,
                    daemon=True,
                    name=f"mooncake-host-prefetch-tp{tp_rank}-{worker_id}",
                )
                self.host_prefetch_threads.append(thread)
                thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def _local_host_prefetch_capability(self, requested: bool) -> tuple[bool, str]:
        if not requested:
            return False, "disabled"
        store = self.storage.store
        required = (
            "batch_get_session_start_with_sources",
            "batch_get_session_prefetch",
            "batch_get_session_refresh",
            "dfs_prefetch_arena_available",
            "disable_dfs_prefetch",
        )
        missing = [
            name for name in required if not callable(getattr(store, name, None))
        ]
        if missing:
            return False, "Mooncake package lacks " + ", ".join(missing)
        try:
            if store.dfs_prefetch_arena_available():
                return True, "ready"
            status = getattr(store, "dfs_prefetch_arena_status", None)
            reason = status() if callable(status) else "unavailable"
        except Exception as exc:
            reason = f"arena query failed: {exc!r}"
        return False, f"DFS prefetch arena unavailable ({reason})"

    def _resolve_host_prefetch_enabled(
        self, params: CacheInitParams, *, requested: bool
    ) -> bool:
        local_ok, reason = self._local_host_prefetch_capability(requested)
        verdict = torch.tensor([int(local_ok)], dtype=torch.int)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced = False
            for group in (params.attn_cp_cache_group, params.attn_tp_cache_group):
                if group is not None and torch.distributed.get_world_size(group) > 1:
                    torch.distributed.all_reduce(
                        verdict, op=torch.distributed.ReduceOp.MIN, group=group
                    )
                    reduced = True
            if (
                not reduced
                and params.tp_cache_group is not None
                and torch.distributed.get_world_size(params.tp_cache_group) > 1
            ):
                torch.distributed.all_reduce(
                    verdict,
                    op=torch.distributed.ReduceOp.MIN,
                    group=params.tp_cache_group,
                )
        enabled = bool(verdict.item())
        if requested and not enabled:
            logger.warning(
                "Mooncake waiting-queue DFS prefetch disabled on all ranks; "
                "local status: %s",
                reason,
            )
        return enabled

    def waiting_queue_prefetch_enabled(self) -> bool:
        return self.host_prefetch_enabled

    def _disable_refresh_skip_for_groups(self) -> None:
        group_semantics = getattr(self.storage, "_can_use_group_semantics", None)
        if (
            self.session_refresh_age_s > 0
            and callable(group_semantics)
            and group_semantics()
        ):
            # A grouped object's reported TTL can exceed its remaining lease.
            logger.warning(
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S is ignored with "
                "Mooncake group semantics"
            )
            self.session_refresh_age_s = 0.0

    def disable_waiting_queue_prefetch(self) -> None:
        if self.host_prefetch_enabled and not self.storage.store.disable_dfs_prefetch():
            raise RuntimeError(
                "Mooncake cannot disable DFS prefetch after requests have started."
            )
        self.host_prefetch_enabled = False

    def _component_keys(self, transfers: list[PoolTransfer]) -> list[str]:
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        keys: list[str] = []
        seen: set[str] = set()
        for transfer in expanded:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            for key in self.storage._tag_keys(component_keys):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        if not self.host_prefetch_enabled:
            return False
        keys = self._component_keys(transfers)
        if not keys:
            return False
        reused: list[str] = []
        acquired: list[str] = []
        with self.session_lock:
            existing = self.prepared_load_sessions.get(rid)
            if existing is not None:
                if existing != keys:
                    return False
                reused = keys
            else:
                new_keys = []
                for key in keys:
                    if key in self.session_refcounts:
                        self.session_refcounts[key] += 1
                        self.session_lease_base.pop(key, None)
                        reused.append(key)
                        acquired.append(key)
                    else:
                        new_keys.append(key)
                lease_base = time.monotonic()
                try:
                    results, sources = (
                        self.storage.store.batch_get_session_start_with_sources(
                            new_keys
                        )
                        if new_keys
                        else ([], [])
                    )
                    results, sources = list(results), list(sources)
                except BaseException:
                    logger.warning("Mooncake session preparation failed", exc_info=True)
                    self._rollback_session_refs_locked(acquired)
                    return False
                for index, key in enumerate(new_keys):
                    if index < len(results) and results[index] == 0:
                        self.session_refcounts[key] = 1
                        self.session_sources[key] = str(
                            sources[index] if index < len(sources) else "unknown"
                        ).lower()
                        self.session_lease_base[key] = lease_base
                        acquired.append(key)
                if len(results) != len(new_keys) or any(code != 0 for code in results):
                    self._rollback_session_refs_locked(acquired)
                    return False
                self.prepared_load_sessions[rid] = acquired
        if reused and not self._refresh_session_keys(reused):
            self._abort_prepared_load_now(rid)
            return False
        return True

    def _refresh_session_keys(self, keys: list[str]) -> bool:
        if not keys:
            return False
        with self.session_lock:
            bases = [self.session_lease_base.get(key) for key in keys]
            private = all(self.session_refcounts.get(key, 0) == 1 for key in keys)
        oldest = None if None in bases else min(bases)
        age = float("inf") if oldest is None else time.monotonic() - oldest
        if (
            self.session_refresh_age_s > 0
            and private
            and age < self.session_refresh_age_s
        ):
            return True
        lease_base = time.monotonic()
        try:
            results = list(self.storage.store.batch_get_session_refresh(keys))
        except BaseException:
            logger.warning("Mooncake session refresh failed", exc_info=True)
            results = []
        ok = len(results) == len(keys) and all(code == 0 for code in results)
        with self.session_lock:
            for key in keys:
                if not ok:
                    self.session_lease_base.pop(key, None)
                elif key in self.session_lease_base:
                    self.session_lease_base[key] = lease_base
        return ok

    def submit_host_prefetch(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        if not self.host_prefetch_enabled:
            return False
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        if kv is None or not kv.keys:
            return False
        with self.host_prefetch_lock:
            active = sum(
                entry.get("state")
                in {
                    "queued",
                    "preparing",
                    "reading",
                    "dfs_prefetched",
                    "claimed",
                    "loading",
                }
                for entry in self.host_prefetch_entries.values()
            )
            if rid in self.host_prefetch_entries or active >= self.host_prefetch_limit:
                return False
            self.host_prefetch_generation += 1
            transfers = list(transfers)
            self.host_prefetch_entries[rid] = {
                "state": "queued",
                "cancelled": False,
                "transfers": transfers,
                "session_rid": (
                    f"__sglang_waiting_queue_prefetch__:{rid}:"
                    f"{self.host_prefetch_generation}"
                ),
            }
        self.host_prefetch_queue.put((rid, transfers))
        return True

    def get_host_prefetch_status(self, rid: str) -> str | None:
        if not self.host_prefetch_enabled:
            return None
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            return None if entry is None else str(entry.get("state"))

    def revalidate_host_prefetch(self, rid: str) -> bool:
        if not self.host_prefetch_enabled:
            return False
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "dfs_prefetched"
                or entry.get("cancelled")
            ):
                return False
            session_rid = str(entry["session_rid"])
        with self.session_lock:
            keys = list(self.prepared_load_sessions.get(session_rid, ()))
        return self._refresh_session_keys(keys)

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        if not self.host_prefetch_enabled:
            return False
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "no_prefetch_needed"
                or entry.get("cancelled")
            ):
                return False
            keys = list(entry.get("no_prefetch_keys", ()))
        if not keys:
            return False
        try:
            states = list(self.storage._batch_exist(keys))
        except BaseException:
            return False
        return len(states) == len(keys) and all(state == 1 for state in states)

    def claim_ready_host_prefetch(self, rid: str) -> bool:
        # Preserve the lock ordering fixed by 2b75b648f9: session RPC paths may
        # hold session_lock while publishing worker state.
        with self.session_lock, self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None or entry.get("state") != "dfs_prefetched":
                return False
            session_rid = str(entry["session_rid"])
            if (
                rid in self.prepared_load_sessions
                or session_rid not in self.prepared_load_sessions
            ):
                return False
            self.prepared_load_sessions[rid] = self.prepared_load_sessions.pop(
                session_rid
            )
            entry["state"] = "claimed"
            return True

    def cancel_host_prefetch(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return
            state = str(entry.get("state"))
            session_rid = str(entry["session_rid"])
            if state in {"preparing", "reading", "loading"}:
                entry["cancelled"] = True
                return
            self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid if state == "claimed" else session_rid)

    def host_prefetch_thread_func(self) -> None:
        while True:
            task = self.host_prefetch_queue.get()
            try:
                if task is None:
                    return
                rid, transfers = task
                with self.host_prefetch_lock:
                    entry = self.host_prefetch_entries.get(rid)
                    if entry is None or entry.get("transfers") is not transfers:
                        continue
                    session_rid = str(entry["session_rid"])
                    if entry.get("cancelled"):
                        self.host_prefetch_entries.pop(rid, None)
                        continue
                    entry["state"] = "preparing"
                try:
                    prepared = self.prepare_load(session_rid, transfers)
                except BaseException:
                    logger.warning(
                        "Mooncake DFS prefetch preparation failed for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    self._finish_host_prefetch(rid, entry, session_rid, "failed")
                    continue
                with self.session_lock:
                    keys = list(self.prepared_load_sessions.get(session_rid, ()))
                    sources = {
                        key: self.session_sources.get(key, "unknown") for key in keys
                    }
                if not prepared or not keys:
                    self._finish_host_prefetch(rid, entry, session_rid, "failed")
                    continue
                dfs_keys = [key for key in keys if sources[key] in {"dfs", "unknown"}]
                if not dfs_keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "no_prefetch_needed", keys=keys
                    )
                    continue
                with self.host_prefetch_lock:
                    current = self.host_prefetch_entries.get(rid)
                    if current is not entry or current.get("cancelled"):
                        cancelled = True
                    else:
                        cancelled = False
                        current["state"] = "reading"
                if cancelled:
                    self._finish_host_prefetch(rid, entry, session_rid, "failed")
                    continue
                try:
                    results = list(
                        self.storage.store.batch_get_session_prefetch(dfs_keys)
                    )
                    success = len(results) == len(dfs_keys) and all(
                        code == 0 for code in results
                    )
                except BaseException:
                    logger.warning(
                        "Mooncake DFS prefetch failed for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    success = False
                self._finish_host_prefetch(
                    rid, entry, session_rid, "dfs_prefetched" if success else "failed"
                )
            finally:
                self.host_prefetch_queue.task_done()

    def _finish_host_prefetch(
        self,
        rid: str,
        entry: dict[str, object],
        session_rid: str,
        outcome: str,
        *,
        keys: list[str] | None = None,
    ) -> None:
        keep_session = False
        with self.host_prefetch_lock:
            current = self.host_prefetch_entries.get(rid)
            cancelled = current is not entry or bool(
                current and current.get("cancelled")
            )
            if (
                current is entry
                and not cancelled
                and outcome in {"dfs_prefetched", "no_prefetch_needed"}
            ):
                current["state"] = outcome
                if outcome == "no_prefetch_needed":
                    current["no_prefetch_keys"] = list(keys or ())
                keep_session = outcome == "dfs_prefetched"
            elif current is entry:
                if cancelled:
                    self.host_prefetch_entries.pop(rid, None)
                else:
                    current["state"] = "failed"
        if not keep_session:
            self._abort_prepared_load_now(session_rid)

    def _rollback_session_refs_locked(self, keys: list[str]) -> None:
        to_end = []
        for key in keys:
            count = self.session_refcounts.get(key, 0)
            if count <= 1:
                self.session_refcounts.pop(key, None)
                self.session_sources.pop(key, None)
                self.session_lease_base.pop(key, None)
                to_end.append(key)
            else:
                self.session_refcounts[key] = count - 1
        if to_end:
            try:
                self.storage.store.batch_get_session_end(to_end)
            except BaseException:
                logger.warning("Mooncake session cleanup failed", exc_info=True)

    def _abort_prepared_load_now(self, rid: str) -> None:
        with self.session_lock:
            self._rollback_session_refs_locked(self.prepared_load_sessions.pop(rid, []))

    def abort_prepared_load(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is not None and entry.get("state") in {"claimed", "loading"}:
                self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid)

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
        # Direct-linker keys carry PP rank suffixes. Requiring every PP shard
        # here prevents PP0 from advertising a prefix another stage lacks.
        result = self.storage.batch_exists_v2(page_keys, expanded, query_all_pp=True)
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

    def revalidate_load(self, transfers: list[PoolTransfer]) -> bool:
        """Turn a match/load eviction race into a normal cache miss."""
        try:
            expanded = self.pool_group.resolve_transfers(
                transfers, allow_partial=True, allow_missing_kv=True
            )
            if not expanded:
                return True
            keys: list[str] = []
            for transfer in expanded:
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                if self.storage.pp_size > 1:
                    component_keys = [
                        key.replace(f"_pp{self.storage.pp_rank}_", f"_pp{pp_rank}_")
                        for key in component_keys
                        for pp_rank in range(self.storage.pp_size)
                    ]
                keys.extend(self.storage._tag_keys(component_keys))
            if not keys:
                return True
            states = list(self.storage._batch_exist(keys))
        except BaseException:
            logger.exception("Mooncake direct-linker load revalidation error")
            return False
        if len(states) != len(keys):
            logger.warning(
                "Mooncake load revalidation returned %d results for %d keys",
                len(states),
                len(keys),
            )
            return False
        missing = [key for key, state in zip(keys, states) if state != 1]
        if not missing:
            return True
        logger.warning(
            "Mooncake direct-linker load revalidation found %d/%d missing "
            "objects; falling back to recomputation",
            len(missing),
            len(keys),
        )
        failed_get_cache = getattr(self.storage, "failed_get_cache", None)
        if failed_get_cache is not None:
            failed_get_cache.update_batch([], missing)
        return False

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
        if self.host_prefetch_enabled:
            with self.host_prefetch_lock:
                entry = self.host_prefetch_entries.get(rid)
                if entry is not None and entry.get("state") == "claimed":
                    entry["state"] = "loading"
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        cancelled = self.pending_loads.pop(rid, None) is not None
        if cancelled and self.host_prefetch_enabled:
            self.abort_prepared_load(rid)
        return cancelled

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        # An MLA dedup peer's layers complete at start; its data arrives through
        # the broadcast on the forward stream, ordered before any later reuse.
        return self.layer_done_counter.finish(counter_index)

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
                # Only the broadcaster source reads Mooncake. Peers receive KV
                # over TP broadcast, so release any sessions they claimed at
                # admission instead of retaining native cache references.
                for rid in pending:
                    self.abort_prepared_load(rid)
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready_event = device_module.Event()
        ready_event.record()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (0, prepared, ready_event)
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, ready_event = batch
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
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
                ready_event,
            )

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                try:
                    ready_event.synchronize()
                    self.load_layer_wise(counter_index, list(pending.items()))
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> None:
        started: list[list[str]] = []
        borrowed_rids: list[str] = []
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            ordinary_keys: dict[PoolName, list[str]] = {}
            for rid, transfers in request_transfers:
                with self.session_lock:
                    borrowed = rid in self.prepared_load_sessions
                if self.host_prefetch_enabled and not borrowed:
                    raise RuntimeError(
                        f"Mooncake load admitted without a prepared session: rid={rid}"
                    )
                if borrowed:
                    borrowed_rids.append(rid)
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    tagged = self.storage._tag_keys(component_keys)
                    keys.extend(tagged)
                    if not borrowed:
                        ordinary_keys.setdefault(transfer.name, []).extend(tagged)
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            for keys in ordinary_keys.values():
                result = list(self.storage.store.batch_get_session_start(keys))
                # Partial starts still own successful sessions and must end.
                started.append(keys)
                if result != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )

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
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")
            for rid in borrowed_rids:
                self.abort_prepared_load(rid)

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
        ready_events = self.record_offload_ready()
        self.offload_queue.put((expanded, tokens, ready_events, session_id))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_events, session_id = task
                for event in ready_events:
                    event.synchronize()
                extra_info = (
                    HiCacheStorageExtraInfo(extra_info={"session_id": session_id})
                    if session_id
                    else None
                )
                results = self.storage.batch_set_v2(expanded, extra_info)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        if self.host_prefetch_threads:
            with self.host_prefetch_lock:
                for entry in self.host_prefetch_entries.values():
                    entry["cancelled"] = True
            self.host_prefetch_queue.join()
            # Loads that have not started can release their claimed sessions
            # now. Started loads must finish before their sessions are ended.
            for rid in list(self.pending_loads):
                self._abort_prepared_load_now(rid)
        self.pending_loads.clear()
        self.load_queue.join()
        self.offload_queue.join()
        if self.host_prefetch_threads:
            for rid in list(self.prepared_load_sessions):
                self._abort_prepared_load_now(rid)
            with self.host_prefetch_lock:
                self.host_prefetch_entries.clear()
        self.broadcast_loads.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        for _ in self.host_prefetch_threads:
            self.host_prefetch_queue.put(None)
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        for thread in self.host_prefetch_threads:
            thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
