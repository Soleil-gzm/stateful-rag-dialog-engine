"""
ModuleRegistry：按需创建、缓存 Module 实例。

启动时不再一次性构造全部 36 个 Module，
真正用到某个 file_prefix 时才创建，并缓存备用。
"""
import logging
from collections import OrderedDict

from .module import Module
from .specs import ModuleSpec

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    按需构造 Module。默认缓存 64 个（总数 36，实际不会触发淘汰）。
    将来如果模块数量级扩大，LRU 会自动开始淘汰。
    """

    def __init__(self, embeddings, max_size: int = 64):
        self.embeddings = embeddings
        self.max_size = max_size
        self._cache: OrderedDict[str, Module] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, spec: ModuleSpec) -> Module:
        """
        返回 spec 对应的 Module。命中缓存则直接返回；
        未命中则构造一个，并加入缓存。
        """
        key = spec.file_prefix
        if key in self._cache:
            # LRU：命中后移到末尾
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        logger.info("加载 Module: %s (%s)", spec.file_prefix, spec.category)
        module = Module(spec.category, spec.file_prefix, self.embeddings)

        self._cache[key] = module
        if len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("淘汰 Module: %s", evicted_key)
        return module

    def stats(self) -> dict:
        """用于监控：命中率、当前缓存大小。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "capacity": self.max_size,
        }