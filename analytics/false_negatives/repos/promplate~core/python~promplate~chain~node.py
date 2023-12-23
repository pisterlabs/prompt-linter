from collections import ChainMap
from typing import Any, Callable, Mapping, MutableMapping, TypeVar

from ..llm.base import *
from ..prompt.template import Context, Loader, Template
from .utils import appender, resolve


class ChainContext(ChainMap, dict):
    def __init__(
        self,
        primary: MutableMapping[Any, Any] | None,
        /,
        *fallback: Mapping[Any, Any] | None,
    ):
        fallback = tuple(m for m in fallback if m is not None)
        if isinstance(primary, ChainContext):
            self.primary_map = primary.primary_map
            self.fallback_map = primary.fallback_map if not fallback else ChainMap(*fallback, *primary.fallback_map.maps)  # type: ignore
        else:
            self.primary_map = ChainMap() if primary is None else ChainMap(primary)
            self.fallback_map = ChainMap() if not fallback else ChainMap(*fallback)  # type: ignore
        super().__init__(self.primary_map, self.fallback_map)

    @property
    def result(self):
        return self.__getitem__("__result__")

    @result.setter
    def result(self, result):
        self.__setitem__("__result__", result)

    @result.deleter
    def result(self):
        self.__delitem__("__result__")

    def __or__(self, other: Mapping | None):
        if other is None:
            return self.copy()

        if isinstance(other, ChainContext):
            ctx = self.copy()
            ctx |= other
            return ctx

        return super().__or__(other)

    def __ror__(self, other: Mapping | None):
        if other is None:
            return self.copy()

        if isinstance(other, ChainContext):
            ctx = self.copy()
            ctx.primary_map.maps.extend(other.primary_map)
            ctx.fallback_map.maps.extend(other.fallback_map)
            return ctx

        return super().__ror__(other)

    def __ior__(self, other: MutableMapping | None):
        if other is not None and other is not self:
            if isinstance(other, ChainContext):
                self.primary_map.maps[0:0] = other.primary_map
                self.fallback_map.maps[0:0] = other.fallback_map
            else:
                return super().__ior__(other)
        return self

    def __repr__(self):
        return f"<ChainContext primary={self.primary_map} fallback={self.fallback_map}>"


CTX = TypeVar("CTX", Context, ChainContext)


Process = Callable[[CTX], CTX | None]

AsyncProcess = Callable[[CTX], Awaitable[CTX | None]]


class AbstractChain(Protocol):
    def run(
        self,
        context: Context | None = None,
        /,
        complete: Complete | None = None,
    ) -> ChainContext:
        ...

    async def arun(
        self,
        context: Context | None = None,
        /,
        complete: Complete | AsyncComplete | None = None,
    ) -> ChainContext:
        ...

    context: Context

    complete: Complete | AsyncComplete | None


class Interruptable(AbstractChain, Protocol):
    def _run(
        self,
        context: ChainContext,
        /,
        complete: Complete | None = None,
    ) -> ChainContext:
        ...

    async def _arun(
        self,
        context: ChainContext,
        /,
        complete: Complete | AsyncComplete | None = None,
    ) -> ChainContext:
        ...

    def run(self, context=None, /, complete=None) -> ChainContext:
        context = ChainContext(context, self.context)
        try:
            return self._run(context, complete) or context
        except JumpTo as jump:
            if jump.target is None or jump.target is self:
                return jump.chain.run(context | jump.context, complete)
            else:
                raise jump from None

    async def arun(self, context=None, /, complete=None) -> ChainContext:
        context = ChainContext(context, self.context)
        try:
            return await self._arun(context, complete) or context
        except JumpTo as jump:
            if jump.target is None or jump.target is self:
                return await jump.chain.arun(context | jump.context, complete)
            else:
                raise jump from None

    _context: Context | None

    @property
    def context(self):
        if self._context is None:
            self._context = {}
        return self._context

    @context.setter
    def context(self, context: Context | None):
        self._context = context

    @context.deleter
    def context(self):
        self._context = None


class Node(Loader, Interruptable):
    def __init__(
        self,
        template: Template | str,
        partial_context: Context | None = None,
        pre_processes: list[Process | AsyncProcess] | None = None,
        post_processes: list[Process | AsyncProcess] | None = None,
        complete: Complete | AsyncComplete | None = None,
        **config,
    ):
        self.template = Template(template) if isinstance(template, str) else template
        self._context = partial_context
        self.pre_processes = pre_processes or []
        self.post_processes = post_processes or []
        self.complete = complete
        self.run_config = config

    @property
    def pre_process(self):
        return appender(self.pre_processes)

    @property
    def post_process(self):
        return appender(self.post_processes)

    def _apply_pre_processes(self, context):
        for process in self.pre_processes:
            context |= process(context)

    def _apply_post_processes(self, context):
        for process in self.post_processes:
            context |= process(context)

    def _run(self, context, /, complete=None):
        complete = self.complete or complete
        assert complete is not None

        self._apply_pre_processes(context)
        prompt = self.template.render(context)

        context.result = complete(prompt, **self.run_config)

        self._apply_post_processes(context)

    async def _apply_async_pre_processes(self, context):
        for process in self.pre_processes:
            context |= await resolve(process(context))

    async def _apply_async_post_processes(self, context):
        for process in self.post_processes:
            context |= await resolve(process(context))

    async def _arun(self, context, /, complete=None):
        complete = self.complete or complete
        assert complete is not None

        await self._apply_async_pre_processes(context)
        prompt = await self.template.arender(context)

        context.result = await resolve(complete(prompt, **self.run_config))

        await self._apply_async_post_processes(context)

    def next(self, chain: AbstractChain):
        if isinstance(chain, Chain):
            return Chain(self, *chain)
        else:
            return Chain(self, chain)

    def __add__(self, chain: AbstractChain):
        return self.next(chain)

    def render(self, context: Context | None = None):
        context = ChainContext(context, self.context)
        self._apply_pre_processes(context)
        return self.template.render(context)

    async def arender(self, context: Context | None = None):
        context = ChainContext(context, self.context)
        await self._apply_async_pre_processes(context)
        return await self.template.arender(context)

    def __str__(self):
        return f"</{self.name}/>"


class Chain(Interruptable):
    def __init__(
        self,
        *nodes: AbstractChain,
        partial_context: Context | None = None,
        complete: Complete | AsyncComplete | None = None,
    ):
        self.nodes = nodes
        self._context = partial_context
        self.complete = complete

    def next(self, chain: AbstractChain):
        if isinstance(chain, Node):
            return Chain(*self, chain)
        elif isinstance(chain, Chain):
            return Chain(*self, *chain)
        else:
            raise NotImplementedError

    def __add__(self, chain):
        return self.next(chain)

    def __iter__(self):
        return iter(self.nodes)

    def _run(self, context, /, complete=None):
        for node in self.nodes:
            context = node.run(context, self.complete or complete)  # type: ignore

        return context

    async def _arun(self, context, /, complete=None):
        for node in self.nodes:
            context = await node.arun(context, self.complete or complete)

        return context

    def __repr__(self):
        return " + ".join(map(str, self.nodes))


class JumpTo(Exception):
    def __init__(
        self,
        chain: Interruptable,
        context: Context | None = None,
        target: Interruptable | None = None,
    ):
        self.chain = chain
        self.context = context
        self.target = target

    def __str__(self) -> str:
        return f"{self.target} does not exist in the hierarchy"
