# -*- coding: utf-8 -*-
from functools import wraps as _wraps
from functools import partial
import weakref
from hashlib import md5
from base64 import b64encode
import time
import logging
import random
import pydoc
from .clients import base, asyncache, tiered

try:
    from .mq import coherence
    from .clients import coherent
    no_coherence = False
except ImportError:
    no_coherence = True

from .clients.base import CacheMissError

try:
    import cython
except ImportError:
    # Make cython annotations work without cython
    _cy_noop = lambda f: f
    class _cython:
        locals = lambda **kw: _cy_noop
    globals()['cython'] = _cython

class NO_NAMESPACE:
    pass

class _NONE:
    pass

try:
    import multiprocessing
except ImportError:
    class multiprocessing:
        @staticmethod
        def cpu_count():
            return 1

def wraps(wrapped):
    w = _wraps(wrapped)
    def decor(wrapper):
        wrapper = w(wrapper)
        wrapper.__doc__ = "\n".join(
            pydoc.render_doc(wrapped, 'cached %s:').split('\n', 4)[:3]
            + list(filter(bool, [wrapper.__doc__])))
        return wrapper
    return decor

def _make_namespace(f, salt = None, salt2 = None):
    f = getattr(f, 'im_func', f)
    fname = getattr(f, '__name__', None)
    if fname is None:
        fname = getattr(f, 'func_name', None)
    if fname is None:
        # FTW
        return repr(f)

    mname = getattr(f, '__module__', '')

    fcode = getattr(f, '__code__', None)
    if fcode is None:
        fcode = getattr(f, 'func_code', None)
    if fcode is not None:
        fpath = '%s:%d' % (fcode.co_filename, fcode.co_firstlineno)
    else:
        fpath = ''

    try:
        body_digest = md5(fpath.encode("utf8"))
        if salt:
            body_digest.update(salt.encode("utf8"))
        if salt2:
            body_digest.update(salt2.encode("utf8"))
        if fcode:
            body_digest.update(getattr(fcode, 'co_code', b''))
        return "%s.%s#%s" % (mname,fname,b64encode(body_digest.digest()).rstrip(b"=\n"))
    except Exception as e:
        return repr(f)

def _simple_put_deferred(future, client, f, key, ttl, *p, **kw):
    defer = asyncache.Defer(f, *p, **kw)
    if future is not None:
        defer.future = future
    return client.put(key, defer, ttl)

def _coherent_put_deferred(shared, async_ttl, future, client, f, key, ttl, *p, **kw):
    return client.put_coherently(key, ttl,
        lambda : not shared.contains(key, async_ttl),
        future,
        f, *p, **kw)

class CacheStats(object):
    __slots__ = (
        'hits', 'misses', 'errors',
        'sync_misses', 'sync_errors',
        'min_miss_time', 'max_miss_time', 'sum_miss_time', 'sum_miss_time_sq',
        'miss_time_histogram', 'miss_time_histogram_bins', 'miss_time_histogram_max', 'wait_time',
    )

    def __init__(self):
        self.clear()
        self.set_histogram_bins(0, None)

    def clear(self):
        self.hits = self.misses = self.errors = self.sync_misses = self.sync_errors = 0
        self.max_miss_time = self.sum_miss_time = self.sum_miss_time_sq = self.wait_time = 0
        self.min_miss_time = -1

    def set_histogram_bins(self, bins, maxtime):
        if bins <= 0:
            self.miss_time_histogram_bins = self.miss_time_histogram_max = 0
            self.miss_time_histogram = None
        else:
            self.miss_time_histogram = [0] * bins
            self.miss_time_histogram_bins = bins
            self.miss_time_histogram_max = maxtime

    def add_histo(self, time):
        if self.miss_time_histogram:
            hmax = self.miss_time_histogram_max
            hbins = self.miss_time_histogram_bins
            hbin = min(hbins-1, min(hmax, time) * hbins / hmax)
            self.miss_time_histogram[hbin] += 1

decorated_functions = weakref.WeakSet()

def cached(client, ttl,
        key = lambda *p, **kw:(p,frozenset(kw.items()) or ()),
        namespace = None,
        value_serialization_function = None,
        value_deserialization_function = None,
        async_writer_queue_size = None,
        async_writer_workers = None,
        async_writer_threadpool = None,
        async_writer_kwargs = None,
        async_ttl = None,
        async_client = None,
        async_expire = None,
        lazy_kwargs = {},
        async_lazy_recheck = False,
        async_lazy_recheck_kwargs = {},
        async_processor = None,
        async_processor_workers = None,
        async_processor_threadpool = None,
        async_processor_kwargs = None,
        renew_time = None,
        future_sync_check = None,
        initialize = None,
        decorate = None,
        timings = True,
        ttl_spread = True,
        autonamespace_version_salt = None,
        _eff_async_ttl = None,
        _put_deferred = None,
        _fput_deferred = None,
        _lazy_recheck_put_deferred = None,
        _flazy_recheck_put_deferred = None ):
    """
    This decorator provides cacheability to suitable functions.

    To be considered suitable, the values received on parameters must be immutable types (otherwise caching will
    be unreliable).

    Caches are thread-safe only if the provided clients are thread-safe, no additional safety is provided. If you
    have a thread-unsafe client you want to make safe, use a (ReadWrite)SyncAdapter. Since synchronization adapters
    only apply to store manipuation functions, and not the wrapped function, deadlocks cannot occur.

    The decorated function will provide additional behavior through

    Attributes
    ----------

        client: the backing cache client. The provided client is never used as-is, and instead is wrapped in a
            NamespaceWrapper. This is it.

        ttl: The configured TTL

        async_ttl: The configured async TTL

        clear(): forget all cached values. Since the client might be shared, it will only increase an internal
            revision mark used to decorate keys, so the cache will not be immediately purged. For that, use
            client.clear() (but beware that it will also clear other caches sharing the same client).

        invalidate(...): mimicking the underlying function's signature, it will, instead of invoking the function,
            invalidate cached entries sharing the call's key.

        expire(...): like invalidate, but the key is just marked as out-of-date, requiring immediate refresh,
            but not completely invalid.

        put(_cache_put, ...): mimicking the underlying function's signature after the first positional argument,
            except for one keyword argument _cache_put, it will forcibly set the cached value for that key to
            be what's supplied in _cache_put. Although there will be no invocation of the target function,
            the write will be synchronous unless the underlying cache client is async, and for external caches
            this might still mean a significant delay.

        refresh(...): mimicking the underlying function's signature, it will forcefully invoke the function,
            regardless of cache status, and refresh the cache with the returned value.

        uncached(...): this is the underlying function, undecorated.

        peek(...): mimicking the underlying function's signature, it will query the cache without ever invoking
            the underlying function. If the cache doesn't contain the key, a CacheMissError will be raised.

        get_ttl(...): mimicking the underlying function's signature, it will query the cache without ever invoking
            the underlying function, and return both the result and the ttl. Misses return NONE as value and
            a negative ttl, instead of raising a CacheMissError.

        lazy(...): mimicking the underlying function's signature, it will behave just like a cached function call,
            except that if there is no value, instead of waiting for one to be computed, it will just raise
            a CacheMissError. If the access is async, it will start a background computation. Otherwise, it will
            behave just as peek.

        future(...): probably the preferred way of accessing the cache on an asynchronous app, it will return
            a decorated function that will return futures that will receive the value when done.
            For straight calls, if lazy would not raise a CacheMissError, the future will already contain the value,
            resulting in no delays. If the client is tiered and has remote tiers, it's recommendable to add proper
            lazy_kwargs to avoid this synchronous call blocking. Other forms of access perform similarly, always
            returning a future immediately without blocking. Alternatively, setting future_sync_check to False
            will disable this check and always do it through the processor.

        async(): if the underlying client is async, it will return the decorated function (self). Otherwise, it will
            be another decorated function, created on demand, backed by the same client wrapped in an async adapter.
            As such, it can be used to perform asynchronous operations on an otherwise synchronous function.

        on_promote: subdecorator that registers callbacks that will receive promotion events from this function.

        on_value: subdecorator that registers callbacks that will receive freshly computed values from this function.
            The callback will receive the value as its first argument, and all the original arguments for the call
            that initiated the computation following that argument (including keyword arguments).

        stats: cache statistics, containing:
            hits - number of cache hits
            misses - number of cache misses
            errors - number of exceptions caught
            sync_misses - number of synchronous (blocking) misses
            sync_errors - number of synchronous exceptions (propagated to caller)

            min_miss_time - minimum time spent computing a miss
            max_miss_time - maximum time spent computing a miss
            sum_miss_time - total time spent computing a miss (divide by misses and get an average)
            sum_miss_time_sq - sum of squared times spent computing a miss (to compute standard deviation)

            miss_time_histogram - histogram of times spent computing misses, computed only if histogram
                bins and limits are set.
            miss_time_histogram_bins - number of bins configured
            miss_time_histogram_max - maximum histogram time configured

            wait_time - time spent waiting for async updates, that's sync miss time on async queries

            reset(): clear all statistics
            set_histogram_bins(bins, max): configure histogram collection to use "bins" bins spanning
                from 0 to max. If bins is set to 0, max is ignored, and histogram collection is disabled

            All values are approximate, as no synchroniation is attempted while updating.

            sync_misses and sync_errors are caller-visible misses or exceptions. The difference with
            misses and errors respectively are the number of caller-invisible misses or errors.

    Parameters
    ----------

        client: the cache store client to be used

        ttl: the time, in seconds, during which values remain valid.

        callkey: the given key-computing callable

        renew_time: if not None, the time, in seconds, to add to the TTL when an item is scheduled for
            refresh. This renews the current item at a cost, but prevents concurrent readers from attempting
            their own refresh in a rather simple way, short of using a coherence protocol. Roughly
            equivalent to the dogpile pattern with a timeout as specified.

        key: (optional) A key derivation function, that will take the same arguments as the underlying function,
            and should return a key suitable to the client. If not provided, a default implementation that will
            usually work for most primitive argument types will be used.

        namespace: (optional) If provided, the namespace used to identify cache entries will be the one provided.
            If not, a default one will be derived out of the function's module and name, which may differ between
            platforms, so you'll want to provide a stable one for shared caches. If NO_NAMEPSACE is given,
            no namespace decoration will be applied. Specify if somehow collisions are certain not to occur.

        autonamespace_version_salt: (optional) If provided, it will alter the automatically generated namespace
            in a predictable and stable way. Can be used to force version upgrades when the automatic namespace
            is not able to pick up code changes.

        value_serialization_function: (optional) If provided, values will not be stored directly into the cache,
            but the result of applying this function to them. Use if the cache is remote and does not natively
            support the results given by the underlying function, or if stripping of runtime-specific data is
            required.

        value_deserialization_function: (optional) Counterpart to value_serialization_function, it will be applied
            on values coming from the cache, before returning them as cached function calls.

        async_writer_queue_size: (optional) Writer queue size used for the async() client. Default is 100.

        async_writer_workers: (optional) Number of async workers for the async() client.
            Default is multiprocessing.cpu_count

        async_writer_threadpool: (optional) Threadpool to be used for the async writer, instead of
            workers, this can specify a specific thread pool to use (perhaps shared among other writers).
            It can also be a callable, in which case it must be a factory function that takes the number
            of workers as argument and returns a threadpool to be used. It's recommendable to always use
            factories instead of instances, to avoid premature thread initialization.

        async_writer_kwargs: (optional) Optional arguments to be used when constructing AsyncWriteCacheClient
            wrappers. Ignored if an explicit async_client is given.

        async_client: (optional) Alternatively to async_writer_queue_size and async_writer_workers, a specific
            async client may be specified. It is expected this client will be an async wrapper of the 'client'
            mandatorily specified, that can be shared among decorated functions (to avoid multiplying writer
            threads).

        async_ttl: (optional) The TTL at which an async refresh starts. For example, async_ttl=1 means to start
            a refresh just 1 second before the entry expires. It must be smaller than ttl. Default is half the TTL.
            If negative, it means ttl - async_ttl, which reverses the logic to mean "async_ttl" seconds after
            creation of the cache entry.

        async_processor_workers: (optional) Number of threads used for async cache operations (only applies to
            future() calls, other async operations are configured with async_writer args). Only matters if the
            clients perform expensive serialization (there's no computation involved otherwise). Default is
            multiprocessing.cpu_count

        async_processor_threadpool: (optional) Threadpool to be used for the async processor, instead of
            workers, this can specify a specific thread pool to use (perhaps shared among other processors).
            It can also be a callable, in which case it must be a factory function that takes the number
            of workers as argument and returns a threadpool to be used. It's recommendable to always use
            factories instead of instances, to avoid premature thread initialization.

        async_processor_kwargs: (optional) Optional arguments to be used when constructing AsyncCacheProcessors.
            Ignored if an explicit async_processor is given.

        async_processor: (optional) Shared processor to utilize in future() calls.

        future_sync_check: (optional) Whether to perform a quick synchronous check of the cache with lazy_kwargs
            in order to avoid stalling on the processor for cached access. Set to False if there's no non-blocking
            tier in the given cache client.

        async_expire: (optional) A callable that will get an expired key, when TTL falls below async_ttl.
            Common use case is to pass a first-tier's expire bound method, thus initiating a refresh.

        lazy_kwargs: (optional) kwargs to send to the client's getTtl when doing lazy requests. Useful for
            tiered clients, that can accept access modifiers through kwargs

        async_lazy_recheck: (optional) when sending lazy_kwargs that may result in spurious CacheMissError,
            specifying this on True will trigger an async re-check of the cache (to verify the need for an
            async refresh).

        async_lazy_recheck_kwargs: (optional) when setting async_lazy_recheck, recheck kwargs can be specified
            here (default empty).

        initialize: (optional) A callable hook to be called right before all accesses. It should initialize whatever
            is needed initialized (like daemon threads), and only once (it should be a no-op after it's called once).
            It can return True to avoid being called again (any return value that doesn't evaluate to True will
            be ignored).

        decorate: (optional) A decorator to apply to all call-like decorated functions. Since @cached creates many
            variants of the function, this is a convenience over manually decorating all variants.

        timings: (optional) Whether to gather timing statistics. If true, misses will be timed, and timing data
            will be included in the stats attribute. It does imply some overhead. Default is True.

        ttl_spread: (optional - default True). If None, the TTL will be used as-is for cache insertions.
            If True, an automatic TTL spread will be computed so insertions and recomputatins are better distributed
            in time. If a number (float or int), an equal-type random amount between [-ttl_spread, ttl_spread]
            will be added.
    """
    if value_serialization_function or value_deserialization_function:
        client = base.DecoratedWrapper(client,
            value_decorator = value_serialization_function,
            value_undecorator = value_deserialization_function )
    if namespace is not None and namespace is not NO_NAMESPACE:
        client = base.NamespaceWrapper(namespace, client)
        if async_client:
            async_client = base.NamespaceMirrorWrapper(client, async_client)

    if not client.is_async:
        if async_writer_queue_size is None:
            async_writer_queue_size = 100
        if async_writer_workers is None:
            async_writer_workers = multiprocessing.cpu_count()

    if async_ttl is None:
        async_ttl = ttl / 2
    elif async_ttl < 0:
        async_ttl = ttl + async_ttl

    if ttl_spread is True:
        ttl_spread = min(ttl, async_ttl, abs(ttl-async_ttl)) / 2
    if ttl_spread:
        spread_type = type(ttl_spread)
        eff_ttl = lambda r = random.random : ttl + spread_type(ttl_spread * (r() - 0.5))
    else:
        eff_ttl = lambda : ttl

    if _put_deferred is None:
        _fput_deferred = _simple_put_deferred
        _put_deferred = partial(_fput_deferred, None)
    if _lazy_recheck_put_deferred is None:
        _flazy_recheck_put_deferred = _simple_put_deferred
        _lazy_recheck_put_deferred = partial(_flazy_recheck_put_deferred, None)

    if async_processor is not None and async_processor_workers is None:
        async_processor_workers = multiprocessing.cpu_count()

    if async_writer_kwargs is None:
        async_writer_kwargs = {}
    async_writer_kwargs.setdefault('threadpool', async_writer_threadpool)

    if async_processor_kwargs is None:
        async_processor_kwargs = {}
    async_processor_kwargs.setdefault('threadpool', async_processor_threadpool)

    # Copy to avoid modifying references to caller objects
    elazy_kwargs = lazy_kwargs.copy()
    easync_lazy_recheck_kwargs = async_lazy_recheck_kwargs.copy()
    eget_async_lazy_recheck_kwargs = easync_lazy_recheck_kwargs.copy()

    EMPTY_KWARGS = {}
    _NONE_ = _NONE
    Future = asyncache.Future

    def decor(f):
        if namespace is None:
            salt2 = repr((ttl,))
            nclient = base.NamespaceWrapper(_make_namespace(
                f, salt = autonamespace_version_salt, salt2 = salt2), client)
            if async_client:
                nasync_client = base.NamespaceMirrorWrapper(nclient, async_client)
            else:
                nasync_client = async_client
        else:
            nclient = client
            nasync_client = async_client

        stats = CacheStats()

        if initialize is not None:
            def _initialize():
                nonlocal _initialize
                stop_initializing = initialize()
                if stop_initializing:
                    _initialize = None
        else:
            _initialize = None

        # static async ttl spread, to avoid inter-process contention
        if ttl_spread:
            if _eff_async_ttl:
                eff_async_ttl = _eff_async_ttl
            else:
                eff_async_ttl = max(async_ttl / 2, async_ttl - spread_type(ttl_spread * 0.25 * random.random()))
        else:
            eff_async_ttl = async_ttl

        # Wrap and track misses and timings
        if timings:
            of = f
            @wraps(of)
            @cython.locals(t0=cython.double, t1=cython.double, t=cython.double)
            def af(*p, **kw):
                stats.misses += 1
                try:
                    t0 = time.time()
                    rv = of(*p, **kw)
                    t1 = time.time()
                    t = t1-t0
                    try:
                        if t > stats.max_miss_time:
                            stats.max_miss_time = t
                        if stats.min_miss_time < 0 or stats.min_miss_time > t:
                            stats.min_miss_time = t
                        stats.sum_miss_time += t
                        stats.sum_miss_time_sq += t*t
                        if stats.miss_time_histogram:
                            stats.add_histo(t)
                    except:
                        # Ignore stats collection exceptions.
                        # Quite possible since there is no thread synchronization.
                        pass
                    if value_callbacks:
                        try:
                            _value_callback(rv, *p, **kw)
                        except:
                            # Just log callback exceptions, orthogonal behavior shouldn't propagate to the caller
                            logging.getLogger('chorde').error("Error on value callback", exc_info = True)
                            pass
                    return rv
                except:
                    stats.errors += 1
                    raise
            @wraps(of)
            def f(*p, **kw):
                stats.sync_misses += 1
                return af(*p, **kw)
        else:
            of = f
            @wraps(of)
            def af(*p, **kw):  # lint:ok
                stats.misses += 1
                try:
                    rv = of(*p, **kw)
                    if value_callbacks:
                        try:
                            _value_callback(rv, *p, **kw)
                        except:
                            # Just log callback exceptions, orthogonal behavior shouldn't propagate to the caller
                            logging.getLogger('chorde').getLogger('chorde').error("Error on value callback", exc_info = True)
                            pass
                    return rv
                except:
                    stats.errors += 1
                    raise
            @wraps(of)
            def f(*p, **kw):  # lint:ok
                stats.sync_misses += 1
                rv = af(*p, **kw)
                if value_callbacks:
                    try:
                        _value_callback(rv, *p, **kw)
                    except:
                        # Just log callback exceptions, orthogonal behavior shouldn't propagate to the caller
                        logging.getLogger('chorde').error("Error on value callback", exc_info = True)
                        pass
                return rv

        @wraps(of)
        def cached_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return f(*p, **kw)

            try:
                rv = nclient.get(callkey, **get_kwargs)
                stats.hits += 1
            except CacheMissError:
                rv = f(*p, **kw)
                nclient.put(callkey, rv, eff_ttl())
            return rv
        if decorate is not None:
            cached_f = decorate(cached_f)

        @wraps(of)
        def get_ttl_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return f(*p, **kw)
            rv = nclient.getTtl(callkey, **get_kwargs)
            if rv[1] < 0:
                stats.misses += 1
            else:
                stats.hits += 1
            return rv
        if decorate is not None:
            get_ttl_f = decorate(get_ttl_f)

        @wraps(of)
        @cython.locals(t0=cython.double, t1=cython.double, t=cython.double)
        def async_cached_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return f(*p, **kw)

            client = aclient
            __NONE = _NONE_
            rv, rvttl = client.getTtl(callkey, __NONE, **get_kwargs)

            if (rv is __NONE or rvttl < eff_async_ttl) and not client.contains(callkey, eff_async_ttl):
                if renew_time is not None:
                    if rv is not __NONE:
                        nclient.renew(callkey, eff_async_ttl + renew_time)
                    elif placeholder_value_fn_cell:
                        placeholder = placeholder_value_fn_cell[0](*p, **kw)
                        nclient.add(callkey, placeholder, eff_async_ttl + renew_time)
                        rv = placeholder
                        rvttl = eff_async_ttl + renew_time
                # Launch background update
                _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
            elif rv is not __NONE:
                if rvttl < eff_async_ttl:
                    client.promote(callkey, ttl_skip = eff_async_ttl, **get_kwargs)
                stats.hits += 1

            if rv is __NONE:
                # Must wait for it
                if timings:
                    t0 = time.time()
                client.wait(callkey)
                rv, rvttl = client.getTtl(callkey, __NONE, **get_kwargs)
                if rv is __NONE or rvttl < eff_async_ttl:
                    # FUUUUU
                    rv = f(*p, **kw)
                stats.sync_misses += 1
                if timings:
                    t1 = time.time()
                    t = t1-t0
                    stats.wait_time += t

            return rv
        if decorate is not None:
            async_cached_f = decorate(async_cached_f)

        @wraps(of)
        def future_cached_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return fclient.do_async(f, *p, **kw)

            client = aclient
            clientf = fclient
            frv = Future()
            __NONE = _NONE_

            if future_sync_check:
                # Quick sync call with lazy_kwargs
                rv, rvttl = client.getTtl(callkey, __NONE, **elazy_kwargs)
            else:
                rv = __NONE
                rvttl = -1

            if (rv is __NONE or rvttl < eff_async_ttl):
                # The hard way
                if rv is __NONE:
                    # It was a miss, so wait for setting the value
                    def on_value(value):
                        stats.hits += 1
                        frv.set(value[0])
                        # If it's stale, though, start an async refresh
                        if value[1] < eff_async_ttl and not nclient.contains(callkey, eff_async_ttl,
                                **easync_lazy_recheck_kwargs):
                            if renew_time is not None and (rv is not __NONE or lazy_kwargs):
                                nclient.renew(callkey, eff_async_ttl + renew_time)
                            _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                    def on_miss():
                        if renew_time is not None and placeholder_value_fn_cell:
                            placeholder = placeholder_value_fn_cell[0](*p, **kw)
                            nclient.add(callkey, placeholder, eff_async_ttl + renew_time)
                        _fput_deferred(frv, client, af, callkey, eff_ttl(), *p, **kw)
                    def on_exc(exc_info):
                        stats.errors += 1
                        return frv.exc(exc_info)
                    clientf.getTtl(callkey, ttl_skip = eff_async_ttl, **eget_async_lazy_recheck_kwargs)\
                        .on_any(on_value, on_miss, on_exc)
                else:
                    # It was a stale hit, so set the value now, but start a touch-refresh
                    stats.hits += 1
                    frv._set_nothreads(rv)
                    def on_value(contains):  # lint:ok
                        if not contains:
                            if renew_time is not None and (rv is not __NONE or lazy_kwargs):
                                nclient.renew(callkey, eff_async_ttl + renew_time)
                            _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                        else:
                            # just promote
                            nclient.promote(callkey, ttl_skip = eff_async_ttl, **get_kwargs)
                    def on_miss():  # lint:ok
                        if renew_time is not None and placeholder_value_fn_cell:
                            placeholder = placeholder_value_fn_cell[0](*p, **kw)
                            nclient.add(callkey, placeholder, eff_async_ttl + renew_time)
                        _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                    def on_exc(exc_info):  # lint:ok
                        stats.errors += 1
                    clientf.contains(callkey, eff_async_ttl, **easync_lazy_recheck_kwargs)\
                        .on_any_once(on_value, on_miss, on_exc)
            else:
                stats.hits += 1
                frv._set_nothreads(rv)
            return frv
        if decorate is not None:
            future_cached_f = decorate(future_cached_f)

        @wraps(of)
        def lazy_cached_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                raise CacheMissError

            try:
                rv = nclient.get(callkey, **elazy_kwargs)
                stats.hits += 1
                return rv
            except CacheMissError:
                stats.misses += 1
                raise
        if decorate is not None:
            lazy_cached_f = decorate(lazy_cached_f)
        peek_cached_f = lazy_cached_f

        @wraps(of)
        def future_lazy_cached_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                raise CacheMissError

            client = aclient
            frv = Future()
            __NONE = _NONE_

            if future_sync_check:
                # Quick sync call with lazy_kwargs
                rv, rvttl = client.getTtl(callkey, __NONE, **elazy_kwargs)
            else:
                rv = __NONE
                rvttl = -1

            if (rv is __NONE or rvttl < eff_async_ttl):
                # The hard way
                clientf = fclient
                if rv is __NONE and (not future_sync_check or async_lazy_recheck):
                    # It was a preliminar miss, so wait for a recheck to set the value
                    def on_value(value):
                        if value[1] >= 0:
                            stats.hits += 1
                            frv.set(value[0])
                        else:
                            # Too stale
                            frv.miss()
                        # If it's stale, though, start an async refresh
                        if value[1] < eff_async_ttl and not client.contains(callkey, eff_async_ttl,
                                **easync_lazy_recheck_kwargs):
                            if renew_time is not None and (rv is not __NONE or lazy_kwargs):
                                nclient.renew(callkey, eff_async_ttl + renew_time)
                            _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                    def on_miss():
                        # Ok, real miss, report it and start the computation
                        frv.miss()
                        if renew_time is not None and placeholder_value_fn_cell:
                            placeholder = placeholder_value_fn_cell[0](*p, **kw)
                            nclient.add(callkey, placeholder, eff_async_ttl + renew_time)
                        _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                    def on_exc(exc_info):
                        stats.errors += 1
                        frv.exc(exc_info)
                    clientf.getTtl(callkey, ttl_skip = eff_async_ttl, **eget_async_lazy_recheck_kwargs)\
                        .on_any(on_value, on_miss, on_exc)
                else:
                    # It was a stale hit or permanent miss, so set the value now, but start a touch-refresh
                    if rv is __NONE:
                        stats.misses += 1
                        frv._miss_nothreads()
                    else:
                        stats.hits += 1
                        frv._set_nothreads(rv)
                    def on_value(contains):  # lint:ok
                        if not contains:
                            if renew_time is not None and (rv is not __NONE or lazy_kwargs):
                                nclient.renew(callkey, eff_async_ttl + renew_time)
                            _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                        else:
                            # just promote
                            nclient.promote(callkey, ttl_skip = eff_async_ttl, **get_kwargs)
                    def on_miss():  # lint:ok
                        if renew_time is not None and placeholder_value_fn_cell:
                            placeholder = placeholder_value_fn_cell[0](*p, **kw)
                            nclient.add(callkey, placeholder, eff_async_ttl + renew_time)
                        _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                    def on_exc(exc_info):  # lint:ok
                        stats.errors += 1
                    clientf.contains(callkey, eff_async_ttl, **easync_lazy_recheck_kwargs)\
                        .on_any_once(on_value, on_miss, on_exc)
            else:
                stats.hits += 1
                frv._set_nothreads(rv)
            return frv
        if decorate is not None:
            future_lazy_cached_f = decorate(future_lazy_cached_f)

        @wraps(of)
        def future_peek_cached_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                raise CacheMissError

            client = aclient
            clientf = fclient
            frv = Future()
            __NONE = _NONE_

            if future_sync_check:
                # Quick sync call with lazy_kwargs
                rv, rvttl = client.getTtl(callkey, __NONE, **elazy_kwargs)
            else:
                rv = __NONE
                rvttl = -1

            if (rv is __NONE or rvttl < eff_async_ttl):
                # The hard way
                if rv is __NONE and (not future_sync_check or async_lazy_recheck):
                    # It was a miss, so wait for setting the value
                    def on_value(value):
                        if value[1] >= 0:
                            stats.hits += 1
                            return frv.set(value[0])
                        else:
                            # Too stale
                            stats.misses += 1
                            return frv.miss()
                    def on_miss():
                        # Ok, real miss, report it
                        stats.misses += 1
                        return frv.miss()
                    def on_exc(exc_info):
                        stats.errors += 1
                        return frv.exc(exc_info)
                    clientf.getTtl(callkey, ttl_skip = eff_async_ttl, **eget_async_lazy_recheck_kwargs)\
                        .on_any(on_value, on_miss, on_exc)
                else:
                    # It was a stale hit or permanent miss
                    if rv is __NONE or rvttl < 0:
                        stats.misses += 1
                        frv._miss_nothreads()
                    else:
                        stats.hits += 1
                        frv._set_nothreads(rv)
            else:
                stats.hits += 1
                frv._set_nothreads(rv)
            return frv
        if decorate is not None:
            future_peek_cached_f = decorate(future_peek_cached_f)

        @wraps(of)
        def future_get_ttl_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                raise CacheMissError

            # To-Do: intercept hits/misses and update stats?
            #   (involves considerable overhead...)
            return fclient.getTtl(callkey, **get_kwargs)
        if decorate is not None:
            future_get_ttl_f = decorate(future_get_ttl_f)

        @wraps(of)
        def invalidate_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            nclient.delete(callkey)
        if decorate is not None:
            invalidate_f = decorate(invalidate_f)

        @wraps(of)
        def future_invalidate_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            return fclient.delete(callkey)
        if decorate is not None:
            future_invalidate_f = decorate(future_invalidate_f)

        @wraps(of)
        def expire_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            nclient.expire(callkey)
        if decorate is not None:
            expire_f = decorate(expire_f)

        @wraps(of)
        def async_expire_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            aclient.expire(callkey)
        if decorate is not None:
            async_expire_f = decorate(async_expire_f)

        @wraps(of)
        def future_expire_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            return fclient.expire(callkey)
        if decorate is not None:
            future_expire_f = decorate(future_expire_f)

        @wraps(of)
        def put_f(*p, **kw):
            value = kw.pop('_cache_put')
            put_kwargs = kw.pop('_cache_put_kwargs', None)
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            nclient.put(callkey, value, eff_ttl(), **(put_kwargs or EMPTY_KWARGS))
        if decorate is not None:
            put_f = decorate(put_f)

        @wraps(of)
        def async_put_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            value = kw.pop('_cache_put')
            put_kwargs = kw.pop('_cache_put_kwargs', None)
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            aclient.put(callkey, value, eff_ttl(), **(put_kwargs or EMPTY_KWARGS))
        if decorate is not None:
            async_put_f = decorate(async_put_f)

        @wraps(of)
        def future_put_f(*p, **kw):
            value = kw.pop('_cache_put')
            put_kwargs = kw.pop('_cache_put_kwargs', None)
            try:
                callkey = key(*p, **kw)
            except:
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return
            return fclient.put(callkey, value, eff_ttl(), **(put_kwargs or EMPTY_KWARGS))
        if decorate is not None:
            future_put_f = decorate(future_put_f)

        @wraps(of)
        def async_lazy_cached_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                raise CacheMissError

            __NONE = _NONE_
            client = aclient

            rv, rvttl = client.getTtl(callkey, __NONE, **elazy_kwargs)

            if (rv is __NONE or rvttl < eff_async_ttl) and not client.contains(callkey, eff_async_ttl, **elazy_kwargs):
                if async_lazy_recheck:
                    stats.misses += 1

                    # send a Defer that touches the client with recheck kwargs
                    # before doing the refresh. Needs not be coherent.
                    def touch_key(*p, **kw):
                        rv, rvttl = nclient.getTtl(callkey, __NONE, ttl_skip = eff_async_ttl,
                            **eget_async_lazy_recheck_kwargs)
                        if (rv is __NONE or rvttl < eff_async_ttl) and not nclient.contains(callkey, eff_async_ttl,
                                **easync_lazy_recheck_kwargs):
                            if renew_time is not None and (rv is not __NONE or lazy_kwargs):
                                nclient.renew(callkey, eff_async_ttl + renew_time)
                            _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
                        return base.NONE

                    # This will make contains return True for this key, until touch_key returns
                    # This is actually good, since it'll result in immediate misses from now on,
                    # avoiding trying to queue up touch after touch
                    _lazy_recheck_put_deferred(client, touch_key, callkey, ttl, *p, **kw)
                else:
                    _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
            elif rv is not __NONE:
                if rvttl < eff_async_ttl:
                    if async_expire:
                        async_expire(callkey)
                    else:
                        # means client.contains(callkey, eff_async_ttl), so promote
                        client.promote(callkey, ttl_skip = eff_async_ttl, **get_kwargs)
                stats.hits += 1
            else:
                stats.misses += 1

            if rv is __NONE:
                raise CacheMissError(callkey)
            else:
                return rv
        if decorate is not None:
            async_lazy_cached_f = decorate(async_lazy_cached_f)

        @wraps(of)
        def refresh_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return

            rv = f(*p, **kw)
            nclient.put(callkey, rv, eff_ttl())
            return rv
        if decorate is not None:
            refresh_f = decorate(refresh_f)

        @wraps(of)
        def async_refresh_f(*p, **kw):
            if _initialize is not None:
                _initialize()
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return

            client = aclient
            if not client.contains(callkey, 0):
                _put_deferred(client, af, callkey, eff_ttl(), *p, **kw)
        if decorate is not None:
            async_refresh_f = decorate(async_refresh_f)

        @wraps(of)
        def future_refresh_f(*p, **kw):
            try:
                callkey = key(*p, **kw)
            except:
                # Bummer
                logging.getLogger('chorde').error("Error evaluating callkey", exc_info = True)
                stats.errors += 1
                return

            frv = Future()
            _fput_deferred(frv, aclient, af, callkey, eff_ttl(), *p, **kw)
            return frv
        if decorate is not None:
            future_refresh_f = decorate(future_refresh_f)

        if client.is_async:
            cached_f = async_cached_f
            lazy_cached_f = async_lazy_cached_f
        elif nasync_client:
            aclient = nasync_client
        else:
            aclient = None

        promote_callbacks = []
        value_callbacks = []
        get_kwargs = {}
        def _promote_callback(*p, **kw):
            for cb in promote_callbacks:
                cb(*p, **kw)
        def _value_callback(*p, **kw):
            for cb in value_callbacks:
                cb(*p, **kw)
        def on_promote_f(callback):
            promote_callbacks.append(callback)
            get_kwargs.setdefault('promote_callback', _promote_callback)
            elazy_kwargs.setdefault('promote_callback', _promote_callback)
            eget_async_lazy_recheck_kwargs.setdefault('promote_callback', _promote_callback)
            return callback
        def on_value_f(callback):
            value_callbacks.append(callback)
            return callback

        placeholder_value_fn_cell = []
        def placeholder_value_f(fn):
            placeholder_value_fn_cell[:] = [fn]

        fclient = None
        def future_f(initialize = True):
            nonlocal fclient
            if _initialize is not None:
                _initialize()
            if fclient is None and initialize:
                if aclient:
                    _client = aclient
                else:
                    async_f() # initializes aclient
                    _client = aclient

                if async_processor:
                    _client = async_processor.bound(_client)
                else:
                    _client = asyncache.AsyncCacheProcessor(async_processor_workers, _client,
                        **async_processor_kwargs)
                # atomic
                fclient = _client
                future_cached_f.client = fclient
            return future_cached_f

        if not client.is_async:
            def async_f(initialize = True):
                nonlocal aclient
                if _initialize is not None:
                    _initialize()
                if aclient is None and initialize:
                    # atomic
                    aclient = asyncache.AsyncWriteCacheClient(nclient,
                        async_writer_queue_size,
                        async_writer_workers,
                        **async_writer_kwargs)
                    async_cached_f.client = aclient
                return async_cached_f
            async_cached_f.clear = nclient.clear
            async_cached_f.client = aclient if aclient is not None else None
            async_cached_f.bg = weakref.ref(async_cached_f)
            async_cached_f.lazy = async_lazy_cached_f
            async_cached_f.refresh = async_refresh_f
            async_cached_f.peek = peek_cached_f
            async_cached_f.invalidate = invalidate_f
            async_cached_f.expire = async_expire_f
            async_cached_f.uncached = of
            async_cached_f.put = async_put_f
            async_cached_f.ttl = ttl
            async_cached_f.async_ttl = async_ttl
            async_cached_f.callkey = key
            async_cached_f.stats = stats
            async_cached_f.get_ttl = get_ttl_f
            async_cached_f._promote_callbacks = promote_callbacks
            async_cached_f._value_callbacks = value_callbacks
            async_cached_f.on_promote = on_promote_f
            async_cached_f.on_value = on_value_f
            async_cached_f.placeholder_value = placeholder_value_f
            cached_f.bg = async_f
            cached_f.lazy = lazy_cached_f
            cached_f.refresh = refresh_f
            cached_f.peek = peek_cached_f
            cached_f.invalidate = invalidate_f
            cached_f.expire = expire_f
            cached_f.put = put_f
        else:
            aclient = nclient
            cached_f.bg = async_f = weakref.ref(cached_f)
            cached_f.lazy = async_lazy_cached_f
            cached_f.refresh = async_refresh_f
            cached_f.peek = peek_cached_f
            cached_f.invalidate = invalidate_f
            cached_f.expire = expire_f
            cached_f.put = async_put_f

        cached_f.future = future_f
        cached_f.clear = nclient.clear
        cached_f.client = nclient
        cached_f.ttl = ttl
        cached_f.async_ttl = async_ttl or ttl
        cached_f.callkey = key
        cached_f.stats = stats
        cached_f.uncached = of
        cached_f.get_ttl = get_ttl_f
        cached_f._promote_callbacks = promote_callbacks
        cached_f._value_callbacks = value_callbacks
        cached_f.on_promote = on_promote_f
        cached_f.on_value = on_value_f
        cached_f.placeholder_value = placeholder_value_f

        future_cached_f.clear = lambda : fclient.clear()
        future_cached_f.client = None
        future_cached_f.bg = cached_f.bg
        future_cached_f.lazy = future_lazy_cached_f
        future_cached_f.refresh = future_refresh_f
        future_cached_f.peek = future_peek_cached_f
        future_cached_f.invalidate = future_invalidate_f
        future_cached_f.expire = future_expire_f
        future_cached_f.put = future_put_f
        future_cached_f.ttl = ttl
        future_cached_f.async_ttl = async_ttl
        future_cached_f.callkey = key
        future_cached_f.stats = stats
        future_cached_f.uncached = of
        future_cached_f.get_ttl = future_get_ttl_f
        future_cached_f._promote_callbacks = promote_callbacks
        future_cached_f._value_callbacks = value_callbacks
        future_cached_f.on_promote = on_promote_f
        future_cached_f.on_value = on_value_f
        future_cached_f.placeholder_value = placeholder_value_f

        decorated_functions.add(cached_f)
        return cached_f
    return decor

if not no_coherence:

    def coherent_cached(private, shared, ipsub, ttl,
            key = lambda *p, **kw:(p,frozenset(kw.items()) or ()),
            tiered_ = None,
            namespace = None,
            coherence_namespace = None,
            coherence_encoding = 'pyobj',
            coherence_timeout = None,
            value_serialization_function = None,
            value_deserialization_function = None,
            async_writer_queue_size = None,
            async_writer_workers = None,
            async_writer_threadpool = None,
            async_writer_kwargs = None,
            async_ttl = None,
            async_expire = None,
            lazy_kwargs = {},
            async_lazy_recheck = False,
            async_lazy_recheck_kwargs = {},
            async_processor = None,
            async_processor_workers = None,
            async_processor_threadpool = None,
            async_processor_kwargs = None,
            renew_time = None,
            future_sync_check = None,
            initialize = None,
            decorate = None,
            tiered_opts = None,
            ttl_spread = True,
            wait_time = None,
            autonamespace_version_salt = None,
            **coherence_kwargs ):
        """
        This decorator provides cacheability to suitable functions, in a way that maintains coherency across
        multiple compute nodes.

        For suitability considerations and common parameters, refer to cached. The following describes the
        aspects specific to the coherent version.

        The decorated function will provide additional behavior through attributes:
            coherence: the coherence manager created for this purpse

            ipsub: the given IPSub channel

        Params
            ipsub: An IPSub channel that will be used to publish and subscribe to update events.

            private: The private (local) cache client, the one that needs coherence.

            shared: The shared cache client, that reflects changes made by other nodes, or a tuple
                to specify multiple shared tiers.

            tiered_: (optional) A client that queries both, private and shared. By default, a TieredInclusiveClient
                is created with private and shared as first and second tier, which should be the most common case.
                The private client will be wrapped in an async wrapper if not async already, to be able to execute
                the coherence protocol asynchronously. This should be adequate for most cases, but in some,
                it may be beneficial to provide a custom client.

            tiered_opts: (optional) When using the default-constructed tiered client, you can pass additional (keyword)
                arguments here.

            coherence_namespace: (optional) There is usually no need to specify this value, the namespace in use
                by caching will be used for messaging as well, or if caching uses NO_NAMESPACE, the default that
                would be used instead. However, for really high-volume channels, sometimes it is beneficial to pick
                a more compact namespace (an id formatted with struct.pack for example).

            coherence_encoding: (optional) Keys will have to be transmitted accross the channel, and this specifies
                the encoding that will be used. The default 'pyobj' should work most of the time, but it has to
                be initialized, and 'json' or others could be more compact, depending on keys.
                (see CoherenceManager for details on encodings)

            coherence_timeout: (optional) Time (in ms) of peer silence that will be considered abnormal. Default
                is 2000ms, which is sensible given the IPSub protocol. You may want to increase it if node load
                creates longer hiccups.

            wait_time: (optional) Time (in ms) a call that is being computed externally will wait blocking for
                a result. Waiting forever (None, the default) could block the writer threadpool, so it's best to
                specify a reasonably adequate (for the application) timeout.

            Any extra argument are passed verbatim to CoherenceManager's constructor.
        """
        if async_ttl is None:
            async_ttl = ttl / 2
        elif async_ttl < 0:
            async_ttl = ttl + async_ttl

        if ttl_spread is True:
            ttl_spread = min(ttl, async_ttl, abs(ttl-async_ttl)) / 2

        if not private.is_async:
            if async_writer_queue_size is None:
                async_writer_queue_size = 100
            if async_writer_workers is None:
                async_writer_workers = multiprocessing.cpu_count()

        if async_writer_kwargs is None:
            async_writer_kwargs = {}
        async_writer_kwargs.setdefault('threadpool', async_writer_threadpool)

        def decor(f):
            if ttl_spread:
                spread_type = type(ttl_spread)
                eff_async_ttl = max(async_ttl / 2, async_ttl - spread_type(ttl_spread * 0.25 * random.random()))
            else:
                eff_async_ttl = None

            salt2 = repr((ttl,))
            if coherence_namespace is None:
                _coherence_namespace = _make_namespace(f, salt = autonamespace_version_salt, salt2 = salt2)
            else:
                _coherence_namespace = coherence_namespace

            if namespace is None:
                _namespace = _make_namespace(f, salt = autonamespace_version_salt, salt2 = salt2)
            else:
                _namespace = namespace

            if not private.is_async:
                nprivate = asyncache.AsyncWriteCacheClient(private,
                    async_writer_queue_size,
                    async_writer_workers,
                    **async_writer_kwargs)
            else:
                nprivate = private

            if not isinstance(shared, tuple):
                shareds = (shared,)
                sharedt = shared
            elif len(shared) == 1:
                shareds = shared
                sharedt = shared[0]
            else:
                shareds = shared
                sharedt = tiered.TieredInclusiveClient(*shareds, **(tiered_opts or {}))

            if tiered_ is None:
                ntiered = tiered.TieredInclusiveClient(nprivate, *shareds, **(tiered_opts or {}))
            else:
                ntiered = tiered_

            if _namespace is not NO_NAMESPACE:
                ntiered = base.NamespaceWrapper(_namespace, ntiered)
                nprivate = base.NamespaceMirrorWrapper(ntiered, nprivate)
                nshared = base.NamespaceMirrorWrapper(ntiered, sharedt)
            else:
                nshared = sharedt

            coherence_manager = coherence.CoherenceManager(
                _coherence_namespace, nprivate, nshared, ipsub,
                encoding = coherence_encoding,
                **coherence_kwargs)

            nclient = coherent.CoherentWrapperClient(ntiered, coherence_manager, coherence_timeout)

            expired_ttl = async_ttl
            if eff_async_ttl:
                expired_ttl = max(expired_ttl, eff_async_ttl)
            if renew_time:
                expired_ttl = max(expired_ttl, (eff_async_ttl or async_ttl) + renew_time)

            rv = cached(nclient, ttl,
                namespace = NO_NAMESPACE, # Already covered
                key = key,
                value_serialization_function = value_serialization_function,
                value_deserialization_function = value_deserialization_function,
                async_writer_queue_size = async_writer_queue_size,
                async_writer_workers = async_writer_workers,
                async_writer_threadpool = async_writer_threadpool,
                async_writer_kwargs = async_writer_kwargs,
                async_ttl = async_ttl,
                async_expire = async_expire,
                initialize = initialize,
                decorate = decorate,
                lazy_kwargs = lazy_kwargs,
                async_lazy_recheck = async_lazy_recheck,
                async_lazy_recheck_kwargs = async_lazy_recheck_kwargs,
                async_processor = async_processor,
                async_processor_workers = async_processor_workers,
                async_processor_threadpool = async_processor_threadpool,
                async_processor_kwargs = async_processor_kwargs,
                renew_time = renew_time,
                future_sync_check = future_sync_check,
                ttl_spread = ttl_spread,
                _eff_async_ttl = eff_async_ttl,
                _put_deferred = partial(_coherent_put_deferred, nshared, expired_ttl, None,
                    expire_private = nprivate.expire),
                _fput_deferred = partial(_coherent_put_deferred, nshared, expired_ttl,
                    wait_time=wait_time,
                    expire_private = nprivate.expire) )(f)
            rv.coherence = coherence_manager
            rv.ipsub = ipsub
            return rv
        return decor
