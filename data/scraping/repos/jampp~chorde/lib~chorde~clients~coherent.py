# -*- coding: utf-8 -*-
import time
import logging

from .async import Defer, REGET, CancelledError
from .base import BaseCacheClient, NONE

from chorde.mq import coherence

class CoherentDefer(Defer):
    """
    Wrap a callable in this, and pass it as a value to an AsyncWriteCacheClient,
    and the evaluation of the given callable will happen asynchronously. The cache
    will return stale entries until the callable is finished computing a new value.
    Computation will be aborted if the coherence protocol shows someone else already
    computing, or if the shared cache is re-validated somehow when this method is
    called.
    """

    def __init__(self, callable_, *args, **kwargs):
        """
        Params
            callable_, args, kwargs: see Defer. All others must be passed by keyword.

            manager: A CoherenceManager instance that coordinates the corresponding cache clients.

            expired: A callable of the sort CoherenceManager takes, to re-check expiration status
                against the shared cache.

            expire_private: An optional callable that will be called with the given key (see below)
                right before receiving an OOB update. It should mark any privately held value for
                that key as expired, since a fresher one will be coming OOB. Usually the expire
                method of the private cache client works for this purpose.

            key: The key associated with this computation, to be provided to the CoherenceManager.

            timeout: Coherence protocol timeout, in ms, if the peers don't answer in this time,
                detached operation will be assumed and computation will proceed.
                Default: whatever the heartbeat timeout is on the underlying IPSub channel

            wait_time: Whether and how long to wait for other node's computation to be done.
                Normally, the default value of 0, which means "no wait", is preferrable so as not
                to stall deferred computation workers. However, in quick computations, it may be
                beneficial to provide a small wait time, to decrease latency in case some node
                goes down. This deferred would then take it upon him to start computing, and the
                whole group could be spared a whole cycle (versus just waiting for the value to
                be needed again), trading thoughput for latency. A value of None will cause
                infinite waits.
        """
        self.manager = kwargs.pop('manager')
        self.expired = kwargs.pop('expired')
        self.expire_private = kwargs.pop('expire_private', None)
        self.fetch = kwargs.pop('fetch', None)
        self.key = kwargs.pop('key')
        self.timeout = kwargs.pop('timeout', None)
        if self.timeout is None:
            self.timeout = self.manager.ipsub.heartbeat_push_timeout
        self.wait_time = kwargs.pop('wait_time', 0)
        self.computed = False
        self.aborted = False
        super(CoherentDefer, self).__init__(callable_, *args, **kwargs)

    def undefer(self):
        logger = logging.getLogger('chorde.coherence')
        while True:
            if not self.expired():
                logger.debug("Not computing because already fresh, key=%r", self.key)
                if hasattr(self, 'future'):
                    if self.expire_private is not None:
                        self.expire_private(self.key)
                    return REGET
                else:
                    return NONE
            else:
                computer = self.manager.query_pending(self.key, self.expired, self.timeout, True)
                if computer is None:
                    # My turn
                    logger.debug("Acquired computer duty on key=%r", self.key)
                    try:
                        rv = super(CoherentDefer, self).undefer()
                        if rv is not NONE:
                            logger.debug("Computing done on key=%r", self.key)
                            self.computed = True
                        else:
                            logger.debug("Computing skipped on key=%r", self.key)
                            self.aborted = True
                    except:
                        logger.debug("Computing aborted on key=%r due to exception", self.key, exc_info=True)
                        self.aborted = True
                        raise
                    return rv
                elif computer is coherence.OOB_UPDATE and not self.expired():
                    # Skip, tiered caches will read it from the shared cache and push it downstream
                    logger.debug("Fresh value available OOB on key=%r", self.key)
                    if hasattr(self, 'future'):
                        if self.expire_private is not None:
                            self.expire_private(self.key)
                        return REGET
                    else:
                        return NONE
                elif self.wait_time != 0:
                    logger.debug("Waiting for computation on node %r on key=%r", computer, self.key)
                    if self.manager.wait_done(self.key, timeout = self.wait_time):
                        logger.debug("Computation done (was waiting on node %r) on key=%r", computer, self.key)
                        if hasattr(self, 'future'):
                            if self.expire_private is not None:
                                self.expire_private(self.key)
                            return REGET
                        else:
                            return NONE
                    else:
                        # retry
                        logger.debug("Computation still in progress (waiting on node %r) on key=%r", computer, self.key)
                        continue
                else:
                    logger.debug("Computation in progress on node %r key=%r, not waiting", computer, self.key)
                    if hasattr(self, 'future'):
                        # Must cancel it if we're not going to wait
                        self.future.cancel()
                        self.future.set_exception(CancelledError())
                    return NONE

    def done(self):
        if self.computed:
            self.manager.mark_done(self.key)
        elif self.aborted:
            self.manager.mark_aborted(self.key)
        super(CoherentDefer, self).done()

class CoherentWrapperClient(BaseCacheClient):
    """
    Client wrapper that publishes cache activity with the coherence protocol.
    Given a manager, it will invoke its methods to make sure other instances
    on that manager are notified of invalidations.

    It adds another method of putting, put_coherently, that will additionally
    ensure that only one node is working on computing the result (thust taking
    a callable rather than a value, as defers do).

    The regular put method will publish being done, if the manager is configured
    in quick refresh mode, but will not attempt to obtain a computation lock,
    resulting in less overhead, decent consistency, but some duplication of
    effort. Therefore, put_coherently should be applied to expensive computations.

    If the underlying client isn't asynchronous, put_coherently will implicitly
    undefer the values, executing the coherence protocol in the calling thread.
    """

    def __init__(self, client, manager, timeout = 2000):
        self.client = client
        self.manager = manager
        self.timeout = timeout

    @property
    def async(self):
        return self.client.async

    @property
    def capacity(self):
        return self.client.capacity

    @property
    def usage(self):
        return self.client.usage

    def wait(self, key, timeout = None):
        # Hey, look at that. Since locally it it all happens on a Defer,
        # we can just wait on the wrapped client first to wait for ourselves
        if timeout is not None:
            deadline = time.time() + timeout
        else:
            deadline = None
        self.client.wait(key, timeout)

        # But in the end we'll have to ask the manager to talk to the other nodes
        if deadline is not None:
            timeout = int(max(0, deadline - time().time()) * 1000)
        else:
            timeout = None
        self.manager.wait_done(key, timeout=timeout)

    def put(self, key, value, ttl, coherence_timeout = None, **kw):
        manager = self.manager
        if manager.quick_refresh:
            # In quick refresh mode, we publish all puts
            if self.async and isinstance(value, Defer) and not isinstance(value, CoherentDefer):
                callable_ = value.callable_
                def done_after(*p, **kw):
                    rv = callable_(*p, **kw)
                    if rv is not NONE:
                        manager.fire_done([key])
                    return rv
                value.callable_ = done_after
                self.client.put(key, value, ttl, **kw)
            else:
                self.client.put(key, value, ttl, **kw)
                manager.fire_done([key], timeout = coherence_timeout)
        else:
            self.client.put(key, value, ttl, **kw)

    def put_coherently(self, key, ttl, expired, future, callable_, *args, **kwargs):
        """
        Another method of putting, that will additionally ensure that only one node
        is working on computing the result. As such, it takes  a callable rather
        than a value, as defers do.

        If the underlying client isn't asynchronous, put_coherently will implicitly
        undefer the value, executing the coherence protocol in the calling thread.

        Params
            key, ttl: See put
            expired: A callable that will re-check expiration status of the key on
                a shared cache. If this function returns False at mid-execution of
                the coherence protocol, the protocol and the computation will be
                aborted (assuming the underlying client will now instead fetch
                values from the shared cache).
            future: (optional) A future to be associated with the deferred
                computation.
            callable_, args, kwargs: See Defer. In contrast to normal puts, the
                callable may not be invoked if some other node has the computation
                lock.
        """
        wait_time = kwargs.pop('wait_time', 0)
        put_kwargs = kwargs.pop('put_kwargs', None) or None
        value = CoherentDefer(
            callable_,
            key = key,
            manager = self.manager,
            expired = expired,
            timeout = self.timeout,
            wait_time = wait_time,
            *args, **kwargs )
        if future is not None:
            value.future = future
        deferred = None
        try:
            if not self.async:
                # cannot put Defer s, so undefer right now
                deferred = value
                value = value.undefer()
                if value is NONE:
                    # Abort
                    return
            self.client.put(key, value, ttl, **(put_kwargs or {}))
        finally:
            if deferred is not None:
                deferred.done()

    def renew(self, key, ttl):
        self.client.renew(key, ttl)

    def delete(self, key, coherence_timeout = None):
        self.client.delete(key)

        # Warn others
        self.manager.fire_deletion(key, timeout = coherence_timeout)

    def expire(self, key):
        self.client.expire(key)

    def clear(self, coherence_timeout = None):
        self.client.clear()

        # Warn others
        self.manager.fire_deletion(coherence.CLEAR, timeout = coherence_timeout)

    def purge(self, *p, **kw):
        self.client.purge(*p, **kw)

    def getTtl(self, key, default = NONE, **kw):
        return self.client.getTtl(key, default, **kw)

    def get(self, key, default = NONE, **kw):
        return self.client.get(key, default, **kw)

    def getTtlMulti(self, keys, default = NONE, **kw):
        return self.client.getTtlMulti(keys, default, **kw)

    def getMulti(self, keys, default = NONE, **kw):
        return self.client.getMulti(keys, default, **kw)

    def contains(self, key, ttl = None, **kw):
        return self.client.contains(key, ttl, **kw)

    def promote(self, key, *p, **kw):
        return self.client.promote(key, *p, **kw)
