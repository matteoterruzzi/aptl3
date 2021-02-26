"""
Actor model in a single-file module for simple and effective python pipelines,
based on queues and threading and/or multiprocessing.

See :class:`Actor`
"""

__all__ = [
    'ActorStats',
    'Actor',
    'MapActor',
    'ActorSystem',
]

import logging
import multiprocessing
import queue
import signal
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from time import perf_counter
from typing import Callable, Dict, Iterator, Optional, Union, Any, List, Set, NamedTuple, Tuple

MailBox = Union[
    queue.Queue,
    multiprocessing.JoinableQueue,
]


class _ActorStop(object):
    """System message asking the actor to stop
    NOTE: Only a stop message can be properly delivered to all actors in a pool: by leaving it in the mailbox."""
    pass


class _ActorFinish(object):
    """System message asking the actor to finish
    NOTE: Only one actor will receive this message, so don't rely on this mechanism to notify a whole actors pool.
          Pools must not block on `Actor.ask_available` in order to ensure they do not wait timeouts for nothing."""
    pass


class ActorStats(NamedTuple):
    """Stats returned by :method:`Actor.get_perf_counters`"""
    received: int
    concurrent: int
    outbound: int
    busy_time: float
    stopped: bool


class _SharedState(NamedTuple):
    """
    - lock: used for write synchronization
    - empty: condition notified when the total number of messages drops to zero (or when the system becomes degraded).
    - messages: shared counter of the total number of messages in the system (excluding
      system stop and finish notifications) incremented before putting msgs in the queues;
      can be used to let the source actors limit themselves.
    """
    lock: Union[threading.Lock, multiprocessing.Lock]
    empty: Union[threading.Condition, multiprocessing.Condition]
    messages: multiprocessing.Value

    def decrement_messages(self, n=1):
        with self.lock:
            old_value = self.messages.value
            self.messages.value = old_value - n
            if old_value <= n:
                self.empty.notify_all()

    def increment_messages(self, n=1):
        with self.lock:
            self.messages.value += n


class Actor(ABC):
    """
        From https://en.wikipedia.org/wiki/Actor_model:
        « An actor is a computational entity that, in response to a message it receives, can concurrently:
          - send a finite number of messages to other actors;
          - designate the behavior to be used for the next message it receives.  »

    Other actors are addressed by name (using a string).
    Two actors may be associated to the same name and may read from the same mailbox.
    Only one actor will receive a given message that has been put in the shared mailbox.
    Specific actor classes must implement the method `receive(msg)`.
    To send a message: `tell('other_actor', msg)`
    Contrary to the original model, actors cannot create other actors in this implementation.

    Subclasses are responsible of making sure that instances can be pickled when using multiprocessing
    and accesses to additional fields are synchronized when using multithreading.

    See :class:`ActorSystem` and module docstring.
    """

    def __init__(self):
        """Initialize the actor state. Subclasses MUST call super().__init__()"""
        # Queue routing (name and mailboxes) not known yet, will be set during `ActorSystem.start`
        self.__name: Optional[str] = None
        self.__mailboxes: Optional[Dict[str, MailBox]] = None
        self.__shared_state: Optional[_SharedState] = None

        # Performance statistics collection in shared memory (see `get_perf_counters`)
        self.__stats_lock = multiprocessing.Lock()
        self.__received = multiprocessing.Value('l', lock=False)
        self.__concurrent = multiprocessing.Value('l', lock=False)
        self.__outbound = multiprocessing.Value('l', lock=False)
        self.__busy_time = multiprocessing.Value('d', lock=False)

        # Flags (0=default, 1=set) do not require locking
        self.__stopped = multiprocessing.Value('b', lock=False)
        self.__finish = multiprocessing.Value('b', lock=False)

    def init_routing(self, name: str, mailboxes: Dict[str, MailBox], shared_state: _SharedState):
        """Initialize queue routing (name and mailboxes) during `ActorSystem.start`.
        NOTE: this method is repeatedly invoked on the same object with same values in case of a multithreading pool."""
        self.__name = name
        self.__mailboxes = mailboxes
        self.__shared_state = shared_state

    def init(self) -> None:
        """Called in the running thread. Subclasses can override this to initialize non-serializable fields."""
        pass

    def run(self) -> None:
        self.init()
        inbox = self.__mailboxes[self.__name]
        while not self.__stopped.value:
            msg = inbox.get()
            if isinstance(msg, _ActorStop):
                self.set_stop_flag()
                inbox.put(msg)  # Other actors on the same pool must receive the message
                break
            if isinstance(msg, _ActorFinish):
                inbox.task_done()
                continue
            with self.__stats_lock:
                self.__received.value += 1
                self.__concurrent.value += 1

            out = None
            _elapsed = None
            _tic = perf_counter()
            try:
                out = self.receive(msg)
                _elapsed = perf_counter() - _tic
                if out is not None:
                    _out_recipient, _out_msg = out
                    self.__mailboxes[_out_recipient].put(_out_msg)
                    # NOTE: 1-bounded, safe; still involved in deadlock detection
            finally:
                with self.__stats_lock:
                    self.__concurrent.value -= 1
                    if _elapsed is not None:
                        self.__busy_time.value += _elapsed
                if out is None:
                    self.__shared_state.decrement_messages(n=1)
                if _elapsed is None:
                    # the system is going to be degraded
                    with self.__shared_state.lock:
                        self.__shared_state.empty.notify_all()
                inbox.task_done()

    def set_stop_flag(self):
        """Irreversibly stop this actor instance AFTER processing the next message. You may prefer ActorSystem.stop"""
        self.__stopped.value = 1

    def set_finish_flag(self):
        """Can be used to suggest to subclasses to only finish processing the pending messages."""
        self.__finish.value = 1

    def get_pending(self) -> int:
        """
        Approximate and unreliable (due to race conditions) total number of messages in the system.

        Subclasses can use this to limit the production of additional messages
        (e.g. when this number it above a predefined threshold) to limit the risk
        of deadlocks in presence of actors with cyclic dependencies in conjunction
        with the use of an ActorSystem with queues with maxsize > 0 (the default).
        """
        return self.__shared_state.messages.value

    @property
    def finish_requested(self) -> bool:
        """Can be accessed by subclasses to know in anticipation that no further
        messages will arrive once the current pending messages are processed."""
        return bool(self.__finish.value)

    def tell(self, name: Optional[str], msg: Any, *,
             block: bool = True, timeout: float = 5.) -> None:
        """
        Send a message to another mailbox (if None, send to this and other actors associated to the same name).

        NOTE: `timeout=None` is NOT accepted, as deadlocks in circular flows would not be structurally prevented.

        Consumers that are also producers are responsible to actually consume the messages,
        without using this method in a loop to retry sending out other messages.

        In future, an additional `retry: bool` parameter may be added as an alternative to
        allow to indefinitely retry after an expired timeout, until a deadlock is detected.

        :param name: of the recipient mailbox
        :param msg: to be sent
        :param block: if False, send the message only if it's immediately possible; wait otherwise
        :param timeout: maximum wait time (seconds); it is mandatory to specify a positive timeout.
        """
        if timeout <= 0:
            raise ValueError(timeout)  # must be > 0
        if name is None:
            name = self.__name
        _tic = perf_counter()
        self.__shared_state.increment_messages(n=1)
        self.__mailboxes[name].put(msg, block=block, timeout=timeout)
        with self.__stats_lock:
            self.__outbound.value += 1
            self.__busy_time.value -= perf_counter() - _tic

    def ask_available(self, inbox: Optional[Union[str, MailBox]] = None, block=False, timeout=None) -> Iterator[Any]:
        """
        Iterates the available messages on the named inbox, possibly blocking.

        If optional args 'block' is true and 'timeout' is None, block if necessary
        until a msg is available. If 'timeout' is a non-negative number, it blocks
        at most 'timeout' seconds for each message and returns if no msg was
        available within that time. Otherwise ('block' is false), yields msg as
        long as they are immediately available ('timeout' ignored in that case).

        If no messages are available, then no messages are iterated.

        NOTE: with `block==True and timeout==None` it waits for future messages
        until stopped or asked to finish as described below.

        Stops yielding messages immediately after a system stop message.

        This method may also stop yielding messages to try to let the actor
        implementation handle the `finish_request` flag as soon as it set by
        :func:`ActorSystem.finish` during the execution of this method.

        For example, this can be used to try to collect a batch of messages::

            class MyBatchActor(Actor):
                def receive(msg):
                    batch = [msg]
                    batch.expand(self.ask_available(None, block=not self.finish_requested))
                    ...  # process batch (ignore msg)

        :param inbox: mailbox (or its name) to take messages from; defaults to None, i.e. the inbox of this actor.
        :param block: passed to the get method of the inbox queue
        :param timeout: passed to the get method of the inbox queue
        """
        if inbox is None:
            inbox = self.__name
        if isinstance(inbox, str):
            inbox = self.__mailboxes[inbox]
        while not self.__stopped.value:
            _tic = perf_counter()
            try:
                msg = inbox.get(block=block, timeout=timeout)
            except queue.Empty:
                return
            if isinstance(msg, _ActorStop):
                if inbox is self.__mailboxes[self.__name]:
                    self.set_stop_flag()
                inbox.put(msg)
                return
            with self.__stats_lock:
                if block:
                    self.__busy_time.value -= perf_counter() - _tic
                if isinstance(msg, _ActorFinish):
                    self.__finish.value = 1
                    block, timeout = False, None
                else:
                    self.__received.value += 1
            if not isinstance(msg, _ActorFinish):
                self.__shared_state.decrement_messages(n=1)
            inbox.task_done()
            # NOTE: at least one pending job is guaranteed at this point in the stack of run>receive>iter_available

            if not isinstance(msg, _ActorFinish):
                yield msg  # Do not let the caller stop the iteration before the above operations are done.

    def get_perf_counters(self) -> ActorStats:
        """
        Multiprocessing and thread-safe access to this actor's statistics.
        Synchronization should not affect the performance unless this function is called too frequently.

        NOTE: received >= completions + concurrent; (concurrent == 0) => (received == completions)
        """
        with self.__stats_lock:
            return ActorStats(
                received=self.__received.value,
                concurrent=self.__concurrent.value,
                outbound=self.__outbound.value,
                busy_time=self.__busy_time.value if self.__busy_time.value > 0 else 0,  # NOTE: ask_available decreases
                stopped=bool(self.__stopped.value)
            )

    @abstractmethod
    def receive(self, msg: Any) -> Optional[Tuple[str, Any]]:
        """
        This method is called by the actor when a msg is receive.

        Subclasses of Actor must implement it.

        The return value is a message to send out.

        The advantage of returning the message (at most one, instead of calling :func:`Actor.tell`)
        is the structural limitation that allows optimizations and is safe against
        deadlocks caused by circular wait-for relations when queues fill up.

        :return: None or a new message: tuple(recipient name, message)
        """
        raise NotImplementedError


class MapActor(Actor):
    """Very simple, stateless actor that can be used with a msg transform function and a recipient."""

    def __init__(self, f: Callable[[Any], Any], recipient: str) -> None:
        """
        :param f: mapping function (will be pickled, so prefer global functions instead of lambdas)
        :param recipient: name of the mailbox where mapped messages should be sent
        """
        super().__init__()
        self._f = f
        self._recipient = recipient

    def receive(self, msg: Any) -> Optional[Tuple[str, Any]]:
        result = self._f(msg)
        return self._recipient, result


class _CollectorActor(Actor):
    """
    Very simple actor that can be used to temporarily store in memory the messages that it receives

    The stored messages will only be accessible from the interface of the system that created this actor.
    """

    def __init__(self):
        super().__init__()
        self.store: queue.Queue = queue.Queue(maxsize=0)

    def receive(self, msg: Any) -> Optional[Tuple[str, Any]]:
        self.store.put_nowait(msg)
        return None


########################################################################################################################


def _multiprocessing_actor_run(actor: Actor):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    actor.run()


class _ActorStruct:
    name: str
    actor: Actor
    executor_class: type
    executor: Union[threading.Thread, multiprocessing.Process]

    def __init__(self, name: str, actor: Actor, executor_class: type) -> None:
        self.name = name
        self.actor = actor
        self.executor_class = executor_class


class ActorSystem:
    """
    Very simple Actor model implementation with python queues and pools.

    Not thread-safe: access to an ActorSystem object itself is allowed only from one thread (the creator)
    """

    _maxsize: int

    _mailboxes: Dict[str, MailBox]
    _actor_names: Set[str]
    _actors: List[_ActorStruct]
    _actor_instances: Set[Actor]

    _multiprocessing: bool
    _started: bool

    _enqueued_stop: Dict[str, int]
    _enqueued: Dict[str, int]
    _dequeued: Dict[str, int]
    _time_start: float
    _time_func: Callable[[], float]

    _shared_state: _SharedState

    _collectors: Dict[str, _CollectorActor]

    _monitor_th: Optional['_ActorSystemMonitorThread']

    def __init__(self, *,
                 maxsize: int = 1,
                 use_multiprocessing: bool = False,
                 time_func: Callable[[], float] = perf_counter,
                 monitor_thread: bool = False,
                 ):
        """
        Initialize the actor system.
        :param maxsize: of the message queues
        :param use_multiprocessing: set True if you intend to add processes
        :param time_func: to measure performance
        :param monitor_thread: set True if you want the actor system to request finish when the caller thread finishes
        """

        self._maxsize = maxsize

        self._mailboxes = OrderedDict()
        self._actor_names = set()
        self._actors = list()
        self._actor_instances = set()

        self._multiprocessing = use_multiprocessing
        self._started = False

        self._enqueued_stop = defaultdict(lambda: 0)
        self._enqueued = defaultdict(lambda: 0)
        self._dequeued = defaultdict(lambda: 0)
        self._time_func = time_func

        _shared_lock = multiprocessing.RLock()
        self._shared_state = _SharedState(
            lock=_shared_lock,
            empty=multiprocessing.Condition(lock=_shared_lock),
            messages=multiprocessing.Value('l', lock=False),
        )

        self._collectors = dict()

        if monitor_thread:
            self._monitor_th = _ActorSystemMonitorThread(threading.current_thread(), self)
            self._monitor_th.start()
        else:
            self._monitor_th = None

    def add_collector(self, name: str):
        """Add a mailbox that is not constrained by the queue maxsize and is not counted in the pending messages."""
        if self._started:
            raise RuntimeError('Already started')
        if name in self._mailboxes:
            raise ValueError(f'name "{name}" already present')
        collector = _CollectorActor()
        self.add_thread(name, collector)
        self._collectors[name] = collector

    def add_mailbox(self, name: str, mailbox: Optional[MailBox] = None):
        """Add a mailbox. It's safer to use `add_collector` to prevent deadlocks from full queue or pending msgs."""
        if self._started:
            raise RuntimeError('Already started')
        if name in self._mailboxes:
            raise ValueError(f'name "{name}" already present')
        if mailbox is None:
            if self._multiprocessing:
                mailbox = multiprocessing.JoinableQueue(maxsize=self._maxsize)
            else:
                mailbox = queue.Queue(maxsize=self._maxsize)
        self._mailboxes[name] = mailbox
        return self

    def _add(self,
             inbox_name: str,
             actor: Actor,
             executor_class: type,
             pool: int,
             mailbox: Optional[MailBox] = None):
        if self._started:
            raise RuntimeError('Already started')
        if not issubclass(executor_class, (threading.Thread, multiprocessing.Process)):
            raise ValueError(executor_class)
        if issubclass(executor_class, multiprocessing.Process) and not self._multiprocessing:
            raise RuntimeError('use_multiprocessing was False')
        if pool <= 0:
            pool = multiprocessing.cpu_count()
        assert pool > 0, pool
        if inbox_name in self._actor_names:
            for _actor in self._actors:
                if _actor is actor and _actor.name != inbox_name:
                    raise ValueError('The same actor instance was already added with a different inbox name')
        if inbox_name in self._collectors:
            raise ValueError(f'Mailbox name "{inbox_name}" is already used for a collector')
        if inbox_name in self._mailboxes and mailbox is not None and mailbox is not self._mailboxes[inbox_name]:
            raise RuntimeError('Another mailbox was already associated with the given inbox name')

        for i in range(pool):
            _struct = _ActorStruct(inbox_name, actor, executor_class)
            self._actors.append(_struct)
        self._actor_names.add(inbox_name)
        self._actor_instances.add(actor)
        if inbox_name not in self._mailboxes:
            self.add_mailbox(inbox_name, mailbox=mailbox)
        return self

    def add_thread(self, inbox_name: str, actor: Actor, pool: int = 1):
        """Add the given actor (with given inbox_name) to the system, executing it in a new pool of threads."""
        return self._add(inbox_name, actor, threading.Thread, pool)

    def add_process(self, inbox_name: str, actor: Actor, pool: int = 1):
        """Add the given actor (with given inbox_name) to the system, executing it in a new pool of processes."""
        return self._add(inbox_name, actor, multiprocessing.Process, pool)

    def start(self):
        """Start the actor executors"""
        if self._started:
            raise RuntimeError('Already started')
        self._time_start = self._time_func()
        mailboxes = self._mailboxes.copy()
        for _actor in self._actors:
            _actor.actor.init_routing(_actor.name, mailboxes, self._shared_state)
            if issubclass(_actor.executor_class, multiprocessing.Process):
                _actor.executor = _actor.executor_class(
                    target=_multiprocessing_actor_run, args=(_actor.actor,))
            else:
                _actor.executor = _actor.executor_class(
                    target=_actor.actor.run)
        self._started = True
        for _actor in self._actors:
            _actor.executor.start()

    def stop(self):
        """Try to stop the actor executors as soon as possible (possibly after they finish processing one message)."""
        for _a in self._actor_instances:
            _a.set_stop_flag()

        self.cancel_pending()  # Let's facilitate the stop

        for _actor in self._actors:
            # NOTE: the following is a notification that is only relevant if the queue is empty => block=False
            try:
                self._mailboxes[_actor.name].put(_ActorStop(), block=False)
            except queue.Full:
                pass
            else:
                self._enqueued_stop[_actor.name] += 1

        for _actor in self._actors:
            _actor.executor.join()
        self.cancel_pending()  # You don't want to leave a stopped system with pending messages

    def cancel_pending(self):
        """Discards the messages currently enqueued. NOTE: pending count is not guaranteed to be 0 after this call."""
        for q in self._mailboxes.values():
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    break
                else:
                    if not isinstance(msg, (_ActorStop, _ActorFinish)):
                        self._shared_state.decrement_messages(n=1)
                    q.task_done()

    def join_pending(self):
        """Wait completion of the messages that are pending in the system. (You may then safely stop)."""
        if not self._started:
            raise RuntimeError('Not started')
        poll = 5.  # Just for safety
        with self._shared_state.lock:
            while self.pending > 0:
                if self.is_degraded:
                    raise RuntimeError("Trying to wait for processing of messages pending in a degraded system.")
                self._shared_state.empty.wait_for(
                    lambda: self.pending <= 0 or self.is_degraded, timeout=poll)

    def finish(self):
        """Gracefully ask the actors to only finish processing the pending messages.
        One actor for each mailbox will be notified, so don't rely on this method to notify a whole actors pool.
        """
        for _a in self._actor_instances:
            _a.set_finish_flag()
        for _actor in self._actors:
            # NOTE: the following is a notification that is only relevant if the queue is empty => block=False
            try:
                self._mailboxes[_actor.name].put(_ActorFinish(), block=False)
            except queue.Full:
                pass

    def __enter__(self):
        """Used for context management: see __exit__"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Join pending msgs and actor executors.

        The system must be either stopped or finished processing the messages.
        """
        if self._started:
            if exc_type is None:
                self.join_pending()
            self.stop()

    def tell(self, mailbox: str, msg: Any, *,
             block=True, timeout=5.) -> None:
        """Send a message (to an actor). Equivalent to :func:`Actor.tell`."""
        self._shared_state.increment_messages()
        self._mailboxes[mailbox].put(msg, block=block, timeout=timeout)
        self._enqueued[mailbox] += 1

    def ask(self, mailbox: str, *,
            block: bool = True, timeout: Optional[float] = None) -> Any:
        """Wait for a message on the specified mailbox, ignoring system stop messages."""
        if not self._started:
            raise RuntimeError('Not started')
        if block and timeout is None and self.is_degraded:
            try:
                return self.ask(mailbox, block=False)
            except queue.Empty:
                raise RuntimeError('Trying to indefinitely wait for a message while in degraded state.')

        if mailbox in self._collectors:
            msg = self._collectors[mailbox].store.get(block=block, timeout=timeout)
            assert not isinstance(msg, (_ActorStop, _ActorFinish))  # should not be enqueued by CollectorActor
            self._dequeued[mailbox] += 1
            self._collectors[mailbox].store.task_done()
            return msg

        msg = self._mailboxes[mailbox].get(block=block, timeout=timeout)
        if isinstance(msg, _ActorStop):
            raise RuntimeError('Where does this system stop message come from?')
        if isinstance(msg, _ActorFinish):
            # ignore it and retry
            return self.ask(mailbox, block=block, timeout=timeout)
        self._dequeued[mailbox] += 1
        self._shared_state.decrement_messages(n=1)
        self._mailboxes[mailbox].task_done()
        return msg

    def ask_available(self, mailbox: str) -> Iterator[Any]:
        """Yield messages until no more are immediately available."""
        while True:
            try:
                yield self.ask(mailbox, block=False)
            except queue.Empty:
                return

    @property
    def elapsed(self) -> float:
        """Time passed since system start (calculated using time_func specified during initialization of the system)"""
        if not self._started:
            raise RuntimeError('Not started')
        return self._time_func() - self._time_start

    @property
    def working(self) -> int:
        """Approximate (due to race conditions) number of actors currently processing some message in the system"""
        return sum(_a.get_perf_counters().concurrent for _a in self._actor_instances)

    @property
    def is_alive(self) -> bool:
        """True if any of the actor executors is still alive; False if all executors have terminated."""
        if not self._started:
            return False
        return any(_a.executor.is_alive() for _a in self._actors)

    @property
    def is_degraded(self) -> bool:
        """True if started and any of the actor executors is NOT alive; False otherwise if not started or all alive."""
        if not self._started:
            raise False
        return any(not _a.executor.is_alive() for _a in self._actors)

    @property
    def pending(self) -> int:
        """Gives an approximate (due to race conditions) and unreliable estimate of the pending msgs in the system"""
        return self._shared_state.messages.value

    @property
    def queue_sizes(self) -> Dict[str, int]:
        """Approximate (due to race conditions) total number of messages in the queues (unreliable!)
        including the system notification messages for stop and finish."""
        return {name: q.qsize() for name, q in self._mailboxes.items()}

    @property
    def status(self) -> str:
        """
        Describe the status of the systems and its performance so far.

        NOTE: There's a race condition in the access to the queue sizes and actor stats.
              Also, the number of received messages is used in place of the number of completions.
              Results are approximate unless the system is stopped.
        """
        perf_counters: Dict[Actor, ActorStats] = {_a: _a.get_perf_counters() for _a in self._actor_instances}
        pending = self.pending
        queue_sizes = {name: max(0, q.qsize() - self._enqueued_stop[name]) for name, q in self._mailboxes.items()}
        elapsed = self.elapsed
        # NOTE: status data is collected in the above lines to limit the lines of code in the race condition.
        queues_info = ''
        for _name, _size in queue_sizes.items():
            _working = sum(perf_counters[_a.actor].concurrent for _a in self._actors if _a.name == _name)
            queues_info += f' {_name!r}:{_size + _working:3d}'
            _c_i = sum(perf_counters[_a.actor].received for _a in self._actors if _a.name == _name)
            _b_i = sum(perf_counters[_a.actor].busy_time for _a in self._actors if _a.name == _name)
            if _c_i != 0:
                _r_i = _b_i / _c_i
                queues_info += f'*{format_short_time(_r_i):5s}'
            else:
                queues_info += ' '*6
        utilization = sum(_pc.busy_time for _pc in perf_counters.values()) / elapsed
        return f'pending: {pending:3d} {{{queues_info.lstrip()}}} @{utilization:.1f}'

    def __repr__(self):
        return (f'<{self.__class__.__module__}.{self.__class__.__name__}: '
                f'{len(self._actors)} actors, '
                f'{self.pending:d} pending msgs, '
                f'{"" if self._started else "not started, "}'
                f'at 0x{id(self):x}>')


class _ActorSystemMonitorThread(threading.Thread):
    def __init__(self, monitored_th: threading.Thread, actor_system: 'ActorSystem') -> None:
        super().__init__(daemon=False)
        self.__monitored_th = monitored_th
        self.__actor_system = actor_system

    def run(self) -> None:
        assert self.__monitored_th is not threading.current_thread()
        self.__monitored_th.join()
        logger = logging.getLogger(__name__)
        s = self.__actor_system

        s.finish()
        if s.pending:
            logger.info(
                f'{s.__class__.__module__}.{s.__class__.__name__} '
                f'at 0x{id(self):x} '
                f'waiting   {self.__actor_system.status}')
        s.join_pending()
        s.stop()
        logger.info(
            f'{s.__class__.__module__}.{s.__class__.__name__} '
            f'at 0x{id(self):x} '
            f'finished ({self.__actor_system.status})')


def format_short_time(t: float) -> str:
    """
    Given a positive time duration in seconds, produce a string like 999ns, 9.9µs, 999µs, 9.9ms, 999ms, 9.9s or 9999s.

    >>> # noinspection PyUnresolvedReferences
    >>> print('    '.join(f'{format_short_time(_t):6s}' for _t in (
    >>>     eval(f'{m}e{e}') for e in range(-8, 3) for m in [x/10. for x in range(1, 10, 2)]+[0.99, 0.999, 0.99999])))
    """
    # NOTE: floating point approximation errors may lead to longer strings like 1000ns, 10.0µs, 1000µs, 10.0ms, ...
    if t <= 999.5e-9:
        return f'{t * 1e9:.0f}ns'
    if t <= 9.95e-6:
        return f'{t * 1e6:.1f}µs'
    if t <= 999.5e-6:
        return f'{t * 1e6:.0f}µs'
    if t <= 9.95e-3:
        return f'{t * 1000:.1f}ms'
    if t <= 999.5e-3:
        return f'{t * 1000:.0f}ms'
    if t <= 9.95:
        return f'{t:.1f}s'
    return f'{t:.0f}s'
