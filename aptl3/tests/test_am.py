import itertools
import time
from typing import Any

from ..am import ActorSystem, MapActor, Actor


def double(x):
    return 2 * x


def test_empty_system():
    with ActorSystem(maxsize=0, use_multiprocessing=False) as s:
        s.start()
        print(s.status)
    print(s.status)


def test_simple_actor():
    with ActorSystem(maxsize=0, use_multiprocessing=False) as s:
        s.add_thread('str', MapActor(str, 'main'))
        s.add_mailbox('main')
        s.start()
        s.tell('str', 42)
        out = s.ask('main')
        assert out == '42'
        print(s.status)
    print(s.status)


def test_mixed_actors():
    n = 100
    with ActorSystem(maxsize=n, use_multiprocessing=True) as s:
        s.add_process("double", MapActor(double, "str"), pool=2)
        s.add_thread("str", MapActor(str, "main"), pool=2)
        s.add_mailbox("main")
        s.start()
        for _ in range(3):
            for __ in range(n):
                s.tell("double", 21)
            out = [s.ask("main") for _ in range(n)]
            assert out == ["42"] * n, out
    assert s.pending == 0


def test_actor_system_exit():
    n = 100
    with ActorSystem(maxsize=0) as s:
        s.add_thread("str", MapActor(str, "sleep"))
        s.add_thread("sleep", MapActor(lambda x: [x, time.sleep(0.01)][0], "main"), pool=10)
        s.add_collector("main")
        s.start()
        for _ in range(n):
            s.tell("str", 42)

    assert not s.is_alive, repr(s)
    assert s.pending == 0, s.status
    back = 0
    for got in s.ask_available('main'):
        assert got == "42"
        back += 1
    assert back == n, (back, n, s.status)
    print(s.status)


def test_actor_system_stop_exit():
    delay = 0.01
    n = 100
    _tic = time.perf_counter()
    with ActorSystem(maxsize=0) as s:
        s.add_thread("sleep", MapActor(time.sleep, "main"))
        s.add_collector("main")
        s.start()
        for _ in range(n):
            s.tell("sleep", delay)
        assert s.ask("main") is None  # will wait 1 * delay
        s.stop()
        print(s.status)
        print()  # this will appear in the backtrace of s.__exit__
        # will wait 1 * delay while finishing the job that was started before s.stop()
    assert s.pending == 0, s.status
    # 'main' collector is expected to contain 0 or 1 msg at this point.
    print(s.status)
    _elapsed = time.perf_counter() - _tic
    print(f'elapsed: {_elapsed*1000:.0f}ms')
    assert _elapsed < 10 * delay, _elapsed  # NOTE: 2 * delay is expected; 4 * delay should be enough on a slow system.


def test_actor_system_join_pending():
    for test_i in range(100):
        n = 100
        with ActorSystem(maxsize=0) as s:
            s.add_thread('input', MapActor(str, 'main'))
            s.add_collector('main')
            s.start()
            for i in range(n):
                s.tell('input', i)
            if test_i % 2 == 0:
                s.finish()  # It should have no effect. Let's skip it in half of the tests.
            s.join_pending()
            print('before stop', s.status)
            s.stop()

        print('after exit ', s.status)
        assert list(map(s.ask, itertools.repeat('main', n))) == list(map(str, range(n)))


def test_actor_system_request_finish_batch():
    n = 100

    class BatchingActor(Actor):
        def receive(self, msg: Any):
            _batch = list(itertools.chain(
                [msg],
                self.ask_available(None, block=not self.finish_requested),
            ))
            return 'main', _batch

    with ActorSystem(maxsize=0) as s:
        s.add_thread('batch', BatchingActor())
        s.add_collector('main')
        s.start()
        for i in range(n):
            s.tell('batch', i)
        print('before finish:', s.status)
        s.finish()
        print('before exit:', s.status)
        print()
    print('exited:', s.status)

    num_batches = 0
    concatenated = []
    while len(concatenated) < n:
        batch = s.ask('main')
        print('batch:', batch)
        assert 1 <= len(batch) <= n, batch
        concatenated.extend(batch)
        num_batches += 1
    assert concatenated == list(range(n)), concatenated
    assert num_batches == 1  # finish is requested after all `tell`


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # fork is default on unix, but windows uses spawn, which pickles actors.
    test_empty_system()
    test_simple_actor()
    test_actor_system_request_finish_batch()
    test_actor_system_join_pending()
    test_actor_system_stop_exit()
    test_actor_system_exit()
    test_mixed_actors()
