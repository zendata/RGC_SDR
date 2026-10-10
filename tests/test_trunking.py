"""P25 trunk following's decisions (trunking.py)."""

from src.rgc_sdr.trunking import GRACE_S, HANG_S, TrunkFollower, call_key, playable


def _call(group=501, freq=420.1e6, phase="1", encrypted=False):
    return {"group": group, "freq_hz": freq, "phase": phase, "encrypted": encrypted}


def _follower():
    f = TrunkFollower()
    f.enabled = True
    return f


def anywhere(_hz):
    return True


def test_a_playable_grant_is_followed_and_the_end_brings_it_back():
    f = _follower()
    act = f.step(10.0, [(10.0, 0x123, _call())], 0.0, 0.0, anywhere)
    assert act[0] == "follow" and act[1].freq_hz == 420.1e6 and act[1].key == "123:TG 501"
    assert f.step(10.5, [], 10.4, 0.0, anywhere) == ()            # voice heard
    assert f.step(11.0, [], 10.9, 11.0, anywhere) == ("return",)  # terminator
    assert f.following is None


def test_silence_brings_it_back_after_the_hang():
    f = _follower()
    f.step(0.0, [(0.0, 1, _call())], 0.0, 0.0, anywhere)
    assert f.step(GRACE_S - 0.1, [], 0.0, 0.0, anywhere) == ()     # waiting for voice
    assert f.step(GRACE_S + 0.1, [], 0.0, 0.0, anywhere) == ("return",)
    f.step(10.0, [(10.0, 1, _call())], 0.0, 0.0, anywhere)
    assert f.step(10.0 + HANG_S + 0.5, [], 10.3, 0.0, anywhere) == ("return",)


def test_what_cannot_be_heard_is_not_followed():
    f = _follower()
    grants = [(0.0, 1, _call(phase="2")), (0.0, 1, _call(encrypted=True)),
              (0.0, 1, _call(freq=None)), (0.0, 1, _call(phase="?"))]
    assert f.step(0.0, grants, 0.0, 0.0, anywhere) == ()
    assert f.step(5.0, [(1.0, 1, _call())], 0.0, 0.0, anywhere) == ()   # stale
    assert f.step(5.0, [(5.0, 1, _call())], 0.0, 0.0, lambda hz: False) == ()
    assert "outside the span" in f.note
    assert not playable(_call(phase="2")) and playable(_call())


def test_lockouts_and_the_switch():
    f = _follower()
    assert f.toggle_lockout(call_key(1, _call())) is True
    assert f.step(0.0, [(0.0, 1, _call())], 0.0, 0.0, anywhere) == ()
    assert "locked out" in f.note
    assert f.toggle_lockout("001:TG 501") is False
    f.enabled = False
    assert f.step(1.0, [(1.0, 1, _call())], 0.0, 0.0, anywhere) == ()
    f.enabled = True
    f.step(2.0, [(2.0, 1, _call())], 0.0, 0.0, anywhere)
    f.enabled = False
    assert f.step(2.1, [], 2.05, 0.0, anywhere) == ("return",)
