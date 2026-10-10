"""P25 trunk following (VK3RQ, 2026-10-10): hear the calls a control channel grants.

A trunked system's control channel only announces calls; each goes out on a voice
channel of its own for its few seconds. The follower watches the grants the P25 decoder
reads, and for a call the app can play -- Phase 1, not encrypted, its frequency known
from the channel plan, its talkgroup not locked out -- tells the window to listen there;
when the voice stops (a terminator, or no voice frames for HANG_S), to come back to the
control channel. One call at a time: while it listens to a call the control channel is
not being read, so grants meanwhile are missed (P14's second receiver will lift that).

Decisions only; the window carries them out (how depends on the radio: a voice channel
inside the span is reached by the listening offset, without retuning). No Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

#: After the voice stops, how long to wait for more before going back.
HANG_S = 1.5
#: A call's first voice may take a moment to arrive after the jump.
GRACE_S = 2.5
#: Grants older than this when read are stale: the call may be over.
STALE_S = 1.0


def call_key(nac: int, call: dict) -> str:
    """How a call is named for lockouts: "NAC:TG 501" or "NAC:to 1234"."""
    who = (f"TG {call['group']}" if call.get("group") is not None
           else f"to {call.get('target', '?')}")
    return f"{nac:03X}:{who}"


def playable(call: dict) -> bool:
    return (call.get("phase") == "1" and call.get("encrypted") is not True
            and call.get("freq_hz") is not None)


@dataclass
class Following:
    key: str
    freq_hz: float
    started: float


class TrunkFollower:
    def __init__(self, lockouts=()) -> None:
        self.enabled = False
        self.lockouts: set[str] = set(lockouts)
        self.following: Following | None = None
        #: Why the last grant was not followed ("" when it was, or none came).
        self.note = ""

    def toggle_lockout(self, key: str) -> bool:
        """Lock a talkgroup out, or let it back in; True if now locked out."""
        if key in self.lockouts:
            self.lockouts.discard(key)
            return False
        self.lockouts.add(key)
        return True

    def step(self, now: float, grants, last_voice: float, last_terminator: float,
             reachable) -> tuple:
        """One look: `grants` are (time, NAC, call) since the last step; `reachable(hz)`
        says whether the radio can listen there now. Returns ("follow", Following),
        ("return",), or () for nothing to do."""
        if self.following is not None:
            f = self.following
            heard = max(last_voice, f.started)
            ended = last_terminator > f.started + 0.2
            quiet = now - heard > (HANG_S if last_voice > f.started else GRACE_S)
            if ended or quiet or not self.enabled:
                self.following = None
                return ("return",)
            return ()
        if not self.enabled:
            return ()
        for when, nac, call in reversed(list(grants)):        # the newest first
            if now - when > STALE_S:
                continue
            key = call_key(nac, call)
            if key in self.lockouts:
                self.note = f"{key.split(':', 1)[1]} locked out"
                continue
            if not playable(call):
                continue
            if not reachable(call["freq_hz"]):
                self.note = (f"{key.split(':', 1)[1]} on {call['freq_hz'] / 1e6:.5f} MHz is "
                             f"outside the span")
                continue
            self.following = Following(key, float(call["freq_hz"]), now)
            self.note = ""
            return ("follow", self.following)
        return ()

    def abandon(self) -> None:
        """Stop following without going back (the user tuned elsewhere)."""
        self.following = None
