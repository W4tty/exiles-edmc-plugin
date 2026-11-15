"""The plugin's own read of the game's journal: the net under EDMC's hand-over (0.7.1).

EDMC reads the journal and hands each line to its plugins, and that stays the first way in: those entries come at
once and carry what EDMC fills in from the game's side files. But a line can fail to arrive. EDMC's ship export stops
a Loadout line before any plugin sees it when its output folder is unset; a line EDMC trips over is dropped; what the
game wrote before EDMC started is read for EDMC's own state only; and an EDMC started minimised under a dark theme
reads no journal at all for that run. None of that is the pilot's doing, and none of it shows.

So this reads the newest journal file itself, a couple of seconds behind the game. A line is held for GRACE seconds:
one EDMC hands over in that time is dropped here, one it does not is sent as the game wrote it. A line already older
than OLD_AFTER when it is read (EDMC was not running then, or this is the first look at the file) is not waited for.
The squad's database keeps a line once however often it is sent, so a line that goes both ways is no harm.

It reads the Status file too, whenever EDMC has stopped handing that over, and it keeps the plain facts the journal
states (the commander, the system, the station, whether a session is on) for the core to fall back on while EDMC says
nothing. Chat, the friends list and the music cues are dropped as they are read, as everywhere else in the plugin.

Where the read stands is kept in the spool folder (KEPT), so a restart of EDMC carries on where the last run stopped
and does not send the session again. No tkinter and no network in this file: what it needs is handed in."""
from __future__ import annotations

import calendar
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Optional

POLL = 2.0                  # seconds between looks at the journal file
GRACE = 20.0                # how long EDMC has to hand a line over before it is sent from here
OLD_AFTER = 60.0            # a line this old when it is read was never going to be handed over now
HEADLESS = 600.0            # a line read before its file names the commander waits this long for the name
HANDED_KEEP = 300.0         # how long what EDMC handed over is remembered, for lines read late
HANDED_MOST = 20000         # and how many at most
SCAN_EVERY = 10.0           # how often the folder is looked through for a newer journal file
CHUNK = 262144              # bytes read at a time
ROUND_MOST = 4              # chunks a round, about a megabyte: a whole session read at a start never floods the sender
HEAD_MOST = 32 * 1048576    # how far back a file is looked through when the read starts partway into it
HEAD_FIRST = 1048576        # and how far into a session's first file, for the commander a later part does not name
SAVE_EVERY = 15.0           # how often the place the read has reached is written down
STATUS_LAG = 10.0           # the Status file this far ahead of what EDMC last handed over: EDMC has gone quiet on it
STATUS_LIVE = 120.0         # a Status file older than this belongs to a session that is over
SAY_EVERY = 300.0           # the same trouble is logged once in this long
KEPT = "journal.at"
# The mark on a line sent from here: EDMC handed it to no plugin, so it reached nobody else through EDMC either. The
# squad's database sets the mark aside when it tells lines apart, and treats such a line as one it had to fetch itself.
READ_KEY = "ExilesRead"
READ_FROM_FILE = "file"
STATUS_FILE = "Status.json"
# The game's names for its journal files, old and new: Journal.220315152335.01.log, Journal.2022-03-15T152503.01.log.
# The last number is the part: a session that outgrows one file carries on in part 02 with no new login.
JOURNAL_NAME = re.compile(r"^Journal(Alpha|Beta)?\.[0-9]{2,4}-?[0-9]{2}-?[0-9]{2}T?[0-9]{6}\.([0-9]{2})\.log$")
# the kinds the picture is made of, as they stand in a line the game wrote
FACTS = re.compile(rb'"event":"(Fileheader|Commander|LoadGame|Location|FSDJump|CarrierJump|Docked|Undocked|Shutdown)"')
TEST_BUILDS = ("alpha", "beta")

_epochs: dict = {}


def epoch(stamp: Any) -> Optional[float]:
    """A journal time (2026-11-15T16:13:46Z) as seconds since 1970, or None when it is no such time."""
    if not isinstance(stamp, str):
        return None
    got = _epochs.get(stamp)
    if got is None:
        try:
            got = float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
        except ValueError:
            return None
        if len(_epochs) > 512:
            _epochs.clear()
        _epochs[stamp] = got
    return got


def mark(entry: dict) -> frozenset:
    """What tells one entry from another: a short digest of each thing it says, in whatever order they come. Two
    entries are the same when their marks are equal, and alike by how much of their marks they share."""
    return frozenset(
        hashlib.blake2b((str(key) + "\0" + json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)).encode("utf-8"),
                        digest_size=8).digest()
        for key, value in entry.items())


class _File:
    """One journal file being read: its commander and whether it is a test build's, shared by every line read from it."""
    __slots__ = ("path", "order", "part", "cmdr", "beta")

    def __init__(self, path: Path, order: int, cmdr: Optional[str] = None):
        self.path = path
        self.order = order
        found = JOURNAL_NAME.match(path.name)
        self.part = int(found.group(2)) if found else 1
        self.cmdr = cmdr
        self.beta = bool(found and found.group(1))


class _Line:
    """One line read and not yet settled."""
    __slots__ = ("file", "at", "stamp", "event", "mark", "entry", "read", "old")

    def __init__(self, file: _File, at: int, stamp: str, event: str, entry: dict, read: float, old: bool):
        self.file = file
        self.at = at
        self.stamp = stamp
        self.event = event
        self.entry = entry
        self.mark = mark(entry)
        self.read = read
        self.old = old


def fresh_picture() -> dict:
    return {"cmdr": None, "system": None, "station": None, "docked": None, "session": None,
            "version": None, "build": None, "odyssey": None}


class JournalReader:
    """`folder()` gives the journal's folder or None. `recover(entry, commander, live)` takes a line EDMC did not hand
    over; `status(entry)` a Status entry EDMC did not; `picture(facts)` what the journal last said, once the read has
    caught up with a game in play. `active()` is False while nothing may be gathered (no pairing), `room()` False while
    the sender holds enough. `keep_dir` is where the place reached is written. `clock` counts seconds, `wall` is the
    time of day as seconds since 1970."""

    def __init__(self, *, folder: Callable[[], Optional[Path]], recover: Callable[[dict, Optional[str], bool], None],
                 keep_dir: Path, log: logging.Logger, status: Optional[Callable[[dict], None]] = None,
                 picture: Optional[Callable[[dict], None]] = None, active: Optional[Callable[[], bool]] = None,
                 room: Optional[Callable[[], bool]] = None, never: frozenset = frozenset(),
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self._folder = folder
        self._recover = recover
        self._status = status
        self._picture = picture
        self._active = active or (lambda: True)
        self._room = room or (lambda: True)
        self._keep_dir = Path(keep_dir)
        self._log = log
        self._never = never
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file: Optional[_File] = None
        self._next: Optional[Path] = None           # a newer file, taken up once this one is read to its end
        self._files = 0
        self._offset = 0
        self._eof = False
        self._live = False                          # the newest line read was written just now: the game is in play
        self._last_stamp: Optional[str] = None
        self._from_end = False                      # a pairing was made: what was played before it is not sent
        self._scanned: Optional[float] = None
        self._pending: deque = deque()
        self._handed: dict = {}                     # (kind, time) -> the marks of what EDMC handed over
        self._handed_log: deque = deque()           # (clock, kind, time, the mark's hash), oldest first
        self._seen = fresh_picture()
        self._told: Optional[dict] = None
        self._status_edmc: Optional[float] = None   # the time on the newest Status entry EDMC handed over
        self._status_ahead: Optional[float] = None  # the clock when the Status file was first seen ahead of that
        self._status_read: Optional[tuple] = None   # what the Status file looked like when it was last dealt with
        self._kept_was: Optional[tuple] = None
        self._kept_at: Optional[float] = None
        self._said: dict = {}
        self.counts = {"read": 0, "handed": 0, "recovered": 0, "status": 0}

    # ----- EDMC's side, on its main thread --------------------------------------------------------------------------
    def handed(self, entry: dict) -> None:
        """EDMC handed this entry over: the line it came from needs no sending from here. Whatever goes wrong in
        here stays in here: this runs on EDMC's own thread, ahead of the entry being sent."""
        try:
            self._note_handed(entry)
        except Exception as e:
            self._say(f"journal: an entry EDMC handed over could not be noted ({e.__class__.__name__}: {e})")

    def _note_handed(self, entry: dict) -> None:
        name, stamp = entry.get("event"), entry.get("timestamp")
        if not isinstance(name, str) or not isinstance(stamp, str) or name in self._never:
            return
        try:
            what = mark(entry)
        except (TypeError, ValueError):
            what = frozenset()
        now = self._clock()
        with self._lock:
            self._handed.setdefault((name, stamp), []).append(what)
            self._handed_log.append((now, name, stamp, hash(what)))
            self.counts["handed"] += 1
            if len(self._handed_log) > HANDED_MOST:          # the worker is not keeping up: this never grows without end
                self._drop_old(now)

    def status_handed(self, entry: dict) -> None:
        """EDMC handed this Status entry over: the newest of them says how far its own read of that file has come."""
        try:
            at = epoch(entry.get("timestamp"))
        except Exception:
            at = None
        if at is not None and (self._status_edmc is None or at > self._status_edmc):
            self._status_edmc = at
        self._status_ahead = None

    # ----- the worker -----------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="EDMC_Exiles journal", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """EDMC is closing: the thread ends, and what still waits is settled at once so nothing is left behind."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self._thread is None or not self._thread.is_alive():
            self._round(final=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._round()
            self._stop.wait(POLL)

    def _round(self, final: bool = False) -> None:
        try:
            self.poll(final=final)
        except Exception as e:      # the read must never die, and never reach EDMC
            self._say(f"journal: {e.__class__.__name__}: {e}")

    def _say(self, words: str) -> None:
        """A line of trouble for the log, the same one no oftener than SAY_EVERY."""
        now = self._clock()
        if now - self._said.get(words, -SAY_EVERY) >= SAY_EVERY:
            if len(self._said) > 50:
                self._said.clear()
            self._said[words] = now
            self._log.warning(words)

    def poll(self, final: bool = False) -> None:
        """One round: follow the newest file, read what is new, settle what has waited long enough, look at the Status
        file, tell the core what the journal says, and note the place reached. `final` waits for nothing."""
        now = self._clock()
        if not self._active():
            self._idle()
            return
        self._follow(now)
        self._read(now)
        self._settle(now, final)
        self._forget(now)
        if not final:
            self._look_at_status()
            self._tell()
        self._keep(now, final)

    def _idle(self) -> None:
        """Not paired: nothing is read and nothing is held. Once a pairing is made the read starts at the file's end:
        what was played before it is not this plugin's to send."""
        with self._lock:
            self._pending.clear()
            self._handed.clear()
            self._handed_log.clear()
        self._file, self._next, self._scanned, self._from_end = None, None, None, True

    # ----- which file -------------------------------------------------------------------------------------------------
    def _newest(self) -> Optional[Path]:
        """The journal file the game made last, as EDMC picks it."""
        try:
            folder = self._folder()
        except Exception:
            folder = None
        if not folder:
            return None
        best, best_key = None, None
        try:
            with os.scandir(folder) as found:
                for f in found:
                    if not JOURNAL_NAME.match(f.name):
                        continue
                    try:
                        st = f.stat()
                    except OSError:
                        continue
                    key = (getattr(st, "st_birthtime", None) or st.st_ctime, f.name)
                    if best_key is None or key > best_key:
                        best, best_key = Path(f.path), key
        except OSError:
            return None
        return best

    def _follow(self, now: float) -> None:
        if self._scanned is not None and now - self._scanned < SCAN_EVERY:
            return
        self._scanned = now
        newest = self._newest()
        if newest is None:
            self._from_end = False      # no journal yet: nothing was played before, so the first file is read whole
            return
        if self._file is None:
            self._open(newest, first=True)
        elif newest != self._file.path:
            self._next = newest

    def _open(self, path: Path, first: bool = False) -> None:
        """Start on a journal file: where the last run left it when it is the same file, at its end after a pairing,
        else from its top. A later part of the session being read keeps that session's commander."""
        before = self._file
        self._files += 1
        f = _File(path, self._files)
        if before is not None and f.part > 1 and not first:
            f.cmdr = before.cmdr
            seen = dict(self._seen)
        else:
            seen = fresh_picture()
        self._file, self._next, self._offset, self._eof, self._last_stamp, self._seen = f, None, 0, False, None, seen
        self._live = False
        if first:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            kept = self._kept()
            if self._from_end:
                self._offset = size
            elif kept.get("file") == path.name and isinstance(kept.get("at"), int) and 0 <= kept["at"] <= size:
                self._offset = kept["at"]
                self._kept_was = (path.name, self._offset)
            if f.part > 1:
                # a later part names no commander: the session's first file does, near its top
                self._look_back(f, path.with_name(re.sub(r"\.[0-9]{2}\.log$", ".01.log", path.name)), HEAD_FIRST)
            if self._offset:
                self._look_back(f, path, min(self._offset, HEAD_MOST))
        self._from_end = False

    def _look_back(self, f: _File, path: Path, upto: int) -> None:
        """The read starts partway into a session: what the lines before that point say of the commander, the game and
        where the pilot is, without sending any of them."""
        try:
            with open(path, "rb") as h:
                data = h.read(upto)
        except OSError:
            return
        for found in FACTS.finditer(data):
            start = data.rfind(b"\n", 0, found.start()) + 1
            end = data.find(b"\n", found.end())
            try:
                entry = json.loads(data[start:end if end >= 0 else len(data)])
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("event"), str):
                self._note(f, entry["event"], entry)

    def _kept(self) -> dict:
        try:
            got = json.loads((self._keep_dir / KEPT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return got if isinstance(got, dict) else {}

    def _keep(self, now: float, final: bool = False) -> None:
        """Write down the place reached: the first line still waiting, else the end of what was read."""
        f = self._file
        if f is None:
            return
        with self._lock:
            at = min((line.at for line in self._pending if line.file is f), default=self._offset)
        where = (f.path.name, at)
        if where == self._kept_was or (not final and self._kept_at is not None and now - self._kept_at < SAVE_EVERY):
            return
        try:
            self._keep_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._keep_dir / (KEPT + ".tmp")
            tmp.write_text(json.dumps({"file": where[0], "at": where[1]}), encoding="utf-8")
            tmp.replace(self._keep_dir / KEPT)
            self._kept_was, self._kept_at = where, now
        except OSError as e:
            self._say(f"journal: the place reached could not be written down ({e.__class__.__name__})")

    # ----- reading ----------------------------------------------------------------------------------------------------
    def _read(self, now: float) -> None:
        for _ in range(ROUND_MOST):
            f = self._file
            if f is None or not self._room():
                return
            try:
                size = f.path.stat().st_size
            except OSError:
                self._file, self._scanned = None, None       # the file is gone: look for the newest again
                return
            if size < self._offset:
                self._offset = 0                             # a shorter file under the same name: read it from its top
            if size == self._offset:
                self._eof = True
                if self._next is not None:
                    self._open(self._next)
                    continue
                return
            try:
                with open(f.path, "rb") as h:
                    h.seek(self._offset)
                    data = h.read(min(CHUNK, size - self._offset))
            except OSError:
                return
            cut = data.rfind(b"\n")
            if cut < 0:
                if len(data) < CHUNK:
                    self._eof = True                         # a line the game is still writing: next round
                    return
                self._offset += len(data)                    # no line is this long: not a journal's text, passed over
                continue
            at = self._offset
            wall = self._wall()
            for raw in data[:cut].split(b"\n"):
                self._line(f, at, raw, now, wall)
                at += len(raw) + 1
            self._offset += cut + 1
            self._eof = self._offset >= size

    def _line(self, f: _File, at: int, raw: bytes, now: float, wall: float) -> None:
        raw = raw.strip()
        if not raw:
            return
        try:
            entry = json.loads(raw)
        except ValueError:
            return
        if not isinstance(entry, dict) or not isinstance(entry.get("event"), str):
            return
        name, stamp = entry["event"], entry.get("timestamp")
        self.counts["read"] += 1
        named = f.cmdr
        self._note(f, name, entry)
        when = epoch(stamp)
        if when is None:
            return
        old = wall - when > OLD_AFTER
        self._live = not old
        self._last_stamp = stamp
        if f.cmdr and not named:
            # the file has just named its commander: EDMC hands over what it held back for the same reason about now
            with self._lock:
                for line in self._pending:
                    if line.file is f and not line.old:
                        line.read = now
        if f.beta or name in self._never:
            return
        line = _Line(f, at, stamp, name, entry, now, old)
        with self._lock:
            self._pending.append(line)

    def _note(self, f: _File, name: str, e: dict) -> None:
        """The plain facts a line states, kept for the core to fall back on."""
        s = self._seen
        if name == "Fileheader":
            version = str(e.get("gameversion") or "")
            f.beta = f.beta or any(word in version.lower() for word in TEST_BUILDS)
            s.update(version=version or s["version"], build=e.get("build") or s["build"])
            if e.get("Odyssey") is not None:
                s["odyssey"] = bool(e.get("Odyssey"))
        elif name == "Commander":
            if isinstance(e.get("Name"), str) and e["Name"]:
                f.cmdr = s["cmdr"] = e["Name"]
        elif name == "LoadGame":
            if isinstance(e.get("Commander"), str) and e["Commander"]:
                f.cmdr = s["cmdr"] = e["Commander"]
            s["session"] = True
            for key, field in (("version", "gameversion"), ("build", "build")):
                if e.get(field):
                    s[key] = e[field]
            if e.get("Odyssey") is not None:
                s["odyssey"] = bool(e.get("Odyssey"))
        elif name == "Shutdown":
            s["session"] = False
        elif name in ("Location", "FSDJump", "CarrierJump"):
            docked = bool(e.get("Docked")) if name != "FSDJump" else False
            s.update(system=e.get("StarSystem") or s["system"], docked=docked, station=e.get("StationName") if docked else None)
        elif name == "Docked":
            s.update(system=e.get("StarSystem") or s["system"], docked=True, station=e.get("StationName"))
        elif name == "Undocked":
            s.update(docked=False, station=None)

    # ----- settling ---------------------------------------------------------------------------------------------------
    def _ripe(self, group: list, now: float) -> bool:
        """Whether one second's lines have waited long enough. Old lines are settled as soon as the whole second is
        read (a later one has come, or the file's end); new ones GRACE after the last of them was read. Lines of a
        file that has not named its commander wait HEADLESS for the name."""
        newest = max(line.read for line in group)
        first = group[0]
        if first.file.cmdr is None and now - newest < HEADLESS:
            return False
        if first.old:
            return first.stamp != self._last_stamp or self._eof or first.file is not self._file
        return now - newest >= GRACE

    def _settle(self, now: float, final: bool = False) -> None:
        """Settle every second's lines that have waited long enough; the rest stay, and hold nothing behind them up."""
        missed: list = []
        with self._lock:
            lines = list(self._pending)
            stay: deque = deque()
            i = 0
            while i < len(lines):
                j = i + 1
                while j < len(lines) and lines[j].stamp == lines[i].stamp and lines[j].file is lines[i].file:
                    j += 1
                group = lines[i:j]
                if final or self._ripe(group, now):
                    missed.extend(self._not_handed(group))
                else:
                    stay.extend(group)
                i = j
            self._pending = stay
        if not missed:
            return
        kinds: Counter = Counter()
        for line in missed:
            try:
                self._recover(line.entry, line.file.cmdr, not line.old)
            except Exception as e:
                self._say(f"journal: a {line.event} line could not be passed on ({e.__class__.__name__}: {e})")
                continue
            kinds[line.event] += 1
        sent = sum(kinds.values())
        if sent:
            self.counts["recovered"] += sent
            most = ", ".join(f"{kind} {n}" for kind, n in kinds.most_common(4))
            self._log.info(f"journal: {sent} line{'' if sent == 1 else 's'} EDMC did not hand over read from the journal "
                           f"and sent ({most}{', and more' if len(kinds) > 4 else ''})")

    def _not_handed(self, group: list) -> list:
        """The lines of one second EDMC has not handed over, in the file's order. A line is handed over when EDMC
        handed its very content. What is left then pairs off with what EDMC handed of the same kind and second with
        other content, the most alike first: EDMC fills some entries in from the game's side files, and such an entry
        still says nearly all its line says."""
        by_key: dict = {}
        for line in group:
            by_key.setdefault((line.event, line.stamp), []).append(line)
        out = []
        for key, lines in by_key.items():
            have = self._handed.get(key)
            if have:
                rest = []
                for line in lines:
                    if line.mark in have:
                        have.remove(line.mark)
                    else:
                        rest.append(line)
                while rest and have:
                    _, i, j = max((len(line.mark & what), -i, -j) for i, line in enumerate(rest) for j, what in enumerate(have))
                    rest.pop(-i)
                    have.pop(-j)
                if not have:
                    self._handed.pop(key, None)
                lines = rest
            out.extend(lines)
        out.sort(key=lambda line: (line.file.order, line.at))
        return out

    def _forget(self, now: float) -> None:
        """What EDMC handed over long ago is dropped: an entry EDMC makes up itself has no line to meet."""
        with self._lock:
            self._drop_old(now)

    def _drop_old(self, now: float) -> None:
        log = self._handed_log
        while log and (now - log[0][0] > HANDED_KEEP or len(log) > HANDED_MOST):
            _, name, stamp, what = log.popleft()
            have = self._handed.get((name, stamp))
            if have:
                for i, held in enumerate(have):
                    if hash(held) == what:
                        have.pop(i)
                        break
                if not have:
                    self._handed.pop((name, stamp), None)

    # ----- the Status file and the picture ----------------------------------------------------------------------------
    def _look_at_status(self) -> None:
        """The Status file, passed on from here only while EDMC says nothing of it: the file has been ahead of the last
        entry EDMC handed over for STATUS_LAG seconds, and what it holds is fresh enough to be of a game in play."""
        f = self._file
        if self._status is None or f is None or f.beta:
            return
        path = f.path.with_name(STATUS_FILE)
        try:
            st = path.stat()
        except OSError:
            return
        looks = (st.st_mtime_ns, st.st_size)
        if looks == self._status_read:
            return
        try:
            entry = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return                      # the game is writing it: next round
        at = epoch(entry.get("timestamp")) if isinstance(entry, dict) else None
        if at is None or self._wall() - at > STATUS_LIVE:
            self._status_read = looks
            return
        if self._status_edmc is not None and at <= self._status_edmc:
            self._status_read, self._status_ahead = looks, None      # EDMC has handed this one over, or a newer one
            return
        now = self._clock()
        if self._status_ahead is None:
            self._status_ahead = now
        if now - self._status_ahead < STATUS_LAG:
            return                      # EDMC may yet hand it over: looked at again next round
        self._status_read = looks
        self._status(entry)
        self.counts["status"] += 1

    def _tell(self) -> None:
        """Tell the core what the journal says, once the read has caught up with a game in play and it has changed."""
        if self._picture is None or not self._eof or not self._live or self._seen == self._told:
            return
        self._told = dict(self._seen)
        self._picture(dict(self._seen))
