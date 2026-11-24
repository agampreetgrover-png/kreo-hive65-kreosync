"""KreoSync: screen-reactive ambient lighting for the Kreo Hive 65 (USB 258a:010c).

It pulls a palette of the vivid colors on screen and animates them evenly across the
keyboard. Talks to the board directly
over USB HID (OpenRGB doesn't support it). Protocol source:
https://github.com/Arman-Bisht/kreo-hive65-rgb-suite/blob/main/PROTOCOL.md

Run it, then open the control panel at http://127.0.0.1:8765 (settings save to settings.json,
problems are written to kreosync.log).
"""
import collections
import colorsys
import json
import logging
import math
import os
import random
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler

import hid
import mss
import numpy as np

VID, PID = 0x258A, 0x010C
HEADER = bytes.fromhex("0608000001007a01")  # report 6, cmd 0x08 = direct per-LED frame, 126 slots
PKT_LEN, SLOTS = 520, 126
ROWS, COLS = 5, 16
SLOT = np.arange(COLS)[None, :] * 6 + np.arange(ROWS)[:, None] + 1  # grid cell -> LED slot (column-major)
# float64 on purpose: effect time grows for days, and float32 phases would freeze the animation
X, Y = np.meshgrid(np.arange(COLS, dtype=np.float64), np.arange(ROWS, dtype=np.float64))
WHITE = np.full(3, 255.0)
FPS = 30
PALETTE_HZ = 10  # screen capture is the expensive part; colors don't need 30 Hz
MIRROR_HZ = 20   # mirror follows motion on screen, so it samples faster
MIRROR_RISE, MIRROR_FALL = 0.55, 0.14  # per-frame fade: flashes light keys at once, then glow out
MIRROR_SAT = 1.3  # averaging a patch of screen washes its color out; win some back
BLACK = 16        # screen levels at or below this count as black
SMOOTH = 0.25     # per-LED fade toward each new frame (lower = calmer)
GAMMA = 1.8       # full-intensity colors only gain saturation; dimmed parts (rain's glow, tails) fall off like LEDs should
CAPTURE_RETRY_S = 1.0    # after a failed grab (lock screen, UAC prompt), wait this long before trying again
DISPLAY_REFRESH_S = 5.0  # re-read the display layout this often: resolution, scaling and plug changes

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "settings.json")
LOG_FILE = os.path.join(HERE, "kreosync.log")
PORT = 8765
LIMITS = {"speed": (0.1, 3.0), "brightness": (0.05, 1.0), "floor": (0.0, 0.4)}
MAX_COLORS = 6
MAX_MONITORS = 16  # a saved display index is kept up to this even while that display is unplugged
RIPPLE_SPEED, RIPPLE_LIFE = 9.0, 1.2  # keys per second, seconds
settings = {"enabled": True, "effect": "flow", "speed": 1.0, "brightness": 1.0, "monitor": 1, "floor": 0.1,
            "source": "screen", "colors": ["#ff4d6d", "#ffb703", "#06d6a0", "#118ab2"]}
live = {"grid": [], "palette": [], "connected": False, "monitors": 2}  # read by the control panel
ripples = collections.deque(maxlen=16)  # (col, row, start time) from panel clicks and typing
save_lock = threading.Lock()  # one settings update + file write at a time
log = logging.getLogger("kreosync")


def build_packet(leds):
    """leds: (126, 3) uint8. Only ever report 6 / cmd 0x08: blind writes to report 5
    can drop this MCU into its flash bootloader."""
    return (HEADER + leds.tobytes()).ljust(PKT_LEN, b"\0")


def full(rgb):
    """Scale colors so their strongest channel is 255 (hue kept, black stays black)."""
    rgb = np.asarray(rgb, np.float64)
    mx = rgb.max(-1, keepdims=True)
    return np.where(mx > 1, rgb * (255 / np.maximum(mx, 1)), 0)


def palette(frame, n=5):
    """Up to n vivid screen colors, one per distinct hue, ordered by hue so the order is stable."""
    px = np.asarray(frame, np.float32).reshape(-1, 3)
    if not len(px):
        return [WHITE.copy()]
    mx = px.max(1)
    c = mx - px.min(1)
    keep = c > 40
    if keep.sum() < 0.01 * len(px):  # colorless scene: its own tint at full strength, white when it's dark
        mean = px.mean(0)
        return [full(mean) if mean.max() > BLACK else WHITE.copy()]
    px, c, mx = px[keep], c[keep], mx[keep]
    r, g, b = px.T
    hue = np.where(mx == r, (g - b) / c % 6, np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4)) / 6
    bins = (hue * 12).astype(int) % 12
    w = (c / 255) ** 2
    tot = np.bincount(bins, w, 12)
    kept = []  # (hue, color), strongest first; a hue split across two bins must not come back twice
    for i in np.argsort(tot)[::-1]:
        if tot[i] <= 0.1 * tot.max() or len(kept) == n:
            break
        col = (px[bins == i] * w[bins == i, None]).sum(0) / tot[i]
        h = colorsys.rgb_to_hsv(*(col / 255))[0]
        if all(min(abs(h - k), 1 - abs(h - k)) >= 0.05 for k, _ in kept):
            kept.append((h, full(col)))
    cols = [col for _, col in sorted(kept, key=lambda hc: hc[0])]
    if len(cols) == 1:  # one hue on screen: add analogous neighbours so the board still moves
        h, s, v = colorsys.rgb_to_hsv(*(cols[0] / 255))
        cols = [np.array(colorsys.hsv_to_rgb((h + d) % 1, s, v)) * 255 for d in (-0.06, 0, 0.06)]
    return cols


def gradient(pal, pos):
    """Sample the palette as a seamless loop; pos is any float array, period 1. Every color runs at
    full strength (black stays black) and blends take their hue from the mix but their brightness
    from the two ends, so they never dip between hues."""
    pal = full(pal) if len(pal) else WHITE[None]
    n = len(pal)
    p = (np.asarray(pos, np.float64) % 1) * n
    i = p.astype(np.int64)
    f = p - i
    f = (f * f * (3 - 2 * f))[..., None]  # smoothstep: soft blends, not hard bands
    a, b = pal[i % n], pal[(i + 1) % n]
    level = a.max(-1, keepdims=True) * (1 - f) + b.max(-1, keepdims=True) * f
    return full(a * (1 - f) + b * f) * (level / 255)


def _hsv(rgb):
    """Vectorised RGB -> (hue 0-1 or NaN for greys, saturation 0-1, value 0-255)."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(d > 0, d, 1)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    return np.where(d > 0, h, np.nan), np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0), mx


def _rgb(h, s, v):
    """Vectorised HSV -> RGB (h 0-1, s 0-1, v 0-255)."""
    i = np.floor(h * 6)
    f = h * 6 - i
    i = i.astype(int) % 6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.stack([np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]),
                     np.choose(i, [p, p, t, v, v, q])], -1)


def blend(shown, target, rate):
    """Move each key toward its target by rate, around the color wheel: a plain RGB mix between two
    hues dims (red to green passes dark olive) or washes out (red to cyan passes grey, then white)."""
    rate = np.asarray(rate, np.float64)
    rate = rate[..., 0] if rate.ndim == 3 else rate
    h1, s1, v1 = _hsv(shown)
    h2, s2, v2 = _hsv(target)
    h1, h2 = np.where(np.isnan(h1), h2, h1), np.where(np.isnan(h2), h1, h2)  # a grey takes the other's hue
    s1, s2 = np.where(v1 < 1, s2, s1), np.where(v2 < 1, s1, s2)  # black has no saturation of its own
    dh = (h2 - h1 + 0.5) % 1 - 0.5  # shortest way round
    h = np.nan_to_num(h1 + dh * rate) % 1
    return _rgb(h, s1 + (s2 - s1) * rate, v1 + (v2 - v1) * rate)


def flow(pal, t, dt):
    """Palette colors drift across the board in slow, wobbling diagonal waves."""
    pos = X / COLS + 0.18 * np.sin(Y * 0.9 + t * 1.1) + 0.1 * np.sin(X * 0.45 - t * 1.4) - t * 0.07
    return gradient(pal, pos)


class Rain:
    """Bright drops in palette colors fall down the columns over a dim, drifting glow of the
    palette, and splash out along the bottom row when they land."""
    TAIL = 2.5          # rows of fading trail behind each drop
    GLOW = 0.45         # background level; the contrast against full-bright drops is the effect
    TIP = 0.3           # how far the leading key is pushed toward white
    SPLASH_LIFE = 0.4   # seconds

    def __init__(self):
        self.drops = []     # [col, y, rows_per_s, palette pick 0..1, landed]
        self.splashes = []  # [col, age, color]

    def __call__(self, pal, t, dt):
        for col in range(COLS):
            if random.random() < 0.7 * dt:
                self.drops.append([col, -1.0, random.uniform(5, 9), random.random(), False])
        frame = gradient(pal, X / COLS + t * 0.03) * self.GLOW
        rows = Y[:, 0]
        for d in self.drops:
            d[1] += d[2] * dt
            color = full(pal[int(d[3] * len(pal))])
            dist = d[1] - rows  # >0: row is behind the head (tail), <0: just ahead of it
            k = np.where(dist >= 0, 1 - dist / self.TAIL, 1 + dist).clip(0, 1)[:, None]
            tip = (1 - np.abs(dist)).clip(0, 1)[:, None] * self.TIP
            drop = color + (255 - color) * tip
            frame[:, d[0]] = frame[:, d[0]] * (1 - k) + drop * k
            if not d[4] and d[1] >= ROWS - 1:
                d[4] = True
                self.splashes.append([d[0], 0.0, color])
        for s in self.splashes:
            s[1] += dt
            ring = np.exp(-((np.abs(X[-1] - s[0]) - s[1] * 6) / 0.7) ** 2) * (1 - s[1] / self.SPLASH_LIFE) * 0.8
            ring = ring.clip(0, 1)[:, None]
            frame[-1] = frame[-1] * (1 - ring) + s[2] * ring
        self.drops = [d for d in self.drops if d[1] - self.TAIL < ROWS]
        self.splashes = [s for s in self.splashes if s[1] < self.SPLASH_LIFE]
        return frame


def _bar_span(lit, n):
    """Start/end of the picture along one axis, trimming only centred bars (a dark half of a scene stays)."""
    idx = np.flatnonzero(lit)
    if len(idx) < 0.3 * n:
        return 0, n  # mostly black: nothing to trim
    head, tail = idx[0], n - 1 - idx[-1]
    return (int(idx[0]), int(idx[-1]) + 1) if abs(head - tail) <= 0.05 * n else (0, n)


def cell_means(f):
    """Average color of each key's patch, splitting the frame into ROWS x COLS cells that cover it
    edge to edge (cells differ by at most a pixel, so no edge is cropped)."""
    h, w = f.shape[:2]
    if h < ROWS or w < COLS:  # a tiny frame: repeat pixels so every key still gets a patch
        f = np.repeat(np.repeat(f, -(-ROWS // h), 0), -(-COLS // w), 1)
        h, w = f.shape[:2]
    re_ = np.linspace(0, h, ROWS + 1).astype(int)
    ce = np.linspace(0, w, COLS + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(f, re_[:-1], 0), ce[:-1], 1)
    return sums / (np.diff(re_)[:, None, None] * np.diff(ce)[None, :, None])


class Mirror:
    """Each key takes the color and brightness of the part of the screen it sits under.
    Film bars are trimmed so the picture fills the keyboard, but only once a crop has held for
    HOLD captures, so a lamp switching on in a dark sky doesn't make the picture jump."""
    HOLD = 10  # captures (~0.5 s at MIRROR_HZ)
    NEAR = 0.04  # crops whose edges differ by less than this share of the frame count as the same

    def __init__(self):
        self.crop = self.run = None  # the crop in use; the crop seen over the current run of captures
        self.seen = 0

    @classmethod
    def _near(cls, a, b):
        if a is None or b is None or a[4:] != b[4:]:
            return False
        tol = (cls.NEAR * a[4],) * 2 + (cls.NEAR * a[5],) * 2
        return all(abs(x - y) <= t for x, y, t in zip(a[:4], b[:4], tol))

    @staticmethod
    def find_crop(f):
        h, w = f.shape[:2]
        r = _bar_span(f.max((1, 2)) > BLACK, h)
        c = _bar_span(f.max((0, 2)) > BLACK, w)
        if (r != (0, h) and c != (0, w)) or r[1] - r[0] < ROWS or c[1] - c[0] < COLS:
            r, c = (0, h), (0, w)  # dark on all four sides is a dark scene, not film bars; never crop to slivers
        return (*r, *c, h, w)

    def __call__(self, frame, floor=0.0):
        """floor: the lowest level a key falls to (0-1, as the keys show it), so dark scenes still glow."""
        f = np.asarray(frame, np.float64)
        if f.size == 0:
            return np.zeros((ROWS, COLS, 3))
        found = self.find_crop(f)
        if self._near(found, self.run):  # same crop give or take a flickering edge row: widen to cover it
            self.seen += 1
            self.run = (min(self.run[0], found[0]), max(self.run[1], found[1]),
                        min(self.run[2], found[2]), max(self.run[3], found[3]), *found[4:])
        else:
            self.run, self.seen = found, 1
        if self.crop is None or self.crop[4:] != found[4:]:
            self.crop = found  # first frame or a new display size: adopt at once
        elif self.seen >= self.HOLD:
            self.crop = self.run
        r0, r1, c0, c1 = self.crop[:4]
        cells = cell_means(f[r0:r1, c0:c1])
        lum = cells.mean(-1, keepdims=True)
        cells = np.clip(lum + (cells - lum) * MIRROR_SAT, 0, 255)
        level = ((cells.max(-1, keepdims=True) - BLACK) / (255 - BLACK)).clip(0, 1)
        lift = floor ** (1 / GAMMA)  # to_leds applies GAMMA; undo it so the floor lands where asked
        tint = full(cells + 0.15 * cells.mean((0, 1)))  # dark keys borrow the scene's hue
        tint = np.where(tint.max(-1, keepdims=True) > 0, tint, 255)  # a black screen glows soft white
        return np.nan_to_num(tint * (lift + (1 - lift) * level))


def apply_ripples(frame, touches, now):
    """Expanding white rings from touched keys, laid over the effect."""
    for x, y, t0 in list(touches):
        age = now - t0
        if not 0 <= age < RIPPLE_LIFE:
            continue
        ring = np.exp(-((np.hypot(X - x, Y - y) - age * RIPPLE_SPEED) / 0.9) ** 2) * (1 - age / RIPPLE_LIFE)
        frame = frame + (255 - frame) * ring[..., None]
    return frame


def hex_rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], np.float64)


def to_leds(frame, brightness):
    f = np.clip(np.nan_to_num(frame), 0, 255)
    out = np.zeros((SLOTS, 3))
    out[SLOT] = np.rint(255 * (f / 255) ** GAMMA * brightness)  # round, don't truncate: low levels keep their hue
    return out.clip(0, 255).astype(np.uint8)


def grab(sct, mon, step):
    shot = sct.grab(mon)
    return np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)[::step, ::step, 2::-1]


def open_keyboard():
    probe = build_packet(np.zeros((SLOTS, 3), np.uint8))
    for d in hid.enumerate(VID, PID):
        if d["usage_page"] != 0xFF00:  # RGB lives on a vendor-defined collection
            continue
        h = hid.device()
        try:
            h.open_path(d["path"])
            if h.send_feature_report(probe) == PKT_LEN:
                return h
        except (OSError, ValueError):
            pass
        h.close()
    raise OSError("Hive 65 RGB interface not found (plugged in wired? official Kreo app closed?)")


def send(kb, leds):
    """Write one frame, (re)opening the keyboard as needed. Returns the open handle, or None."""
    try:
        if kb is None:
            kb = open_keyboard()
            log.info("keyboard connected")
        if kb.send_feature_report(build_packet(leds)) != PKT_LEN:
            raise OSError("feature report write failed")
        return kb
    except (OSError, ValueError) as e:
        # firmware sometimes resets and re-enumerates under streaming; reopen next time
        if kb is not None:
            log.warning("keyboard dropped (%s); reconnecting", e)
            kb.close()
        time.sleep(1)
        return None


def _reject_constant(name):
    raise ValueError(f"{name} is not a number")


def _number(val):
    """A real, finite JSON number (bools, strings, NaN and huge values are refused)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{val!r} is not a number")
    try:
        val = float(val)
    except OverflowError:
        raise ValueError("number too large") from None
    if not math.isfinite(val):
        raise ValueError("number must be finite")
    return val


def clean(data, monitors):
    """Validate untrusted settings (panel POST or hand-edited file). Raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("settings must be an object")
    out = {}
    for key, val in data.items():
        if key == "enabled" and isinstance(val, bool):
            out[key] = val
        elif key == "effect" and val in ("flow", "rain", "mirror"):
            out[key] = val
        elif key in LIMITS:
            lo, hi = LIMITS[key]
            out[key] = min(hi, max(lo, _number(val)))
        elif key == "monitor" and type(val) is int and 0 <= val < monitors:
            out[key] = val
        elif key == "source" and val in ("screen", "custom"):
            out[key] = val
        elif key == "colors" and isinstance(val, list) and 1 <= len(val) <= MAX_COLORS and all(
                isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in val):
            out[key] = [c.lower() for c in val]
        else:
            raise ValueError(f"bad setting {key}={val!r}"[:200])
    return out


def load_settings():
    """Apply settings.json key by key, so one stale or broken value doesn't discard the rest."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)  # NaN / Infinity parse here and are refused per key by clean()
    except FileNotFoundError:
        return
    except Exception as e:  # unreadable or corrupt: keep defaults, say why
        log.warning("settings.json unreadable (%s); using defaults", e)
        return
    if not isinstance(data, dict):
        log.warning("settings.json is not an object; using defaults")
        return
    for key, val in data.items():
        try:
            settings.update(clean({key: val}, MAX_MONITORS))
        except ValueError:
            log.warning("ignoring saved setting %s=%r", key, val)


def save_settings():
    """Write settings atomically (temp file, then rename) so a crash or a racing write can't leave
    half a file. The caller holds save_lock. Raises OSError if the file stays locked."""
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    for attempt in range(5):  # Windows refuses the rename while an editor, indexer or antivirus has the file open
        try:
            os.replace(tmp, SETTINGS_FILE)
            return
        except PermissionError:
            if attempt == 4:
                os.remove(tmp)
                raise
            time.sleep(0.03)


class Panel(BaseHTTPRequestHandler):
    timeout = 5  # a client that stalls mid-request can't hold a thread forever

    def log_message(self, *args):  # silent: runs under pythonw with no console
        pass

    def reply(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("X-Frame-Options", "DENY")  # no other site may frame the panel and steer clicks
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        if code != 204:
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if code != 204:
            self.wfile.write(body)

    def local(self):
        # a Host check stops DNS-rebinding web pages from talking to the panel
        return self.headers.get("Host") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self.local():
            return self.reply(403, b"{}")
        if self.path == "/":
            try:
                with open(os.path.join(HERE, "panel.html"), "rb") as f:
                    return self.reply(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self.reply(500, b"panel.html is missing", "text/plain")
        if self.path == "/api/state":
            return self.reply(200, json.dumps({"settings": settings, **live}).encode())
        self.reply(404, b"{}")

    def do_POST(self):
        # requiring a JSON content type forces a CORS preflight, so other sites can't post here
        if not self.local() or self.path not in ("/api/settings", "/api/touch") \
                or self.headers.get("Content-Type") != "application/json":
            return self.reply(403, b"{}")
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= 4096:
                raise ValueError("bad length")
            data = json.loads(self.rfile.read(n), parse_constant=_reject_constant)
            if self.path == "/api/touch":
                if not isinstance(data, dict):
                    raise ValueError("expected an object")
                x, y = _number(data.get("x")), _number(data.get("y"))
                if not (0 <= x <= COLS - 1 and 0 <= y <= ROWS - 1):
                    raise ValueError("key out of range")
                ripples.append((x, y, time.monotonic()))
                return self.reply(204)
            patch = clean(data, live["monitors"])
        except Exception as e:  # untrusted input: any parse or validation failure is the client's
            return self.reply(400, json.dumps({"error": str(e)[:200]}).encode())
        with save_lock:
            settings.update(patch)
            try:
                save_settings()
            except OSError as e:  # applied, but tell the panel it won't survive a restart
                log.warning("couldn't save settings (%s); applied for this session", e)
                return self.reply(500, json.dumps({"error": "settings.json is locked by another program, "
                                                            "so this change lasts until KreoSync restarts"}).encode())
            body = json.dumps(settings).encode()
        self.reply(200, body)


class PanelServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second copy bind the same port; exclusive use makes it fail instead
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        log.debug("panel request from %s failed", client_address, exc_info=True)


def main():
    try:
        server = PanelServer(("127.0.0.1", PORT), Panel)
    except OSError as e:
        log.error("can't open the control panel on port %d (%s); KreoSync is probably already running", PORT, e)
        return
    load_settings()  # before the panel starts answering, so an early save can't be overwritten
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("KreoSync running - control panel at http://127.0.0.1:%d", PORT)

    kb, pal, t = None, [WHITE.copy()], 0.0
    shown = np.zeros((ROWS, COLS, 3))
    target = shown.copy()  # the latest mirror frame, held between captures
    effects = {"flow": flow, "rain": Rain()}
    mirror, mirror_monitor = Mirror(), None
    sct, sct_at, next_capture, capture_failing = None, 0.0, 0.0, False
    prev = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            try:
                s = dict(settings)
                dt, prev = (now - prev) * s["speed"], now
                t += dt
                mirroring = s["effect"] == "mirror"
                need_screen = mirroring or s["source"] == "screen"
                # the display layout refreshes on its timer even when colors don't come from the screen,
                # so the panel can offer a newly attached display
                if now >= next_capture and (need_screen or now - sct_at > DISPLAY_REFRESH_S):
                    frame = None
                    try:
                        if sct is None or now - sct_at > DISPLAY_REFRESH_S:
                            old, sct = sct, None
                            if old is not None:
                                old.close()
                            sct, sct_at = mss.MSS(), now
                            live["monitors"] = len(sct.monitors)
                        if need_screen:
                            mons = sct.monitors
                            # a saved display that's unplugged falls back to the main one; the setting is kept
                            mon = mons[s["monitor"]] if s["monitor"] < len(mons) else mons[1 if len(mons) > 1 else 0]
                            frame = grab(sct, mon, max(1, mon["width"] // 160))  # analyse ~160 px wide
                    except Exception as e:  # lock screen, UAC prompt, display asleep: hold the last colors
                        if not capture_failing:
                            log.warning("screen capture failed (%s); holding the last colors and retrying", e)
                        capture_failing = True
                        if sct is not None:
                            try:
                                sct.close()
                            except Exception:
                                pass
                        sct, next_capture = None, now + CAPTURE_RETRY_S
                    else:
                        if capture_failing:
                            log.info("screen capture working again")
                        capture_failing = False
                    if frame is not None:
                        next_capture = now + 1 / (MIRROR_HZ if mirroring else PALETTE_HZ)
                        if mirroring:
                            if s["monitor"] != mirror_monitor:  # another display: its crop starts fresh
                                mirror, mirror_monitor = Mirror(), s["monitor"]
                            target = mirror(frame, s["floor"])
                        else:
                            pal = palette(frame)
                if mirroring:
                    rate = np.where(target.max(-1) > shown.max(-1), MIRROR_RISE, MIRROR_FALL)[..., None]
                else:
                    if s["source"] == "custom":
                        pal = [hex_rgb(c) for c in s["colors"]]
                    target, rate = effects[s["effect"]](pal, t, dt), SMOOTH
                shown = np.nan_to_num(blend(shown, target, rate))
                leds = to_leds(apply_ripples(shown, ripples, now), s["brightness"])
                live["grid"] = leds[SLOT].tolist()
                live["palette"] = [[int(round(v)) for v in c] for c in pal]
                if s["enabled"]:
                    kb = send(kb, leds)
                elif kb is not None:  # paused: stop streaming and the board returns to its own lighting
                    kb.close()
                    kb = None
                live["connected"] = kb is not None
            except Exception:  # never let one bad frame end the app; it runs unattended under pythonw
                log.exception("frame failed; carrying on")
                time.sleep(1)
            time.sleep(max(0.0, 1 / FPS - (time.monotonic() - now)))
    finally:
        if sct is not None:
            sct.close()
        server.shutdown()


def setup_logging():
    handler = RotatingFileHandler(LOG_FILE, maxBytes=256_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    if sys.stderr is not None:  # a console run also prints; pythonw has no stderr
        log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("KreoSync stopped")
        raise
