"""Hardware-free checks. Run: .venv\\Scripts\\python test_kreosync.py"""
import http.client
import json
import os
import random
import tempfile
import threading

import numpy as np

import kreosync as k

# ---- packet and LED map ----
pkt = k.build_packet(np.full((k.SLOTS, 3), 7, np.uint8))
assert len(pkt) == 520 and pkt[:8] == bytes.fromhex("0608000001007a01")
assert pkt[8:386] == bytes([7]) * 378 and pkt[386:] == bytes(134)
assert k.SLOT[0, 0] == 1 and k.SLOT[4, 5] == 35  # Esc, Space (PROTOCOL.md)

# ---- palette ----
f = np.zeros((90, 160, 3), np.uint8)
f[:, :80] = (200, 0, 0)
f[:, 80:] = (0, 180, 0)
pal = [tuple(int(v) for v in np.round(c)) for c in k.palette(f)]
assert pal == [(255, 0, 0), (0, 255, 0)], pal  # two colors, never a muddy average
assert len(k.palette(np.full((9, 16, 3), (0, 0, 200), np.uint8))) == 3  # one hue gets neighbours
for level in (0, 1, 2, 5, 50, 255):  # colorless screens give one steady white, never black / white flicker
    assert np.allclose(k.palette(np.full((9, 16, 3), level, np.uint8))[0], 255), level
assert np.allclose(k.palette(np.zeros((0, 0, 3), np.uint8))[0], 255)  # empty frame doesn't crash
reds = np.zeros((90, 160, 3), np.uint8)
reds[:, :80] = (255, 0, 8)   # hue just below 360
reds[:, 80:] = (255, 8, 0)   # hue just above 0: the same red, split across the wrap
assert len(k.palette(reds)) == 3  # treated as one hue (gets neighbours), not two near-identical reds

# ---- gradient, blend, flow ----
assert np.allclose(k.full(np.array([[40.0, 0, 0], [0, 0, 0]])), [[255, 0, 0], [0, 0, 0]])
mid = k.gradient([(255, 0, 0), (0, 255, 0)], np.array([0.25]))
assert np.isclose(mid.max(), 255) and mid[0, 0] > 200 and mid[0, 1] > 200  # halfway red->green is bright yellow
dark_custom = k.gradient([(255, 0, 0), (0, 0, 0), (0, 0, 255)], np.linspace(0, 1, 300, endpoint=False))
assert (dark_custom.max(-1) < 30).mean() > 0.1  # a black custom color shows as a real dark band
step = k.blend(np.full((1, 3), (255.0, 0, 0)), np.full((1, 3), (0.0, 255, 0)), 0.5)
assert np.isclose(step.max(), 255)  # moving between hues never dips in brightness
assert np.allclose(k.blend(np.full((1, 3), 200.0), np.zeros((1, 3)), 0.5).max(), 100)  # fading out still fades
red, cyan = np.full((1, 3), (255.0, 0, 0)), np.full((1, 3), (0.0, 255, 255))
cur = red.copy()
for _ in range(20):  # opposite hues travel round the wheel: never a washed-out or white frame
    cur = k.blend(cur, cyan, 0.25)
    assert cur.max() > 254 and cur.min() < 60, cur
assert np.allclose(cur, cyan, atol=8)
fade_in = k.blend(np.zeros((1, 3)), red, 0.5)
assert fade_in[0, 1] < 1 and fade_in[0, 2] < 1  # from black, a color fades in as itself, not pink
for rgb in ((255, 0, 0), (12, 200, 90), (128, 128, 128), (0, 0, 0), (255, 255, 255)):  # HSV round trip is exact
    c = np.array([rgb], float)
    assert np.allclose(k.blend(c, c, 0.3), rgb, atol=0.01), rgb
assert (k.flow(pal, 3.0, 0.03).max(-1) > 254).all()
f1, f2 = k.flow(pal, 1e7, 0.03), k.flow(pal, 1e7 + 1 / 30, 0.03)
assert np.abs(f1 - f2).max() > 1  # still animating after ~115 days of effect time

# ---- effects stay in range ----
random.seed(1)
rain = k.Rain()
for effect in (k.flow, rain):
    for i in range(60):
        out = effect(pal, i / 30, 1 / 30)
        assert out.shape == (k.ROWS, k.COLS, 3) and out.min() >= 0 and out.max() <= 255.01
assert rain.drops, "rain never spawned"
random.seed(3)
rain = k.Rain()
peaks, lows, splashed = [], [], False
for i in range(120):
    level = k.to_leds(rain(pal, i / 30, 1 / 30), 1.0)[k.SLOT].max(-1)
    peaks.append(level.max())
    lows.append(np.median(level))
    splashed |= bool(rain.splashes)
assert max(peaks) >= 250 and np.median(lows) < 80 and splashed, (max(peaks), np.median(lows))

# ---- to_leds ----
leds = k.to_leds(np.full((k.ROWS, k.COLS, 3), (255, 0, 0), np.float32), 1.0)
assert tuple(leds[k.SLOT[0, 0]]) == (255, 0, 0) and tuple(leds[0]) == (0, 0, 0)  # unused slots stay off
assert tuple(k.to_leds(np.full((k.ROWS, k.COLS, 3), (255, 128, 0)), 0.05)[k.SLOT[0, 0]]) == (13, 4, 0)  # rounded
assert not k.to_leds(np.full((k.ROWS, k.COLS, 3), np.nan), 1.0).any()  # NaN never reaches the keyboard

# ---- ripples ----
black = np.zeros((k.ROWS, k.COLS, 3))
touch = [(5.0, 2.0, 10.0)]
assert np.allclose(k.apply_ripples(black, touch, 10.0)[2, 5], 255)
later = k.apply_ripples(black, touch, 10.5)
assert later[2, 5].max() < 50 and later[2, 10].max() > 100
assert not k.apply_ripples(black, touch, 10.01 + k.RIPPLE_LIFE).any()
assert np.array_equal(k.hex_rgb("#ff8000"), [255, 128, 0])

# ---- mirror ----
mirror = lambda frame, floor=0.0: k.Mirror()(frame, floor)  # a fresh instance adopts its first crop at once
f = np.zeros((90, 160, 3), np.uint8)
f[:, :80] = (220, 30, 30)
m = mirror(f)
assert m.shape == (k.ROWS, k.COLS, 3)
assert (m[:, :8, 0] > 200).all() and (m[:, :8, 1:] < 40).all() and not m[:, 8:].any()  # dark half stays dark
grey = mirror(np.full((90, 160, 3), 128, np.uint8))
assert 100 < grey.max() < 140 and np.ptp(grey) < 1
assert not mirror(np.full((90, 160, 3), 10, np.uint8)).any()
film = np.zeros((90, 160, 3), np.uint8)
film[18:72] = (255, 255, 255)
assert (mirror(film) > 250).all()  # letterbox trimmed: all five rows light up
dusk = np.zeros((90, 160, 3), np.uint8)
dusk[54:] = (255, 140, 0)
assert not mirror(dusk)[0].any() and (mirror(dusk)[-1].max(-1) > 250).all()  # a black sky is not a letterbox
subject = np.zeros((90, 160, 3), np.uint8)
subject[30:60, 50:110] = (255, 255, 255)  # dark scene, bright subject in the middle
sm = mirror(subject)
assert not sm[0].any() and not sm[:, 0].any() and sm[2, 8].max() > 250  # not stretched over the whole board
edge = np.zeros((90, 120, 3), np.uint8)
edge[:, -7:] = 255  # 4:3 frame lit only at its right edge
assert mirror(edge)[:, -1].max() > 100  # right-hand keys aren't cropped away
for tiny in ((3, 4), (12, 160), (90, 10), (1, 1)):  # odd capture sizes never crash or produce NaN
    out = mirror(np.full((*tiny, 3), 200, np.uint8))
    assert out.shape == (k.ROWS, k.COLS, 3) and np.isfinite(out).all(), tiny
assert not mirror(np.zeros((0, 0, 3), np.uint8)).any()
lamp = film.copy()
lamp[2, 80] = 255  # a small light switches on in the top bar
mm = k.Mirror()
assert (mm(film) > 250).all()
for _ in range(k.Mirror.HOLD - 1):
    assert (mm(lamp) > 250).all()  # the crop holds: no jump for a brief change
mm(lamp)
assert not (mm(lamp) > 250).all()  # a change that lasts is followed
assert np.allclose(k.to_leds(mirror(np.zeros((90, 160, 3), np.uint8), 0.1), 1.0)[k.SLOT], 26, atol=1.5)
sky = k.to_leds(mirror(dusk, 0.1), 1.0)[k.SLOT][0]
assert np.allclose(sky.max(-1), 26, atol=1.5) and (sky[:, 0] > sky[:, 2]).all()  # floor glows warm, not grey

# ---- settings validation ----
assert k.clean({"speed": 99, "brightness": 0, "effect": "rain", "enabled": False, "monitor": 1}, 2) == \
    {"speed": 3.0, "brightness": 0.05, "effect": "rain", "enabled": False, "monitor": 1}
assert k.clean({"source": "custom", "colors": ["#FF0000", "#00ff00"]}, 2) == \
    {"source": "custom", "colors": ["#ff0000", "#00ff00"]}
assert k.clean({"effect": "mirror", "floor": 0.9}, 2) == {"effect": "mirror", "floor": 0.4}
for bad in ({"effect": "x"}, {"enabled": "yes"}, {"monitor": 5}, {"hack": 1}, [1], {"speed": "fast"},
            {"source": "web"}, {"colors": []}, {"colors": ["red"]}, {"colors": ["#ff0000"] * 7},
            {"colors": "#ff0000"}, {"colors": ["#ff00001"]}, {"speed": True}, {"speed": " 2.5 "},
            {"speed": float("nan")}, {"speed": float("inf")}, {"speed": 10 ** 400}, {"monitor": 1.7},
            {"monitor": "1"}, {"monitor": True}, {"monitor": -1}, {"monitor": float("inf")}):
    try:
        k.clean(bad, 2)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass

# ---- settings file: per-key load, atomic save ----
tmpdir = tempfile.mkdtemp()
k.SETTINGS_FILE = os.path.join(tmpdir, "settings.json")
defaults = dict(k.settings)
with open(k.SETTINGS_FILE, "w") as fh:
    fh.write('{"brightness": 0.3, "effect": "rain", "monitor": 3, "speed": "fast", "floor": Infinity}')
k.load_settings()
assert k.settings["brightness"] == 0.3 and k.settings["effect"] == "rain"  # Infinity drops only that key
assert k.settings["floor"] == defaults["floor"]
k.settings.update(defaults)
with open(k.SETTINGS_FILE, "w") as fh:
    json.dump({"brightness": 0.3, "effect": "rain", "monitor": 3, "speed": "fast"}, fh)
k.load_settings()
assert k.settings["brightness"] == 0.3 and k.settings["effect"] == "rain" and k.settings["monitor"] == 3
assert k.settings["speed"] == defaults["speed"]  # only the broken key is dropped; an unplugged display is kept
k.settings.update(defaults)
with open(k.SETTINGS_FILE, "w") as fh:
    fh.write("{not json")
k.load_settings()  # corrupt file: defaults, no crash
assert k.settings == defaults
k.save_settings()
with open(k.SETTINGS_FILE) as fh:
    assert json.load(fh) == defaults and not os.path.exists(k.SETTINGS_FILE + ".tmp")

# ---- control panel server ----
k.PORT = 8797
server = k.PanelServer(("127.0.0.1", k.PORT), k.Panel)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    k.PanelServer(("127.0.0.1", k.PORT), k.Panel)
    raise AssertionError("a second server bound the same port")  # second copy must fail, not share
except OSError:
    pass


def post(path, body, ctype="application/json", host=f"127.0.0.1:{k.PORT}"):
    c = http.client.HTTPConnection("127.0.0.1", k.PORT, timeout=5)
    c.request("POST", path, body, {"Content-Type": ctype, "Host": host})
    r = c.getresponse()
    return r.status, r.read(), dict(r.getheaders())


status, body, headers = post("/api/settings", '{"brightness": 0.5}')
assert status == 200 and json.loads(body)["brightness"] == 0.5
assert headers["X-Frame-Options"] == "DENY" and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
for bad_body in ('{"monitor": Infinity}', '{"speed": 1' + "0" * 400 + '}', "[" * 3000, '"x"', "null", "\xff"):
    assert post("/api/settings", bad_body)[0] == 400, bad_body  # rejected cleanly, never a dropped connection
assert post("/api/settings", '{"speed": 2}', ctype="text/plain")[0] == 403
assert post("/api/settings", '{"speed": 2}', host="evil.example:8765")[0] == 403
status, body, headers = post("/api/touch", '{"x": 3, "y": 1}')
assert status == 204 and body == b"" and "Content-Length" not in headers
for bad_touch in ('{"x": 99, "y": 1}', '{"x": true, "y": 1}', '{"x": "3", "y": 1}', '{"x": 3}', "[3, 1]"):
    assert post("/api/touch", bad_touch)[0] == 400, bad_touch
with open(k.SETTINGS_FILE) as fh:
    assert json.load(fh)["brightness"] == 0.5
threads = [threading.Thread(target=post, args=("/api/settings", json.dumps({"colors": ["#123456"] * (i % 6 + 1)})))
           for i in range(40)]
for th in threads:
    th.start()
for th in threads:
    th.join()
with open(k.SETTINGS_FILE) as fh:
    assert json.load(fh) == k.settings  # 40 racing saves leave a whole, current file
real_replace = k.os.replace
k.os.replace = lambda *a: (_ for _ in ()).throw(PermissionError("locked"))  # another program holds the file
status, body, _ = post("/api/settings", '{"effect": "mirror"}')
k.os.replace = real_replace
assert status == 500 and b"locked" in body and k.settings["effect"] == "mirror"  # applied, and the panel is told
assert not os.path.exists(k.SETTINGS_FILE + ".tmp")
# a film whose top and bottom picture rows flicker between dark and lit is still cropped once it settles
mm, rng = k.Mirror(), np.random.default_rng(0)
mm(np.full((90, 160, 3), 12, np.uint8))  # first capture is a dark scene
for _ in range(30):
    fr = np.zeros((90, 160, 3), np.uint8)
    fr[12:78] = 200
    fr[12:12 + rng.integers(0, 4)] = 10
    fr[78 - rng.integers(0, 4):78] = 10
    out = mm(fr)
assert out[0].max() > 120 and out[-1].max() > 120, out.max(-1)  # all five rows show picture (uncropped: ~60)
server.shutdown()
print("ok")
