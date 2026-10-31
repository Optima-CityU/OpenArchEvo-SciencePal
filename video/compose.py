#!/usr/bin/env python3
"""
compose.py -- procedural background score for the OpenArchEvo / SciencePal hackathon video.

Everything is synthesised from scratch with numpy/scipy (additive pads, additive plucks,
sine pulse/sub, FM bells, filtered noise, synthetic-IR convolution reverb). No samples.

Usage
    python3 compose.py                              # default section lengths (sections.json if present)
    python3 compose.py --config my_sections.json    # re-cut to new section lengths
    python3 compose.py --config x.json --outdir out --seed 7 --lufs -20

Config JSON (preferred: typed list; "type" is one of TYPES, "dur" or "start"/"end" in seconds):
    {"bpm": 88, "hits": [77.9], "tail": 0.0,
     "sections": [{"label": "intro", "type": "intro", "start": 0.0, "end": 15.9}, ...]}
Legacy forms (6 sections in the order intro, breeders, method, results, discovery, ending):
    {"bpm": 88, "sections": {"intro": 25, "breeders": 30, "method": 45,
                             "results": 50, "discovery": 35, "ending": 15}}
    [25, 30, 45, 50, 35, 15]          # durations in seconds, in the order above

Outputs (in --outdir):
    bgm_full.wav, sections/NN_<name>.wav, bgm_loop.wav, timing.json  (48 kHz, 24-bit stereo)
"""
import argparse, json, os, subprocess, tempfile, wave, re
import numpy as np
from scipy import signal

SR = 48000
NAMES = ["intro", "breeders", "method", "results", "discovery", "ending"]   # legacy 6-section order
# Section *types* (each section in the config picks one). Column order of LEVELS below.
TYPES = ["intro", "breeders", "method", "results", "discovery", "ending", "loop", "bigpicture", "title"]
DEFAULT_DURS = [25, 30, 45, 50, 35, 15]

# ----------------------------------------------------------------------------- harmony
# pitch classes: C0 C#1 D2 D#3 E4 F5 F#6 G7 G#8 A9 A#10 B11      (key: D major / B minor)
CH = {
    "Dmaj9":  (2, [2, 6, 9, 1, 4]),
    "Dadd9":  (2, [2, 6, 9, 4]),
    "D/F#":   (6, [2, 6, 9, 4]),
    "Bm7":    (11, [11, 2, 6, 9]),
    "Bm9":    (11, [11, 2, 6, 9, 1]),
    "Bm11":   (11, [11, 2, 6, 9, 4]),
    "Gmaj7":  (7, [7, 11, 2, 6]),
    "Gmaj9":  (7, [7, 11, 2, 6, 9]),
    "Asus4":  (9, [9, 2, 4]),
    "A":      (9, [9, 1, 4]),
    "A7sus4": (9, [9, 2, 4, 7]),
    "Em9":    (4, [4, 7, 11, 2, 6]),
    "Em7":    (4, [4, 7, 11, 2]),
    "F#m7":   (6, [6, 9, 1, 4]),
}

# Per-section musical design. hr = harmonic rhythm in bars; prog cycles; cadence closes section.
SEC = {
    "intro":     dict(hr=2, prog=["Dmaj9", "Bm11", "Gmaj7", "D/F#"], cadence="Asus4",
                      fc=800, att=3.0, rel=4.0, arp="droplets", drop_p=0.55, pulse=None),
    "breeders":  dict(hr=2, prog=["Bm9", "Gmaj7", "Dmaj9", "Asus4"], cadence="A",
                      fc=950, att=2.0, rel=3.5, arp="droplets", drop_p=0.35,
                      pulse=[1.0, .5, .7, .5], pulse_div=1, pulse_ramp_beats=20),
    "method":    dict(hr=2, prog=["Dadd9", "Bm7", "Gmaj7", "Asus4"], cadence="A7sus4",
                      fc=1050, att=1.4, rel=3.0, arp=[0, 2, 1, 3, 2, 4, 3, 1], arp_fc=2000,
                      pulse=[.9, .5, .7, .5], pulse_div=1),
    "results":   dict(hr=2, prog=["Gmaj7", "A", "F#m7", "Bm9", "Em9", "Gmaj7", "D/F#", "Asus4"],
                      cadence="A", fc=1400, att=1.2, rel=3.0, arp=[0, 2, 4, 2, 1, 3, 5, 3],
                      arp_fc=2300, pulse=[1, .35, .6, .35, .85, .35, .6, .35], pulse_div=2),
    "discovery": dict(hr=2, prog=["Gmaj9", "D/F#", "Em9", "Asus4", "Gmaj7", "A", "Bm9", "Asus4"],
                      cadence="A7sus4", fc=1800, att=1.4, rel=3.2, arp=[1, 3, 5, 4, 2, 4, 6, 3],
                      arp_fc=2700, pulse=[.9, .5, .7, .5], pulse_div=1, bells=True),
    "ending":    dict(fixed=[("Gmaj7", 4), ("A7sus4", 2), ("Dmaj9", None)],
                      fc=1100, att=1.6, rel=5.0, arp="ending", arp_fc=2200,
                      pulse=[.8, 0, 0, 0], pulse_div=1, pulse_beats=8, bells="ending"),
    # "architectures are the foundation of AI": gentle swell, rising bass G-A-B (history/progress)
    "bigpicture": dict(hr=2, prog=["Gmaj9", "A", "Bm9", "D/F#"], cadence="Asus4",
                      fc=1000, att=2.2, rel=3.5, arp="droplets", drop_p=0.7, pulse=None),
    # title card: lands on the tonic (V -> I under the title hit), pulse breathes out
    "title":     dict(fixed=[("Dmaj9", 8), ("Gmaj9", None)],
                      fc=1300, att=0.9, rel=3.5, arp="droplets", drop_p=0.8, pulse=None),
    # 30 s seamless loop bed (88 BPM, 44 beats)
    "loop":      dict(fixed=[("Dadd9", 12), ("Bm9", 8), ("Gmaj7", 12), ("Asus4", 8), ("A", 4)],
                      fc=1100, att=1.4, rel=3.0, arp=[0, 2, 1, 3, 2, 4, 3, 1], arp_fc=2000,
                      pulse=[.9, .5, .7, .5], pulse_div=1),
}

# Layer levels in dB per section (None = layer silent in that section)
LEVELS = {
    #            intro  breeders method results discovery ending  loop   bigpic  title
    "pad":      [-2,    -2.5,    -3.5,   -2.5,   -2,       -2,    -3.5,  -2,     -2],
    "pad_hi":   [None,  None,    None,   -9,     -8,       None,  None,  -11,    -10],
    "arp":      [-9,    -12,     -6,     -5,     -5.5,     -8,    -6,    -10,    -11],
    "pulse":    [None,  -4,      -3,     -1.5,   -3,       -6,    -3,    None,   None],
    "sub":      [None,  -12,     -9,     -6,     -8,       -9,    -9,    -15,    -12],
    "bell":     [None,  None,    None,   None,   -13,      -12,   None,  None,   None],
    "shaker":   [None,  None,    None,   -22,    None,     None,  None,  None,   None],
    "air":      [-24,   None,    None,   None,   -27,      -27,   None,  -28,    -28],
    "swell":    [-22,   None,    -22,    -22,    None,     None,  None,  -22,    -22],
    "hit":      [-8,    -8,      -8,     -8,     -8,       -8,    None,  -8,     -8],
}
# Whole-section offset (dB, start -> end of section, linear in dB) -> gentle dynamic arc
SECTION_DB = {"intro": (-5.5, -5.0), "bigpicture": (-5.0, -2.5), "breeders": (-3.5, -3.0),
              "title": (-2.5, -3.0), "method": (-2.5, -2.0), "results": (-1.0, 0.0),
              "discovery": (-0.5, -0.5), "ending": (-2.0, -2.0), "loop": (0.0, 0.0)}
REVERB_SEND = {"pad": 0.55, "pad_hi": 0.6, "arp": 0.45, "pulse": 0.06, "sub": 0.0,
               "bell": 0.75, "shaker": 0.25, "air": 0.5, "swell": 0.6, "hit": 0.6}
HITS = []          # absolute times (s) of soft picture accents, set from config "hits"


def m2f(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def voice_chord(pcs, bass_pc, prev, lo=55, hi=79, n=5):
    """Pick an n-note voicing in [lo,hi] that covers the chord's colour tones and moves
    as little as possible from the previous voicing (classic minimal-motion voice leading)."""
    import itertools
    cand = [m for m in range(lo, hi + 1) if m % 12 in pcs]
    need = set(pcs) - {bass_pc} if len(pcs) >= 4 else set(pcs)
    best, bestc = None, 1e9
    for combo in itertools.combinations(cand, n):
        if not need <= {m % 12 for m in combo}:
            continue
        ok = True
        for a, b in zip(combo, combo[1:]):
            iv = b - a
            if iv < 2 or (a < 60 and iv < 4) or iv > 9:
                ok = False
                break
        if not ok:
            continue
        cost = 0.0
        if prev is not None:
            cost += sum(abs(a - b) for a, b in zip(combo, prev))
        cost += 0.15 * abs(np.mean(combo) - 67)
        if cost < bestc:
            best, bestc = combo, cost
    if best is None:   # relax
        return voice_chord(pcs, bass_pc, prev, lo - 2, hi + 2, n)
    return list(best)


def chord_plan(cfg, n_beats):
    """List of (chord_name, start_beat, len_beats) filling exactly n_beats."""
    if "fixed" in cfg:
        plan, b = [], 0
        for name, L in cfg["fixed"]:
            L = n_beats - b if L is None else min(L, n_beats - b)
            if L <= 0:
                break
            plan.append((name, b, L))
            b += L
        if b < n_beats:
            plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + n_beats - b)
        return plan
    slot = cfg["hr"] * 4
    k, r = divmod(n_beats, slot)
    seq = []
    if k == 0:
        return [(cfg["cadence"], 0, n_beats)]
    for i in range(k - 1):
        seq.append([cfg["prog"][i % len(cfg["prog"])], slot])
    if r >= slot // 2:                       # extra bar-aligned chord, odd beats go to the cadence
        seq.append([cfg["prog"][(k - 1) % len(cfg["prog"])], (r // 4) * 4])
        seq.append([cfg["cadence"], slot + r % 4])
    else:
        seq.append([cfg["cadence"], slot + r])
    merged = []
    for name, L in seq:                      # never repeat the same chord back to back
        if merged and merged[-1][0] == name:
            merged[-1][1] += L
        else:
            merged.append([name, L])
    plan, b = [], 0
    for name, L in merged:
        plan.append((name, b, L))
        b += L
    return plan


# ----------------------------------------------------------------------------- envelopes
def smooth_ramp(n):
    x = np.linspace(0, 1, max(n, 1), dtype=np.float32)
    return (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32)


def asr_env(n_on, n_att, n_rel):
    n_att = min(n_att, n_on)
    e = np.ones(n_on + n_rel, np.float32)
    e[:n_att] = smooth_ramp(n_att)
    lvl = e[n_on - 1] if n_on > 0 else 0
    e[n_on:] = lvl * smooth_ramp(n_rel)[::-1] ** 1.5
    return e


def pan_gains(p):          # p in [-1,1], constant power
    a = (p + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def add_at(buf, start, sig_l, sig_r=None):
    if sig_r is None:
        sig_r = sig_l
    s = int(start)
    if s >= buf.shape[1]:
        return
    o = 0
    if s < 0:
        o = -s; s = 0
    n = min(len(sig_l) - o, buf.shape[1] - s)
    if n <= 0:
        return
    buf[0, s:s + n] += sig_l[o:o + n]
    buf[1, s:s + n] += sig_r[o:o + n]


# ----------------------------------------------------------------------------- instruments
def pad_note(f, dur_on, att, rel, fc, sr, rng, detunes=(-7, 0, 7), pans=(-0.7, 0, 0.7), amp=1.0):
    """Additive band-limited saw stack with a slowly breathing low-pass (evolving pad)."""
    n_on, n_att, n_rel = int(dur_on * sr), int(att * sr), int(rel * sr)
    env = asr_env(n_on, n_att, n_rel) * amp
    n = len(env)
    t = np.arange(n, dtype=np.float64) / sr
    L = np.zeros(n, np.float32); R = np.zeros(n, np.float32)
    for d, p in zip(detunes, pans):
        d = d + rng.normal(0, 1.5)
        fo = f * 2 ** (d / 1200)
        vib = 1 + 0.0010 * np.sin(2 * np.pi * rng.uniform(0.08, 0.17) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(fo * vib) / sr + rng.uniform(0, 6.28)
        fct = fc * (1 + 0.28 * np.sin(2 * np.pi * t / rng.uniform(7, 12) + rng.uniform(0, 6.28)))
        x4 = ((fo / fct) ** 4).astype(np.float32)
        K = int(min(28, (sr * 0.45) / fo, max(4, 2.2 * fc * 1.3 / fo + 3)))
        s1 = np.sin(ph).astype(np.float32); c2 = (2 * np.cos(ph)).astype(np.float32)
        sk, skm = s1, np.zeros_like(s1)
        acc = np.zeros(n, np.float32)
        for k in range(1, K + 1):
            w = 1.0 / (k * np.sqrt(1 + (k ** 4) * x4))
            acc += w * sk
            sk, skm = c2 * sk - skm, sk
        gl, gr = pan_gains(p)
        L += gl * acc; R += gr * acc
    return L * env, R * env


def pluck_note(f, vel, fc, tau0, rng, length=2.6):
    n = int(length * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    B = 0.00015
    for k in range(1, 9):
        fk = k * f * np.sqrt(1 + B * k * k)
        if fk > 12000:
            break
        a = (1 / k ** 1.25) / np.sqrt(1 + (fk / fc) ** 4)
        tau = tau0 / (1 + 0.55 * (k - 1))
        out += a * np.exp(-t / tau) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28))
    na = int(0.009 * SR)
    out[:na] *= smooth_ramp(na)
    nr = int(0.25 * SR)
    out[-nr:] *= smooth_ramp(nr)[::-1]
    return out * vel


def pulse_note(f, vel, length=0.95):
    n = int(length * SR)
    t = np.arange(n, dtype=np.float32) / SR
    s = np.sin(2 * np.pi * f * t) + 0.38 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    e = np.exp(-t / 0.26).astype(np.float32)
    na = int(0.03 * SR)
    e[:na] *= smooth_ramp(na)
    nr = int(0.2 * SR)
    e[-nr:] *= smooth_ramp(nr)[::-1]
    return (s * e * vel).astype(np.float32)


def sub_note(f, dur_on, rng):
    env = asr_env(int(dur_on * SR), int(1.0 * SR), int(1.6 * SR))
    t = np.arange(len(env), dtype=np.float32) / SR
    ph = rng.uniform(0, 6.28)
    s = np.sin(2 * np.pi * f * t + ph) + 0.22 * np.sin(4 * np.pi * f * t + ph)
    return (s * env).astype(np.float32)


def bell_note(f, vel, rng, length=4.0):
    n = int(length * SR)
    t = np.arange(n, dtype=np.float32) / SR
    I = 1.3 * np.exp(-t / 0.35)
    s = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * 2.0 * f * t)) \
        + 0.25 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.5)
    e = np.exp(-t / 1.6)
    na = int(0.012 * SR)
    e[:na] *= smooth_ramp(na)
    nr = int(0.4 * SR)
    e[-nr:] *= smooth_ramp(nr)[::-1]
    return (s * e * vel).astype(np.float32)


def band_noise(n, lo, hi, rng, order=4):
    x = rng.standard_normal(n + 4000).astype(np.float32)
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)[4000:].astype(np.float32)


# ----------------------------------------------------------------------------- effects
def make_ir(rng, length=3.4):
    """Stereo synthetic hall IR: decorrelated noise, frequency-dependent RT60, early reflections."""
    n = int(length * SR)
    t = np.arange(n) / SR
    bands = [(None, 300, 2.9), (300, 2000, 2.5), (2000, 6000, 1.8), (6000, None, 1.0)]
    irs = []
    for ch in range(2):
        noise = rng.standard_normal(n)
        ir = np.zeros(n)
        for lo, hi, rt in bands:
            if lo is None:
                sos = signal.butter(2, hi, "low", fs=SR, output="sos")
            elif hi is None:
                sos = signal.butter(2, lo, "high", fs=SR, output="sos")
            else:
                sos = signal.butter(2, [lo, hi], "band", fs=SR, output="sos")
            ir += signal.sosfilt(sos, noise) * np.exp(-6.91 * t / rt)
        pre = int(0.022 * SR)
        ir = np.concatenate([np.zeros(pre), ir[:-pre]])
        ramp = int(0.035 * SR)
        ir[pre:pre + ramp] *= np.linspace(0, 1, ramp) ** 2
        for k in range(10):   # sparse early reflections
            d = pre + int(rng.uniform(0.004, 0.06) * SR)
            ir[d] += rng.uniform(-1, 1) * 0.35 * np.exp(-k / 5)
        nr = int(0.3 * SR)
        ir[-nr:] *= np.linspace(1, 0, nr)
        ir /= np.sqrt(np.sum(ir ** 2))
        irs.append(ir.astype(np.float32))
    return irs


def reverb(x, irs):
    l = signal.oaconvolve(x[0], irs[0])[: x.shape[1]]
    r = signal.oaconvolve(x[1], irs[1])[: x.shape[1]]
    lx = signal.oaconvolve(x[1], irs[0])[: x.shape[1]]
    rx = signal.oaconvolve(x[0], irs[1])[: x.shape[1]]
    return np.stack([l + 0.25 * lx, r + 0.25 * rx]).astype(np.float32)


def chorus(x, t0):
    n = x.shape[1]
    idx = np.arange(n, dtype=np.float64)
    t = (idx / SR) + t0
    y = np.empty_like(x)
    for ch, ph in ((0, 0.0), (1, np.pi)):
        d = (0.013 + 0.0035 * np.sin(2 * np.pi * 0.17 * t + ph)) * SR
        y[ch] = x[ch] + 0.45 * np.interp(idx - d, idx, x[ch], left=0.0)
    return y / 1.2


def pingpong(x, delay_s, fb=0.38, taps=5, mix=0.35):
    D = int(delay_s * SR)
    n = x.shape[1]
    mono = 0.5 * (x[0] + x[1])
    wet = np.zeros_like(x)
    for k in range(1, taps + 1):
        if k * D >= n:
            break
        ch = (k + 1) % 2
        wet[ch, k * D:] += (fb ** (k - 1)) * mono[: n - k * D]
    sos = signal.butter(2, [250, 3500], "band", fs=SR, output="sos")
    wet = signal.sosfilt(sos, wet, axis=1).astype(np.float32)
    return x + mix * wet


# ----------------------------------------------------------------------------- section render
LAYERS = list(LEVELS.keys())


def render_section(name, dur, bpm_nom, t0, rng, prev_voicing, prev_bass, col, prev_type=None):
    """Render all layers of one section (dry, un-automated) into section-local buffers.
    Buffers start at t0 and are dur + SPILL long. Returns dict layer->(2,n) and state."""
    cfg = SEC[name]
    n_beats = max(2, int(round(dur * bpm_nom / 60)))
    bpm = n_beats * 60.0 / dur
    beat = 60.0 / bpm
    SPILL = 9.0
    N = int((dur + SPILL) * SR)
    buf = {l: np.zeros((2, N), np.float32) for l in LAYERS}
    plan = chord_plan(cfg, n_beats)
    active = lambda l: LEVELS[l][col] is not None

    # chord voicings with voice leading; common tones tied
    voicings, basses = [], []
    v, b = prev_voicing, prev_bass
    for cname, sb, L in plan:
        bpc, pcs = CH[cname]
        v = voice_chord(pcs, bpc, v)
        cands = [m for m in range(33, 48) if m % 12 == bpc]
        b = min(cands, key=lambda m: abs(m - (b if b else 40)) + 0.3 * abs(m - 40))
        voicings.append(v); basses.append(b)

    # ---- pad (rendered at SR/2, upsampled)
    psr = SR // 2
    pbuf = np.zeros((2, N // 2 + 10), np.float32)
    hbuf = np.zeros((2, N // 2 + 10), np.float32)
    nv = len(voicings[0])
    for vi in range(nv):
        i = 0
        while i < len(plan):
            note = voicings[i][vi]
            j = i
            while j + 1 < len(plan) and voicings[j + 1][vi] == note:
                j += 1                                          # tie common tones
            st = plan[i][1] * beat + (rng.uniform(0, 0.12) if i > 0 else rng.uniform(0, 0.4))
            en = (plan[j][1] + plan[j][2]) * beat + 0.15
            Lp, Rp = pad_note(m2f(note), en - st, cfg["att"], cfg["rel"], cfg["fc"], psr, rng,
                              amp=0.22 * (1 - 0.012 * (note - 60)))
            add_at(pbuf, st * psr, Lp, Rp)
            if active("pad_hi") and vi >= nv - 2:
                Lh, Rh = pad_note(m2f(note + 12), en - st, cfg["att"] * 1.3, cfg["rel"], cfg["fc"] * 1.6,
                                  psr, rng, detunes=(-9, 9), pans=(-0.9, 0.9), amp=0.16)
                add_at(hbuf, st * psr, Lh, Rh)
            i = j + 1
    buf["pad"] = signal.resample_poly(pbuf, 2, 1, axis=1)[:, :N].astype(np.float32)
    buf["pad_hi"] = signal.resample_poly(hbuf, 2, 1, axis=1)[:, :N].astype(np.float32)

    def chord_at(beatpos):
        for k, (cn, sb, L) in enumerate(plan):
            if sb <= beatpos < sb + L:
                return k
        return len(plan) - 1

    # ---- sub
    if active("sub"):
        for k, (cn, sb, L) in enumerate(plan):
            s = sub_note(m2f(basses[k]), L * beat, rng) * 0.32
            add_at(buf["sub"], sb * beat * SR, s)

    # ---- pulse
    if active("pulse") and cfg.get("pulse"):
        pat, div = cfg["pulse"], cfg["pulse_div"]
        nsteps = int(min(n_beats, cfg.get("pulse_beats", n_beats)) * div)
        ramp_b = cfg.get("pulse_ramp_beats", 4)
        for s in range(nsteps):
            bp = s / div
            vel = pat[s % len(pat)]
            if vel <= 0:
                continue
            vel *= rng.uniform(0.9, 1.05) * (0.5 - 0.5 * np.cos(np.pi * min(1.0, (bp + 1) / ramp_b)))
            k = chord_at(bp)
            p = pulse_note(m2f(basses[k] + 12 if basses[k] < 40 else basses[k]), vel * 0.42)
            add_at(buf["pulse"], (bp * beat + rng.normal(0, 0.004)) * SR, p)

    # ---- arp / droplets
    arp = cfg.get("arp")
    if active("arp") and arp is not None:
        if arp == "droplets":
            nb = n_beats // 4
            for bar in range(nb):
                if bar == 0 and col == 0:
                    continue
                if rng.random() < cfg["drop_p"] or bar == 1:
                    bp = bar * 4 + rng.choice([0, 1.5, 2, 3])
                    k = chord_at(bp)
                    cand = [m for m in range(74, 87) if m % 12 in CH[plan[k][0]][1]]
                    note = int(rng.choice(cand))
                    p = pluck_note(m2f(note), rng.uniform(0.25, 0.4), 3000, 1.3, rng, length=4.0)
                    pa = rng.uniform(-0.6, 0.6)
                    gl, gr = pan_gains(pa)
                    add_at(buf["arp"], (bp * beat + rng.normal(0, 0.01)) * SR, gl * p, gr * p)
        elif arp == "ending":
            seq = [(0, 4), (1, 2), (2, 1), (4, 0), (5, 3), (6, 2), (8, 4), (10, 2), (12, 1)]
            for bp, idx in seq:
                if bp >= n_beats - 2:
                    break
                k = chord_at(bp)
                notes = sorted(m for m in range(62, 82) if m % 12 in CH[plan[k][0]][1])
                note = notes[min(idx, len(notes) - 1)]
                vel = 0.5 * (1 - bp / 16)
                p = pluck_note(m2f(note), vel, cfg["arp_fc"], 1.1, rng, length=3.5)
                gl, gr = pan_gains(rng.uniform(-0.35, 0.35))
                add_at(buf["arp"], (bp * beat + rng.normal(0, 0.008)) * SR, gl * p, gr * p)
        else:
            accent = [1, .62, .8, .62, .92, .62, .8, .62]
            entering = prev_type is not None and (LEVELS["arp"][TYPES.index(prev_type)] is None
                                                  or SEC[prev_type].get("arp") == "droplets")
            for s in range(n_beats * 2):
                bp = s / 2
                k = chord_at(bp)
                notes = sorted(m for m in range(62, 82) if m % 12 in CH[plan[k][0]][1])
                idx = arp[s % len(arp)] % len(notes)
                vel = accent[s % 8] * rng.uniform(0.85, 1.05) * 0.55
                if entering:
                    vel *= 0.5 - 0.5 * np.cos(np.pi * min(1.0, (bp + 1) / 8))
                p = pluck_note(m2f(notes[idx]), vel, cfg["arp_fc"], 0.75, rng)
                pa = 0.3 * np.sin(2 * np.pi * s / 16) + rng.normal(0, 0.05)
                gl, gr = pan_gains(pa)
                add_at(buf["arp"], (bp * beat + rng.normal(0, 0.007)) * SR, gl * p, gr * p)

    # ---- bells
    if active("bell") and cfg.get("bells"):
        if cfg["bells"] == "ending":
            k = len(plan) - 1
            for dt, note in ((0.0, 86), (1.5 * beat, 81), (3.5 * beat, 78)):
                bl = bell_note(m2f(note), 0.22, rng, length=6.0)
                gl, gr = pan_gains(rng.uniform(-0.5, 0.5))
                add_at(buf["bell"], (plan[k][1] * beat + dt) * SR, gl * bl, gr * bl)
        else:
            for k, (cn, sb, L) in enumerate(plan):
                cand = [m for m in range(76, 89) if m % 12 in CH[cn][1]]
                for off in [0] + ([L / 2] if rng.random() < 0.45 else []):
                    bl = bell_note(m2f(int(rng.choice(cand))), rng.uniform(0.15, 0.24), rng)
                    gl, gr = pan_gains(rng.uniform(-0.6, 0.6))
                    add_at(buf["bell"], ((sb + off) * beat + rng.normal(0, 0.01)) * SR, gl * bl, gr * bl)

    # ---- shaker (soft, offbeats)
    if active("shaker"):
        nz = band_noise(int(0.2 * SR) * 64, 5000, 11000, rng)
        for s in range(n_beats * 2):
            if s % 2 == 0:
                continue
            seg = nz[(s % 64) * int(0.2 * SR):][: int(0.2 * SR)].copy()
            t = np.arange(len(seg)) / SR
            e = np.exp(-t / 0.035); na = int(0.006 * SR)
            e[:na] *= smooth_ramp(na)
            vel = rng.uniform(0.5, 0.8) * min(1.0, (s + 1) / 16)
            gl, gr = pan_gains(rng.uniform(-0.4, 0.4))
            add_at(buf["shaker"], (s / 2 * beat + rng.normal(0, 0.006)) * SR,
                   gl * seg * e * vel, gr * seg * e * vel)

    # ---- air
    if active("air"):
        n = int((dur + 2) * SR)
        t = np.arange(n) / SR
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * t / 13 + rng.uniform(0, 6))
        env = np.ones(n, np.float32)
        nf = int(3 * SR); env[:nf] = smooth_ramp(nf); env[-nf:] = smooth_ramp(nf)[::-1]
        a = 0.05 * lfo * env
        add_at(buf["air"], 0, band_noise(n, 4500, 10000, rng) * a, band_noise(n, 4500, 10000, rng) * a)

    # ---- transition swell into next section
    if active("swell"):
        n_up, n_dn = int(3.0 * SR), int(1.0 * SR)
        e = np.concatenate([np.linspace(0, 1, n_up) ** 3, smooth_ramp(n_dn)[::-1]]).astype(np.float32)
        a = 0.06 * e
        add_at(buf["swell"], (dur - 3.0) * SR, band_noise(len(e), 2500, 9000, rng) * a,
               band_noise(len(e), 2500, 9000, rng) * a)

    # ---- soft picture accents ("title hit"): rising air swell into a warm bloom (sub + bell triad)
    if active("hit"):
        for th in HITS:
            if not (t0 <= th < t0 + dur):
                continue
            rel = th - t0
            k = chord_at(rel / beat)
            n_up, n_dn = int(2.2 * SR), int(0.6 * SR)
            e = np.concatenate([np.linspace(0, 1, n_up) ** 3, smooth_ramp(n_dn)[::-1]]).astype(np.float32)
            a = 0.05 * e
            add_at(buf["hit"], (rel - 2.2) * SR, band_noise(len(e), 2000, 8000, rng) * a,
                   band_noise(len(e), 2000, 8000, rng) * a)
            n = int(4.0 * SR); t = np.arange(n, dtype=np.float32) / SR
            fb = m2f(basses[k] + 12 if basses[k] < 40 else basses[k])
            bloom = (np.sin(2 * np.pi * fb * t) + 0.5 * np.sin(4 * np.pi * fb * t)) * np.exp(-t / 1.1)
            na = int(0.05 * SR); bloom[:na] *= smooth_ramp(na)
            nr = int(0.5 * SR); bloom[-nr:] *= smooth_ramp(nr)[::-1]
            add_at(buf["hit"], rel * SR, 0.35 * bloom.astype(np.float32))
            tones = sorted(m for m in range(74, 91) if m % 12 in CH[plan[k][0]][1])[:3]
            for j, m in enumerate(tones):
                bl = bell_note(m2f(m), 0.14, rng, length=5.0)
                gl, gr = pan_gains((j - 1) * 0.5)
                add_at(buf["hit"], (rel + 0.012 * j) * SR, gl * bl, gr * bl)

    info = dict(name=name, start=t0, dur=dur, bpm=round(bpm, 3), beats=n_beats,
                chords=[(c, round(t0 + sb * beat, 2), L) for c, sb, L in plan])
    return buf, voicings[-1], basses[-1], info, bpm


def gain_curve(layer, cols, starts, durs, t_abs):
    """Per-sample linear gain for a layer: each section's level (layer dB + section ramp) blended
    with a raised-cosine partition of unity (3 s wide) around every section boundary."""
    W = 3.0
    step = lambda b: (0.5 - 0.5 * np.cos(np.pi * np.clip((t_abs - (b - W / 2)) / W, 0, 1)))
    g = np.zeros(len(t_abs), np.float32)
    for i, c in enumerate(cols):
        if LEVELS[layer][c] is None:
            continue
        a, z = SECTION_DB[TYPES[c]]
        frac = np.clip((t_abs - starts[i]) / durs[i], 0, 1)
        lv = 10 ** ((LEVELS[layer][c] + a + (z - a) * frac) / 20)
        w = (step(starts[i]) if i > 0 else 1.0) - (step(starts[i + 1]) if i + 1 < len(cols) else 0.0)
        g += (lv * w).astype(np.float32)
    return g


def process(buf, cols, starts, durs, t0, bpm, irs, window=None, arp_mix=0.32):
    """Automation + effects for one section buffer -> stereo mix (post reverb)."""
    N = buf["pad"].shape[1]
    t_abs = t0 + np.arange(N) / SR
    mixed = {}
    for l in LAYERS:
        if not np.any(buf[l]):
            continue
        g = gain_curve(l, cols, starts, durs, t_abs)
        if window is not None:
            g = g * window
        mixed[l] = buf[l] * g
    out = np.zeros((2, N), np.float32)
    send = np.zeros((2, N), np.float32)
    for l, x in mixed.items():
        if l in ("pad", "pad_hi"):
            x = chorus(x, t0)
        if l == "arp":
            x = pingpong(x, 0.75 * 60 / bpm, mix=arp_mix)
        out += x
        send += REVERB_SEND[l] * x
    out += 0.55 * reverb(send, irs)
    return out


def master_eq(x):
    sos = np.vstack([signal.butter(2, 30, "high", fs=SR, output="sos"),
                     peaking(1500, -2.5, 0.6), peaking(250, -1.0, 0.8)])
    return signal.sosfilt(sos, x, axis=1).astype(np.float32)


def peaking(f0, gdb, q):
    A = 10 ** (gdb / 40); w = 2 * np.pi * f0 / SR; al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]; a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1, a[1] / a[0], a[2] / a[0]]])


# ----------------------------------------------------------------------------- io / loudness
def write_wav24(path, x):
    x = np.clip(x, -1, 1)
    a = np.ascontiguousarray(np.round(x.T * 8388607).astype("<i4"))
    b = a.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR); w.writeframes(b)


def measure(x):
    """Integrated loudness (LUFS) and true peak (dBTP) via ffmpeg ebur128."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    write_wav24(path, x)
    r = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    os.unlink(path)
    I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r)[-1])
    tp = float(re.findall(r"Peak:\s+(-?[\d.inf]+) dBFS", r)[-1])
    return I, tp


def fade_out_db(n, db=-60):
    """dB-linear fade (sounds natural) ending in a short cosine to true zero."""
    g = 10 ** (np.linspace(0, db, n) / 20)
    k = min(n, int(0.5 * SR))
    g[-k:] *= smooth_ramp(k)[::-1]
    return g.astype(np.float32)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--lufs", type=float, default=-20.0)
    ap.add_argument("--max-tp", type=float, default=-3.0)
    ap.add_argument("--no-loop", action="store_true")
    args = ap.parse_args()

    bpm_nom, durs, types, labels = 88.0, list(DEFAULT_DURS), list(NAMES), list(NAMES)
    TAIL_S = 0.0    # extra digital silence after the fade (file length = sum(durations) + tail)
    cfgp = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sections.json")
    if os.path.exists(cfgp):
        c = json.load(open(cfgp))
        if isinstance(c, list):
            durs = [float(v) for v in c]
        else:
            bpm_nom = float(c.get("bpm", bpm_nom))
            s = c.get("sections", c)
            if isinstance(s, dict):
                durs = [float(s[n]) for n in NAMES]
            elif s and isinstance(s[0], dict):          # [{"type":..., "dur":..., "label":...}, ...]
                types = [e["type"] for e in s]
                durs = [float(e["dur"]) if "dur" in e else float(e["end"]) - float(e["start"]) for e in s]
                labels = [e.get("label", e["type"]) for e in s]
            else:
                durs = [float(v) for v in s]
            HITS[:] = [float(h) for h in c.get("hits", [])]
            TAIL_S = float(c.get("tail", 0.0))
    if args.bpm:
        bpm_nom = args.bpm
    assert len(durs) == len(types), "section count mismatch"
    assert all(t in TYPES for t in types), f"unknown section type; use {TYPES}"
    rng = np.random.default_rng(args.seed)
    irs = make_ir(np.random.default_rng(args.seed + 1))
    os.makedirs(os.path.join(args.outdir, "sections"), exist_ok=True)

    starts = np.concatenate([[0], np.cumsum(durs)[:-1]])
    total = float(sum(durs))
    TAIL = TAIL_S
    NT = int((total + TAIL) * SR)
    full = np.zeros((2, NT), np.float32)
    cols = [TYPES.index(t) for t in types]
    stems, infos = [], []
    v, b = None, None
    for i, name in enumerate(types):
        t0, dur = float(starts[i]), durs[i]
        print(f"[render] {labels[i]:12s} ({name}) {t0:6.1f}s +{dur:.1f}s", flush=True)
        buf, v, b, info, bpm = render_section(name, dur, bpm_nom, t0, rng, v, b, cols[i],
                                              types[i - 1] if i > 0 else None)
        info["label"] = labels[i]
        infos.append(info)
        N = buf["pad"].shape[1]
        am = 0.42 if name in ("results", "discovery") else 0.32
        y = process(buf, cols, starts, durs, t0, bpm, irs, arp_mix=am)
        s0 = int(t0 * SR); n = min(N, NT - s0)
        full[:, s0:s0 + n] += y[:, :n]
        # stem: same automation, spill faded after section end, 2 s tail
        w = np.ones(N, np.float32)
        e = int(dur * SR); nf = int(1.6 * SR)
        w[e:e + nf] = smooth_ramp(nf)[::-1]; w[e + nf:] = 0
        ys = process(buf, cols, starts, durs, t0, bpm, irs, window=w, arp_mix=am)[:, : int((dur + 2.0) * SR)]
        stems.append(ys)

    # master automation: 1.2 s fade-in; ending fades (dB-linear) from the resolving chord to silence at end
    t_abs = np.arange(NT) / SR
    gm = np.ones(NT, np.float32)
    ni = int(1.2 * SR); gm[:ni] = smooth_ramp(ni)
    L = len(types) - 1
    fs = starts[L] + min(durs[L] * 0.55, 8.0)
    a, z = int(fs * SR), int(total * SR)
    gm[a:z] = fade_out_db(z - a, -48); gm[z:] = 0
    full *= gm
    stems[0][:, :ni] *= smooth_ramp(ni)
    s5 = int(starts[L] * SR)
    gme = np.concatenate([gm, np.zeros(stems[L].shape[1], np.float32)])
    stems[L] *= gme[s5:s5 + stems[L].shape[1]]

    full = master_eq(full)
    stems = [master_eq(s) for s in stems]
    I, tp = measure(full)
    gain_db = args.lufs - I
    if tp + gain_db > args.max_tp:
        gain_db = args.max_tp - tp - 0.1
    g = 10 ** (gain_db / 20)
    full *= g
    I2, tp2 = measure(full)
    print(f"[full] loudness {I2:.2f} LUFS, true peak {tp2:.2f} dBTP (gain {gain_db:+.2f} dB)")
    write_wav24(os.path.join(args.outdir, "bgm_full.wav"), full)
    for i, s in enumerate(stems):
        s = s * g
        nf = int(0.3 * SR); s[:, -nf:] *= smooth_ramp(nf)[::-1]
        ni2 = int(0.02 * SR); s[:, :ni2] *= smooth_ramp(ni2)
        write_wav24(os.path.join(args.outdir, "sections", f"{i + 1:02d}_{labels[i]}.wav"), s)
        infos[i]["stem_peak_dbfs"] = round(float(20 * np.log10(np.max(np.abs(s)) + 1e-12)), 2)

    loop_info = None
    if not args.no_loop:
        print("[render] loop", flush=True)
        dur = 30.0
        buf, _, _, linfo, lbpm = render_section("loop", dur, 88.0, 0.0, np.random.default_rng(args.seed + 2),
                                                None, None, 6)
        y = process(buf, [6], [0.0], [dur], 0.0, lbpm, irs)
        n = int(dur * SR)
        loop = y[:, :n].copy()
        spill = y[:, n:]
        while spill.shape[1] > 0:                      # circular wrap -> seamless loop
            k = min(n, spill.shape[1]); loop[:, :k] += spill[:, :k]; spill = spill[:, k:]
        loop = master_eq(np.concatenate([loop, loop, loop], 1))[:, n:2 * n]   # steady-state EQ
        Il, tpl = measure(np.concatenate([loop, loop], 1))
        gl_db = min(args.lufs - Il, args.max_tp - tpl - 0.1)
        loop *= 10 ** (gl_db / 20)
        write_wav24(os.path.join(args.outdir, "bgm_loop.wav"), loop)
        Il2, tpl2 = measure(np.concatenate([loop, loop], 1))
        print(f"[loop] {Il2:.2f} LUFS, {tpl2:.2f} dBTP, bpm {lbpm:.2f}")
        loop_info = dict(dur=30.0, bpm=lbpm, lufs=Il2, true_peak=tpl2, chords=linfo["chords"])

    timing = dict(bpm_nominal=bpm_nom, total_music=total, file_len=total + TAIL,
                  lufs=I2, true_peak_dbtp=tp2, sections=infos, loop=loop_info)
    json.dump(timing, open(os.path.join(args.outdir, "timing.json"), "w"), indent=1)
    print("[done]", args.outdir)


if __name__ == "__main__":
    main()
