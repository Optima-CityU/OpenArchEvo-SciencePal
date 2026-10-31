# BGM: "science discovery" bed for the OpenArchEvo / SciencePal hackathon video

Fully procedural (numpy/scipy only, no samples, no downloaded audio). The composition and code are original, so the track is royalty-free.
Key: **D major / B minor**. Tempo: **88 BPM** nominal. Each section adjusts its tempo by up to about ±3%, which lets it hold a whole number of beats and so start on a downbeat.

## Files
| file | what |
|---|---|
| `bgm_full.wav` | full mix, 227.5 s, 48 kHz / 24-bit stereo, **-20.0 LUFS integrated, -7.1 dBTP true peak**, fades to silence by 227.5 s |
| `sections/NN_<label>.wav` | one stem per section, same gain as the full mix. Each has section length + 2 s tail: notes that spill past the end fade over 1.6 s, then the reverb decays. A 20 ms fade-in and a 0.3 s fade-out are applied |
| `bgm_loop.wav` | 30.0 s seamless loop bed (88 BPM, 11 bars, D-Bm-G-Asus4-A). It is rendered circularly: the reverb and release tail is wrapped onto the start. -20 LUFS |
| `compose.py` | the generator. `sections.json` is its default config |
| `timing.json` | the rendered timeline: section starts, per-section BPM, and the chord at every change (absolute seconds) |
| `analyze.py` | runs the technical checks and writes `spectrogram.png`, `checks_seam_ending.png` and `analysis.json` |

## Structure (final timeline, `sections.json`)
| time (s) | section (type) | music |
|---|---|---|
| 0.0-15.9 | intro | sparse airy pad (slow 3 s attacks, dark filter), occasional high "droplet" plucks, faint air noise. Dmaj9 - Bm11 - Asus4 |
| 15.9-39.2 | bigpicture | gentle swell (+2.5 dB ramp across the section), a high pad layer, soft sub enters, the bass rises G-A-B (progress). Gmaj9 - A - Bm9 - Asus4 |
| 39.2-77.9 | breeders | a soft low pulse (quarters) fades in over about 20 beats. Bm9 - Gmaj7 - Dmaj9 - Asus4 - Bm9 - Gmaj7 - A |
| **77.9** | title hit | a rising air swell into a warm bloom (sub plus a bell triad) on the **V -> I** arrival at Dmaj9 |
| 77.9-87.0 | title | the pulse drops out (a breath), Dmaj9 -> Gmaj9, droplets |
| 87.0-137.2 | method | steady 8th-note plucked arpeggio with a dotted-8th ping-pong echo, light pulse. I-vi-IV-V: Dadd9 - Bm7 - Gmaj7 - Asus4 (x2) - A7sus4 |
| 137.2-173.8 | results | the gentle lift (loudest part, about +3 dB vs method): 8th-note pulse, second pad octave, very soft offbeat shaker. Gmaj7 - A - F#m7 - Bm9 - Em9 - Gmaj7 - A |
| 173.8-218.3 | discovery | warm and brighter (pad filter opened up, sparse FM bells, air). Gmaj9 - D/F# - Em9 - Asus4 - Gmaj7 - A - Bm9 - A7sus4 |
| 218.3-227.5 | ending | Gmaj7 - A7sus4 -> **Dmaj9 resolves at 222.5 s**, with a bell arpeggio. A dB-linear fade starts at 223.4 and reaches silence at 227.5. The visual ending is 218.3-225.1 |

Transitions: the layer levels crossfade over a 3 s raised-cosine window centred on every boundary. Most boundaries also get a soft rising noise swell (2.5-9 kHz, quiet) in the last 3 s before them.

## Craft details
- **Harmony and voice leading:** each chord gets a 5-note voicing in A3-G5, chosen to minimise total voice motion from the previous chord. The voicing has no minor seconds and wider spacing below middle C. Common tones are **tied**, so they are not re-attacked. Bass is a separate voice (B1-B2).
- **Pad:** additive band-limited saw stacks (3 oscillators at ±7 cents, panned L/C/R), with a slowly breathing low-pass (LFO of ±28% over 7-12 s) and slow vibrato. A stereo chorus follows.
- **Pluck:** additive with partials whose decay speeds up as they go higher, a 9 ms raised-cosine attack and slight inharmonicity. Timing is humanised (σ ≈ 7 ms) and velocity varies ±10%.
- **Envelopes:** every note has a raised-cosine attack and release, with no step edges anywhere, so there are no clicks.
- **Reverb:** a synthetic stereo hall impulse response (decorrelated noise, RT60 of 2.9 s low to 1.0 s high, 22 ms pre-delay, early reflections), applied by convolution with per-layer sends.
- **Under-voice mix:** no lead melody. The master EQ is HPF 30 Hz, -2.5 dB at 1.5 kHz (Q 0.6) and -1 dB at 250 Hz. The 1-4 kHz band sits roughly 25-40 dB below the 150-500 Hz pad body (see `spectrogram.png`). Expect to duck it a further 6-10 dB under the voiceover in the editor, as usual.

## Technical checks (`analysis.json`)
- **Full mix:** -20.0 LUFS integrated, LRA 6.1 LU, true peak -7.1 dBTP, sample peak -7.1 dBFS.
- **Clean signal:** 0 clipped samples, and the DC offset is about 1e-11.
- **Section RMS** (dBFS): intro -27.8, bigpicture -25.1, breeders -24.8, title -23.7, method -23.5, results -20.3, discovery -20.2, ending -24.1. That is the intended gentle build into results.
- **Click detector:** this is the ratio of a 1 ms burst of >10 kHz energy to the local median. It reads ≤ 2.8 everywhere except results (about 19). That one comes from the deliberate soft shaker, and its burst sits at -43 dBFS, about 35 dB under the programme.
- **Loop seam:** the sample jump at the seam is 0.56x a typical 99.9th-percentile sample step. RMS is -23.7 dB before the seam and -23.0 dB after it, so the seam is seamless.

## Re-render with different section lengths
Edit `sections.json`, or pass `--config`. Section `type` is one of `intro, bigpicture, breeders, title, method, results, discovery, ending`. Types can be repeated, reordered or dropped. Give either `start`/`end` or `dur`.
```json
{"bpm": 88, "hits": [77.9], "tail": 0.0,
 "sections": [{"label": "intro", "type": "intro", "start": 0.0, "end": 15.9},
              {"label": "bigpicture", "type": "bigpicture", "dur": 23.3}, ...]}
```
```
python3 compose.py --config sections.json [--outdir DIR] [--seed N] [--lufs -20] [--max-tp -3] [--no-loop]
python3 analyze.py [DIR]      # regenerate checks + spectrogram
```
- Sections are laid end to end, so the music ends exactly at the sum of the durations. The fade-out completes at that point.
- `hits` takes absolute times for soft picture accents (a swell into a bloom).
- `tail` adds extra silence after the fade.
- Legacy forms also work: `[25,30,45,50,35,15]`, or `{"sections": {"intro":25,...}}` for the 6 original types.
- A render takes about 1 minute. Results are deterministic for a given `--seed`.
- To change the musical design, edit `SEC` (progressions, harmonic rhythm, patterns), `LEVELS` (per-layer dB per section type) and `SECTION_DB` (per-section dB ramp) in `compose.py`.
