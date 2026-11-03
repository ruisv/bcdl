#!/usr/bin/env python3
"""A voice-controlled robot, end to end on the board:

    wake word ──> listen ──> one utterance ──> text ──> command ──> spoken reply
    KeywordSpotter   VadSegmenter   ParaformerAsr   CommandMatcher   TextToSpeech

Every network runs on the BPU. Audio is fed in 100 ms blocks, the way a
microphone delivers it, so the loop below is the loop you would run on a live
stream; here it reads a recording and writes each spoken reply to a .wav.

    python examples/voice_assistant.py                  # no recording? it makes one
    python examples/voice_assistant.py in.wav           # 16 kHz mono PCM16
    python examples/voice_assistant.py in.wav --commands 前进,后退,停下 \\
        --keywords "n ǐ h ǎo x iǎo ch ē @你好小车"

With no input it first synthesises a short session with TextToSpeech — speech
without the wake word, two commands after it, one request that is not a
command — writes it to <out>/demo_input.wav, and runs the loop on that.

Models: scripts/fetch_models.sh (into models/, or point --models elsewhere).
Capture and playback are not part of BCDL: replace read_blocks() with your
microphone and the .wav writes with your speaker. While the robot is speaking,
stop feeding the microphone to the VAD, or it will hear itself.
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

import bcdl

SR = 16000
BLOCK = SR // 10                       # 100 ms, a typical microphone period
LISTEN_TIMEOUT_S = 5.0                 # back to sleep if nothing is said after the wake word

DEFAULT_KEYWORDS = ("n ǐ h ǎo x iǎo ch ē @你好小车\n"
                    "x iǎo ch ē x iǎo ch ē @小车小车\n")
DEFAULT_COMMANDS = ["前进", "后退", "向左转", "向右转", "停下", "打开灯", "关灯"]

FILES = {
    "kws": ["kws_zipformer_zh_en_3m_c16_int16_nashm.hbm",
            "kws_zipformer_zh_en_3m_decoder_joiner.npy",
            "kws_zipformer_zh_en_3m_tokens.txt"],
    "vad": ["silero_vad_16k_int16_nashm.hbm"],
    "asr": ["paraformer_encoder_int16.hbm", "paraformer_predictor_int16_v2.hbm",
            "paraformer_decoder_int16_v2.hbm", "paraformer_tokens_8404.json",
            "paraformer_am.mvn"],
    "tts": ["matcha_zh_en_enc_l128_int16_nashm.hbm", "matcha_zh_en_dec_t512_int16_nashm.hbm",
            "matcha_zh_en_embedding.npy", "vocos_16k_t256_int16_nashm.hbm",
            "matcha_zh_en_tokens.txt", "matcha_zh_en_lexicon.txt"],
}


# --- audio I/O (replace with your microphone / speaker) ----------------------
def read_wav(path):
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16-bit PCM")
        if w.getframerate() != SR:
            sys.exit(f"{path}: need {SR} Hz, got {w.getframerate()} "
                     f"(resample first, e.g. ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)")
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def read_blocks(x):
    """A recording, delivered the way a microphone would: BLOCK samples at a time."""
    for i in range(0, len(x), BLOCK):
        yield i / SR, x[i:i + BLOCK]


# --- the loop ---------------------------------------------------------------
def load(models, keywords):
    p = {k: [os.path.join(models, f) for f in v] for k, v in FILES.items()}
    missing = [f for v in p.values() for f in v if not os.path.exists(f)]
    if missing:
        sys.exit("missing models (run scripts/fetch_models.sh):\n  " + "\n  ".join(missing))
    t0 = time.perf_counter()
    kws = bcdl.KeywordSpotter(*p["kws"], keywords)
    vad = bcdl.VadSegmenter(p["vad"][0])
    asr = bcdl.ParaformerAsr.from_files(*p["asr"])
    tts = bcdl.TextToSpeech.from_files(*p["tts"])
    print(f"models loaded in {time.perf_counter() - t0:.1f} s; wake words: {kws.keywords}")
    return kws, vad, asr, tts


def demo_session(tts):
    """A short session in the robot's own voice: what the loop should ignore, two
    commands after a wake word, and a request it cannot match."""
    script = [(1.0, "今天天气不错。"), (1.0, "你好小车。"), (0.6, "向左转。"),
              (1.5, "小车，小车。"), (0.6, "打开灯。"), (1.5, "你好小车。"),
              (0.6, "给我唱首歌吧。"), (1.5, None)]
    parts = []
    for gap, text in script:
        parts.append(np.zeros(int(gap * SR), np.float32))
        if text:
            parts.append(tts.synthesize(text).pcm)
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav", nargs="?", help="16 kHz mono PCM16 recording (default: synthesise one)")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--out", default="voice_out", help="where the replies (.wav) go")
    ap.add_argument("--keywords", default=DEFAULT_KEYWORDS,
                    help="keyword lines, or a file of them (sherpa-onnx format, see docs/API.md)")
    ap.add_argument("--commands", default=",".join(DEFAULT_COMMANDS),
                    help="comma-separated command list")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kws, vad, asr, tts = load(args.models, args.keywords)
    commands = [c for c in args.commands.split(",") if c]
    matcher = bcdl.CommandMatcher(commands)

    if args.wav:
        audio = read_wav(args.wav)
    else:
        audio = demo_session(tts)
        write_wav(os.path.join(args.out, "demo_input.wav"), audio)
        print(f"synthesised a {len(audio) / SR:.1f} s demo session -> {args.out}/demo_input.wav")

    n_reply = 0

    def say(text, at):
        nonlocal n_reply
        t0 = time.perf_counter()
        pcm = tts.synthesize(text).pcm
        n_reply += 1
        path = os.path.join(args.out, f"reply_{n_reply:02d}.wav")
        write_wav(path, pcm)
        print(f"  {at:6.2f} s  robot: {text}  ({(time.perf_counter() - t0) * 1000:.0f} ms to "
              f"synthesise {len(pcm) / SR:.1f} s -> {path})")

    # KeywordSpotter and VadSegmenter report times from their last reset; a
    # reset made while handling the block at `now` takes effect with the next one.
    awake, kws_from, listen_from = False, 0.0, 0.0
    step = BLOCK / SR
    print(f"\n{'time':>8}  event")
    for now, block in read_blocks(audio):
        if not awake:
            for d in kws.feed(block):                 # sleeping: only the wake word matters
                print(f"  {kws_from + d.end:6.2f} s  wake word '{d.keyword}'")
                say("我在。", now)
                awake, listen_from = True, now + step
                vad.reset()
            continue
        for seg in vad.feed(block):                   # awake: wait for one whole utterance
            t0 = time.perf_counter()
            text = asr.transcribe(seg.pcm).text
            t_asr = (time.perf_counter() - t0) * 1000
            start = listen_from + seg.start / SR
            end = listen_from + seg.end / SR
            m = matcher.match(text)
            print(f"  {start:6.2f} s  heard '{text}' ({end - start:.1f} s, {t_asr:.0f} ms to recognise)"
                  f" -> {('command: ' + m.command) if m else 'not a command'}")
            say(f"好的，{m.command}。" if m else "没听懂，请再说一遍。", now)
            awake, kws_from = False, now + step
            kws.reset()                               # back to sleep, fresh stream
            break
        if awake and not vad.in_speech and now - listen_from > LISTEN_TIMEOUT_S:
            print(f"  {now:6.2f} s  nothing said after the wake word; going back to sleep")
            awake, kws_from = False, now + step
            kws.reset()
    # end of the recording: let a wake word or utterance right at the end through
    for d in (kws.flush() if not awake else []):
        print(f"  {kws_from + d.end:6.2f} s  wake word '{d.keyword}' (at the very end; nothing follows)")
    for seg in (vad.flush() if awake else []):
        print(f"  heard '{asr.transcribe(seg.pcm).text}' (cut off by the end of the recording)")
    print(f"\n{n_reply} replies in {args.out}/")


if __name__ == "__main__":
    main()
