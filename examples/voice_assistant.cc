// A voice-controlled robot, end to end on the board (C++ counterpart of
// examples/voice_assistant.py):
//
//   wake word -> listen -> one utterance -> text -> command -> spoken reply
//   KeywordSpotter  VadSegmenter  ParaformerAsr  CommandMatcher  TextToSpeech
//
// Every network runs on the BPU. Audio is fed in 100 ms blocks, the way a
// microphone delivers it; here it comes from a recording and every reply is
// written to a .wav.
//
//   ./build/voice_assistant [in.wav] [--models DIR] [--out DIR]
//
// in.wav must be 16 kHz mono PCM16. Without it, a short demo session is
// synthesised first (written to <out>/demo_input.wav). Models come from
// scripts/fetch_models.sh. Capture and playback are not part of BCDL: replace
// the recording with your microphone and the .wav writes with your speaker,
// and stop feeding the VAD while the robot speaks so it does not hear itself.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bcdl/bcdl.h"

namespace {

constexpr int kRate = 16000;
constexpr int kBlock = kRate / 10;            // 100 ms
constexpr double kListenTimeout = 5.0;        // seconds of silence after the wake word

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// --- 16-bit PCM WAV (replace with your microphone / speaker) -----------------
std::vector<float> readWav(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char riff[12];
  f.read(riff, 12);
  if (std::strncmp(riff, "RIFF", 4) || std::strncmp(riff + 8, "WAVE", 4))
    throw std::runtime_error(path + " is not a WAV file");
  std::uint16_t channels = 0, bits = 0;
  std::uint32_t rate = 0;
  for (char id[4]; f.read(id, 4);) {
    std::uint32_t size = 0;
    f.read(reinterpret_cast<char*>(&size), 4);
    if (!std::strncmp(id, "fmt ", 4)) {
      std::vector<char> b(size);
      f.read(b.data(), size);
      std::memcpy(&channels, &b[2], 2);
      std::memcpy(&rate, &b[4], 4);
      std::memcpy(&bits, &b[14], 2);
    } else if (!std::strncmp(id, "data", 4)) {
      if (bits != 16 || rate != kRate)
        throw std::runtime_error(path + ": need 16 kHz 16-bit PCM (resample first, e.g. "
                                        "ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)");
      std::vector<std::int16_t> s(size / 2);
      f.read(reinterpret_cast<char*>(s.data()), size);
      std::vector<float> x(s.size() / channels);
      for (std::size_t i = 0; i < x.size(); ++i) {
        float acc = 0;
        for (int c = 0; c < channels; ++c) acc += s[i * channels + c];
        x[i] = acc / channels / 32768.0f;
      }
      return x;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);
    }
  }
  throw std::runtime_error(path + ": no data chunk");
}

void writeWav(const std::string& path, const std::vector<float>& pcm) {
  std::ofstream f(path, std::ios::binary);
  const std::uint32_t data = static_cast<std::uint32_t>(pcm.size() * 2), riff = 36 + data,
                      fmt = 16, rate = kRate, byte_rate = kRate * 2;
  const std::uint16_t pcm_tag = 1, ch = 1, align = 2, bits = 16;
  f.write("RIFF", 4); f.write(reinterpret_cast<const char*>(&riff), 4); f.write("WAVE", 4);
  f.write("fmt ", 4); f.write(reinterpret_cast<const char*>(&fmt), 4);
  f.write(reinterpret_cast<const char*>(&pcm_tag), 2); f.write(reinterpret_cast<const char*>(&ch), 2);
  f.write(reinterpret_cast<const char*>(&rate), 4); f.write(reinterpret_cast<const char*>(&byte_rate), 4);
  f.write(reinterpret_cast<const char*>(&align), 2); f.write(reinterpret_cast<const char*>(&bits), 2);
  f.write("data", 4); f.write(reinterpret_cast<const char*>(&data), 4);
  for (float v : pcm) {
    const std::int16_t s = static_cast<std::int16_t>(std::max(-1.0f, std::min(1.0f, v)) * 32767);
    f.write(reinterpret_cast<const char*>(&s), 2);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string wav, models = "models", out = "voice_out";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--models" && i + 1 < argc) models = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [in.wav] [--models DIR] [--out DIR]\n", argv[0]);
      return 0;
    } else wav = a;
  }
  const std::string m = models + "/";
  try {
    // --- load (each Engine is one compiled model) ------------------------------
    auto t0 = std::chrono::steady_clock::now();
    bcdl::Engine kws_enc(m + "kws_zipformer_zh_en_3m_c16_int16_nashm.hbm");
    bcdl::KeywordSpotter kws(kws_enc, m + "kws_zipformer_zh_en_3m_decoder_joiner.npy",
                             m + "kws_zipformer_zh_en_3m_tokens.txt");
    kws.setKeywords("n ǐ h ǎo x iǎo ch ē @你好小车\n"
                    "x iǎo ch ē x iǎo ch ē @小车小车\n");

    bcdl::Engine vad_eng(m + "silero_vad_16k_int16_nashm.hbm");
    bcdl::SileroVad silero(vad_eng);
    bcdl::VadSegmenter vad(silero);

    bcdl::Engine asr_enc(m + "paraformer_encoder_int16.hbm");
    bcdl::Engine asr_pred(m + "paraformer_predictor_int16_v2.hbm");
    bcdl::Engine asr_dec(m + "paraformer_decoder_int16_v2.hbm");
    bcdl::ParaformerAsr asr(asr_enc, asr_pred, asr_dec,
                            bcdl::loadTokenList(m + "paraformer_tokens_8404.json"),
                            bcdl::loadKaldiCmvn(m + "paraformer_am.mvn"));

    const std::vector<std::string> commands = {"前进", "后退", "向左转", "向右转",
                                               "停下", "打开灯", "关灯"};
    bcdl::CommandMatcher matcher(commands);

    bcdl::Engine tts_enc(m + "matcha_zh_en_enc_l128_int16_nashm.hbm");
    bcdl::Engine tts_dec(m + "matcha_zh_en_dec_t512_int16_nashm.hbm");
    bcdl::Engine voc_eng(m + "vocos_16k_t256_int16_nashm.hbm");
    bcdl::MatchaAcoustic acoustic(tts_enc, tts_dec, m + "matcha_zh_en_embedding.npy");
    bcdl::VocosVocoder vocoder(voc_eng);
    bcdl::TextToSpeech tts(bcdl::MatchaFrontend(m + "matcha_zh_en_tokens.txt",
                                                m + "matcha_zh_en_lexicon.txt"),
                           acoustic, vocoder);
    std::printf("models loaded in %.1f s\n", msSince(t0) / 1000);

    std::filesystem::create_directories(out);
    std::vector<float> audio;
    if (!wav.empty()) {
      audio = readWav(wav);
    } else {
      // The robot's own voice: speech to ignore, two commands, one request that
      // is not a command.
      const std::vector<std::pair<double, std::string>> script = {
          {1.0, "今天天气不错。"}, {1.0, "你好小车。"}, {0.6, "向左转。"},
          {1.5, "小车，小车。"}, {0.6, "打开灯。"}, {1.5, "你好小车。"},
          {0.6, "给我唱首歌吧。"}, {1.5, ""}};
      for (const auto& s : script) {
        audio.insert(audio.end(), static_cast<std::size_t>(s.first * kRate), 0.0f);
        if (!s.second.empty()) {
          const auto r = tts.synthesize(s.second);
          audio.insert(audio.end(), r.pcm.begin(), r.pcm.end());
        }
      }
      writeWav(out + "/demo_input.wav", audio);
      std::printf("synthesised a %.1f s demo session -> %s/demo_input.wav\n",
                  audio.size() / double(kRate), out.c_str());
    }

    int replies = 0;
    auto say = [&](const std::string& text, double now) {
      auto t = std::chrono::steady_clock::now();
      const auto r = tts.synthesize(text);
      const std::string path = out + "/reply_" + (replies < 9 ? "0" : "") +
                               std::to_string(++replies) + ".wav";
      writeWav(path, r.pcm);
      std::printf("  %6.2f s  robot: %s  (%.0f ms to synthesise %.1f s -> %s)\n", now,
                  text.c_str(), msSince(t), r.pcm.size() / double(kRate), path.c_str());
    };

    // Detections and segments are timed from their last reset, which takes
    // effect with the block after the one being handled.
    bool awake = false;
    double kws_from = 0, listen_from = 0;
    const double step = double(kBlock) / kRate;
    std::printf("\n%8s  event\n", "time");
    for (std::size_t i = 0; i < audio.size(); i += kBlock) {
      const double now = double(i) / kRate;
      const float* block = audio.data() + i;
      const std::size_t n = std::min<std::size_t>(kBlock, audio.size() - i);
      if (!awake) {
        for (const auto& d : kws.feed(block, n)) {           // sleeping: wake word only
          std::printf("  %6.2f s  wake word '%s'\n", kws_from + d.end, d.keyword.c_str());
          say("我在。", now);
          awake = true;
          listen_from = now + step;
          vad.reset();
        }
        continue;
      }
      auto segs = vad.feed(block, n);                        // awake: one whole utterance
      if (!segs.empty()) {
        const auto& s = segs.front();
        auto t = std::chrono::steady_clock::now();
        const std::string text = asr.transcribe(s.pcm.data(), s.pcm.size()).text;
        const double t_asr = msSince(t);
        const bcdl::CommandMatch c = matcher.match(text);
        std::printf("  %6.2f s  heard '%s' (%.1f s, %.0f ms to recognise) -> %s%s\n",
                    listen_from + s.start / double(kRate), text.c_str(),
                    (s.end - s.start) / double(kRate), t_asr,
                    c.index >= 0 ? "command: " : "not a command",
                    c.index >= 0 ? c.command.c_str() : "");
        say(c.index >= 0 ? "好的，" + c.command + "。" : "没听懂，请再说一遍。", now);
        awake = false;
        kws_from = now + step;
        kws.reset();
      } else if (!vad.inSpeech() && now - listen_from > kListenTimeout) {
        std::printf("  %6.2f s  nothing said after the wake word; going back to sleep\n", now);
        awake = false;
        kws_from = now + step;
        kws.reset();
      }
    }
    if (!awake)
      for (const auto& d : kws.flush())
        std::printf("  %6.2f s  wake word '%s' (at the very end)\n", kws_from + d.end,
                    d.keyword.c_str());
    std::printf("\n%d replies in %s/\n", replies, out.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n(models come from scripts/fetch_models.sh; --models DIR)\n",
                 e.what());
    return 1;
  }
  return 0;
}
