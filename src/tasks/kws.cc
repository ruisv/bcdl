#include "bcdl/tasks/kws.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "task_util.h"

namespace bcdl {

namespace {

constexpr int kBlank = 0;
constexpr int kGroupIn = 4;   // decoder conv: input channels per group
constexpr int kKernel = 2;    // decoder conv: context size

FbankConfig kwsFbank() {
  // kaldi-native-fbank as sherpa-onnx configures it for icefall models:
  // samples in [-1, 1], Povey window, 20 Hz .. Nyquist - 400 Hz.
  FbankConfig c;
  c.input_scale = 1.0f;
  c.window = "povey";
  c.low_freq = 20.0f;
  c.high_freq = -400.0f;
  return c;
}

double logAdd(double a, double b) {
  const double m = std::max(a, b);
  return m + std::log1p(std::exp(-std::fabs(a - b)));
}

}  // namespace

// ---------------------------------------------------------------------------
// Phrase graph: a trie with Aho-Corasick fail/output links (sherpa-onnx's
// ContextGraph, strict mode).
// ---------------------------------------------------------------------------

struct KeywordSpotter::Graph {
  struct Node {
    int token = -1;
    float token_score = 0, node_score = 0, output_score = 0, ac_threshold = 0;
    int level = 0;
    bool is_end = false;
    int phrase = -1;
    std::unordered_map<int, int> next;
    int fail = 0, output = -1;
  };
  std::vector<Node> n{Node{}};  // node 0 is the root

  void add(const std::vector<int>& ids, float score, float thr, int phrase) {
    int cur = 0;
    for (std::size_t j = 0; j < ids.size(); ++j) {
      const bool last = j + 1 == ids.size();
      auto it = n[cur].next.find(ids[j]);
      if (it == n[cur].next.end()) {
        Node nd;
        nd.token = ids[j];
        nd.token_score = score;
        nd.node_score = n[cur].node_score + score;
        nd.output_score = last ? nd.node_score : 0.0f;
        nd.level = static_cast<int>(j) + 1;
        nd.ac_threshold = last ? thr : 0.0f;
        nd.is_end = last;
        nd.phrase = last ? phrase : -1;
        n.push_back(nd);
        const int id = static_cast<int>(n.size()) - 1;
        n[cur].next[ids[j]] = id;
        cur = id;
      } else {
        Node& nd = n[it->second];
        nd.token_score = std::max(score, nd.token_score);
        nd.node_score = n[cur].node_score + nd.token_score;
        nd.is_end = last || nd.is_end;
        nd.output_score = nd.is_end ? nd.node_score : 0.0f;
        if (last) {
          nd.phrase = phrase;
          nd.ac_threshold = thr;
        }
        cur = it->second;
      }
    }
  }

  void finish() {
    std::deque<int> q;
    for (auto& kv : n[0].next) {
      n[kv.second].fail = 0;
      q.push_back(kv.second);
    }
    while (!q.empty()) {
      const int u = q.front();
      q.pop_front();
      for (auto& kv : n[u].next) {
        const int t = kv.first, v = kv.second;
        int f = n[u].fail;
        while (!n[f].next.count(t)) {
          if (f == 0) break;
          f = n[f].fail;
        }
        auto it = n[f].next.find(t);
        n[v].fail = (it != n[f].next.end() && it->second != v) ? it->second : 0;
        int o = n[v].fail;
        while (!n[o].is_end) {
          if (o == 0) {
            o = -1;
            break;
          }
          o = n[o].fail;
        }
        n[v].output = o;
        if (o >= 0) n[v].output_score += n[o].output_score;
        q.push_back(v);
      }
    }
  }

  /// (score, next state) for `token` from `state`.
  std::pair<float, int> forward(int state, int token) const {
    int node;
    float score;
    auto it = n[state].next.find(token);
    if (it != n[state].next.end()) {
      node = it->second;
      score = n[node].token_score;
    } else {
      node = n[state].fail;
      while (!n[node].next.count(token)) {
        if (node == 0) break;
        node = n[node].fail;
      }
      auto jt = n[node].next.find(token);
      if (jt != n[node].next.end()) node = jt->second;
      score = n[node].node_score - n[state].node_score;
    }
    return {score + n[node].output_score, node};
  }

  /// The end node `state` completes, or -1.
  int matched(int state) const { return n[state].is_end ? state : n[state].output; }
};

struct KeywordSpotter::Hyp {
  std::vector<int> ys;
  double log_prob = 0;
  std::vector<std::int64_t> timestamps;
  std::vector<float> probs;
  int state = 0;
  int trailing = 0;
};

// ---------------------------------------------------------------------------

KeywordSpotter::KeywordSpotter(Engine& encoder, const std::string& decoder_joiner,
                               const std::string& tokens, KwsConfig cfg)
    : enc_(encoder), cfg_(cfg), fbank_(kwsFbank()), graph_(new Graph) {
  if (cfg_.max_active_paths < 1 || cfg_.num_trailing_blanks < 0)
    throw Error(-1, "BCDL KeywordSpotter: max_active_paths must be >= 1, num_trailing_blanks >= 0");
  // tokens
  std::ifstream tf(tokens);
  if (!tf) throw Error(-1, "BCDL KeywordSpotter: cannot open " + tokens);
  for (std::string line; std::getline(tf, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto sp = line.find_last_of(' ');
    if (sp == std::string::npos || sp == 0) continue;
    const std::string id = line.substr(sp + 1);
    if (id.empty() || id.find_first_not_of("0123456789") != std::string::npos || id.size() > 9)
      throw Error(-1, "BCDL KeywordSpotter: bad token id '" + id + "' in " + tokens);
    const int i = std::stoi(id);
    token_[line.substr(0, sp)] = i;
    if (static_cast<int>(inv_.size()) <= i) inv_.resize(i + 1);
    inv_[i] = line.substr(0, sp);
  }
  vocab_ = static_cast<int>(inv_.size());
  if (vocab_ == 0) throw Error(-1, "BCDL KeywordSpotter: no tokens in " + tokens);
  unk_ = tokenId("<unk>");

  // encoder I/O by name
  std::vector<std::string> in, out;
  for (int i = 0; i < enc_.numInputs(); ++i) in.push_back(enc_.inputName(i));
  for (int i = 0; i < enc_.numOutputs(); ++i) out.push_back(enc_.outputName(i));
  for (int i = 0; i < static_cast<int>(in.size()); ++i) {
    if (in[i] == "x") in_x_ = i;
    else if (in[i] == "pad_mask") in_mask_ = i;
  }
  for (int i = 0; i < static_cast<int>(out.size()); ++i)
    if (out[i] == "encoder_out") out_enc_ = i;
  if (in_x_ < 0 || in_mask_ < 0 || out_enc_ < 0)
    throw Error(-1, "BCDL KeywordSpotter: the encoder needs inputs x and pad_mask and output "
                    "encoder_out (a zipformer2 KWS build with the int64 processed_lens "
                    "replaced)");
  for (int i = 0; i < static_cast<int>(in.size()); ++i) {
    if (i == in_x_ || i == in_mask_) continue;
    const auto it = std::find(out.begin(), out.end(), "new_" + in[i]);
    if (it == out.end())
      throw Error(-1, "BCDL KeywordSpotter: encoder input " + in[i] + " has no new_" + in[i] +
                          " output");
    const int o = static_cast<int>(it - out.begin());
    if (enc_.inputShape(i) != enc_.outputShape(o))
      throw Error(-1, "BCDL KeywordSpotter: cache " + in[i] + " is " +
                          detail::shapeStr(enc_.inputShape(i)) + " in, " +
                          detail::shapeStr(enc_.outputShape(o)) + " out");
    caches_.emplace_back(i, o);
  }
  const std::vector<int> xs = enc_.inputShape(in_x_), es = enc_.outputShape(out_enc_),
                         ms = enc_.inputShape(in_mask_);
  if (xs.size() != 3 || xs[0] != 1 || xs[2] != fbank_.config().num_mel_bins || es.size() != 3 ||
      es[0] != 1 || ms.size() != 2 || ms[0] != 1)
    throw Error(-1, "BCDL KeywordSpotter: unexpected encoder shapes: x " + detail::shapeStr(xs) +
                        ", encoder_out " + detail::shapeStr(es) + ", pad_mask " +
                        detail::shapeStr(ms));
  t_in_ = xs[1];
  out_frames_ = es[1];
  dim_ = es[2];
  shift_ = 4 * out_frames_;           // two 2x subsamplings
  mask_len_ = ms[1];
  left_ = mask_len_ - shift_ / 2;     // cached frames at the embedding rate
  if (left_ <= 0 || t_in_ < shift_)
    throw Error(-1, "BCDL KeywordSpotter: inconsistent encoder geometry");

  // decoder + joiner table
  int rows = 0, cols = 0;
  table_ = detail::loadNpy2d(decoder_joiner, rows, cols, "KeywordSpotter");
  if (cols != dim_ + 1 || rows != 2 * vocab_ + 2 * dim_)
    throw Error(-1, "BCDL KeywordSpotter: decoder_joiner table is [" + std::to_string(rows) +
                        ", " + std::to_string(cols) + "], expected [" +
                        std::to_string(2 * vocab_ + 2 * dim_) + ", " + std::to_string(dim_ + 1) +
                        "] for " + std::to_string(vocab_) + " tokens and D = " +
                        std::to_string(dim_));
  zero_.assign(dim_, 0.0f);
  graph_->finish();
  reset();
}

KeywordSpotter::~KeywordSpotter() = default;

int KeywordSpotter::tokenId(const std::string& token) const {
  auto it = token_.find(token);
  return it == token_.end() ? -1 : it->second;
}

void KeywordSpotter::setKeywords(const std::string& text) {
  std::unique_ptr<Graph> g(new Graph);
  std::vector<std::string> phrases;
  std::istringstream in(text);
  int lineno = 0;
  for (std::string line; std::getline(in, line);) {
    ++lineno;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find_first_not_of(" \t") == std::string::npos) continue;
    const auto at = line.find('@');
    std::string phrase = at == std::string::npos ? "" : line.substr(at + 1);
    phrase.erase(0, phrase.find_first_not_of(" \t"));
    phrase.erase(phrase.find_last_not_of(" \t") + 1);
    std::istringstream ts(line.substr(0, at));
    std::vector<int> ids;
    float score = cfg_.keywords_score, thr = cfg_.keywords_threshold;
    auto number = [&](const std::string& t) {
      try {
        std::size_t used = 0;
        const float v = std::stof(t.substr(1), &used);
        if (used == t.size() - 1) return v;
      } catch (const std::exception&) {
      }
      throw Error(-1, "BCDL KeywordSpotter: keywords line " + std::to_string(lineno) +
                          ": bad number '" + t + "'");
    };
    for (std::string t; ts >> t;) {
      if (t[0] == ':' && t.size() > 1) {
        score = number(t);
      } else if (t[0] == '#' && t.size() > 1) {
        thr = number(t);
      } else {
        const int id = tokenId(t);
        if (id < 0)
          throw Error(-1, "BCDL KeywordSpotter: keywords line " + std::to_string(lineno) +
                              ": token '" + t + "' is not in the model's token table");
        ids.push_back(id);
      }
    }
    if (ids.empty())
      throw Error(-1, "BCDL KeywordSpotter: keywords line " + std::to_string(lineno) +
                          " has no tokens");
    if (phrase.empty()) {
      for (int id : ids) phrase += (phrase.empty() ? "" : " ") + inv_[id];
    }
    g->add(ids, score, thr, static_cast<int>(phrases.size()));
    phrases.push_back(phrase);
  }
  if (phrases.empty()) throw Error(-1, "BCDL KeywordSpotter: no keywords");
  g->finish();
  graph_ = std::move(g);
  phrases_ = std::move(phrases);
  reset();
}

void KeywordSpotter::loadKeywords(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw Error(-1, "BCDL KeywordSpotter: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  setKeywords(ss.str());
}

void KeywordSpotter::reset() {
  state_.clear();
  for (auto& io : caches_) {
    std::size_t sz = 1;
    for (int d : enc_.inputShape(io.first)) sz *= static_cast<std::size_t>(d);
    state_.emplace_back(sz, 0.0f);
  }
  head_.clear();
  wav_.clear();
  feats_.clear();
  started_ = false;
  processed_ = frame_offset_ = 0;
  hyps_.clear();
  Hyp h;
  h.ys = {-1, kBlank};
  hyps_.push_back(h);
}

const float* KeywordSpotter::decoderOut(int y0, int y1) {
  const std::int64_t key = (static_cast<std::int64_t>(y0 + 1) << 32) | (y1 + 1);
  auto it = dec_cache_.find(key);
  if (it != dec_cache_.end()) return it->second.data();
  if (dec_cache_.size() > 4096) dec_cache_.clear();
  const int D = dim_, W = dim_ + 1, V = vocab_;
  const float* e0 = y0 >= 0 ? &table_[static_cast<std::size_t>(y0) * W] : zero_.data();
  const float* e1 = y1 >= 0 ? &table_[static_cast<std::size_t>(y1) * W] : zero_.data();
  std::vector<float> r(D);
  for (int o = 0; o < D; ++o) {
    const float* w = &table_[static_cast<std::size_t>(V + o) * W];
    const int g = o / kGroupIn;
    float acc = 0.0f;
    for (int i = 0; i < kGroupIn; ++i) {
      const int c = g * kGroupIn + i;
      acc += w[i * kKernel] * e0[c] + w[i * kKernel + 1] * e1[c];
    }
    r[o] = std::max(acc, 0.0f);
  }
  std::vector<float> out(D);
  for (int j = 0; j < D; ++j) {
    const float* p = &table_[static_cast<std::size_t>(V + D + j) * W];
    float acc = p[D];
    for (int o = 0; o < D; ++o) acc += p[o] * r[o];
    out[j] = acc;
  }
  return dec_cache_.emplace(key, std::move(out)).first->second.data();
}

void KeywordSpotter::encodeChunk(int timeout_ms) {
  const int F = fbank_.config().num_mel_bins;
  setInputFromFloat(enc_, in_x_, feats_.data(), static_cast<std::size_t>(t_in_) * F, stage_);
  const int q = static_cast<int>(std::min<std::int64_t>(processed_, left_));
  mask_.assign(mask_len_, 0.0f);
  std::fill(mask_.begin(), mask_.begin() + (left_ - q), 1.0f);
  setInputFromFloat(enc_, in_mask_, mask_.data(), mask_.size(), stage_);
  for (std::size_t k = 0; k < caches_.size(); ++k)
    setInputFromFloat(enc_, caches_[k].first, state_[k].data(), state_[k].size(), stage_);
  enc_.infer(timeout_ms);
  std::vector<int> shape;
  const float* e = outputAsFloat(enc_, out_enc_, scratch_, shape);
  enc_out_.assign(e, e + static_cast<std::size_t>(out_frames_) * dim_);
  for (std::size_t k = 0; k < caches_.size(); ++k) {
    const float* s = outputAsFloat(enc_, caches_[k].second, scratch_, shape);
    std::copy_n(s, state_[k].size(), state_[k].begin());
  }
  processed_ += shift_ / 2;
  feats_.erase(feats_.begin(), feats_.begin() + static_cast<std::size_t>(shift_) * F);
}

void KeywordSpotter::decodeFrame(const float* enc, std::vector<KwsDetection>& out) {
  const int V = vocab_, D = dim_, W = dim_ + 1;
  const int H = static_cast<int>(hyps_.size());
  const int unk = unk_;
  logits_.resize(static_cast<std::size_t>(H) * V);
  std::vector<float> act(D);
  for (int h = 0; h < H; ++h) {
    const std::vector<int>& ys = hyps_[h].ys;
    const float* d = decoderOut(ys[ys.size() - 2], ys[ys.size() - 1]);
    for (int j = 0; j < D; ++j) act[j] = std::tanh(enc[j] + d[j]);
    float* lg = &logits_[static_cast<std::size_t>(h) * V];
    for (int v = 0; v < V; ++v) {
      const float* w = &table_[static_cast<std::size_t>(V + 2 * D + v) * W];
      float acc = w[D];
      for (int j = 0; j < D; ++j) acc += w[j] * act[j];
      lg[v] = acc;
    }
    const float mx = *std::max_element(lg, lg + V);
    double s = 0.0;
    for (int v = 0; v < V; ++v) s += std::exp(static_cast<double>(lg[v] - mx));
    const float lse = mx + static_cast<float>(std::log(s));
    for (int v = 0; v < V; ++v) lg[v] -= lse;  // log-softmax
  }
  // top-k over (hypothesis, token) of path score
  const int K = std::min(cfg_.max_active_paths, H * V);
  std::vector<int> idx(static_cast<std::size_t>(H) * V);
  std::iota(idx.begin(), idx.end(), 0);
  auto score = [&](int k) { return hyps_[k / V].log_prob + logits_[k]; };
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
    const double sa = score(a), sb = score(b);
    return sa > sb || (sa == sb && a < b);
  });
  std::vector<Hyp> next;
  for (int r = 0; r < K; ++r) {
    const int k = idx[r], hi = k / V, tok = k % V;
    Hyp h = hyps_[hi];
    float cs = 0.0f;
    if (tok != kBlank && tok != unk) {
      h.ys.push_back(tok);
      h.timestamps.push_back(frame_offset_);
      h.probs.push_back(std::exp(logits_[k]));
      h.trailing = 0;
      const auto f = graph_->forward(h.state, tok);
      cs = f.first;
      h.state = f.second;
      if (graph_->n[h.state].token == -1) {  // back at the root: start over
        h.ys = {-1, kBlank};
        h.timestamps.clear();
        h.probs.clear();
      }
    } else {
      ++h.trailing;
    }
    h.log_prob = score(k) + cs;
    auto same = std::find_if(next.begin(), next.end(), [&](const Hyp& o) { return o.ys == h.ys; });
    if (same != next.end()) same->log_prob = logAdd(same->log_prob, h.log_prob);
    else next.push_back(std::move(h));
  }
  hyps_ = std::move(next);
  const Hyp* best = &hyps_[0];
  for (const Hyp& h : hyps_)
    if (h.log_prob > best->log_prob) best = &h;
  const int m = graph_->matched(best->state);
  if (m >= 0) {
    const Graph::Node& nd = graph_->n[m];
    const int lv = std::min<int>(nd.level, static_cast<int>(best->probs.size()));
    float avg = 0.0f;
    for (int i = 0; i < lv; ++i) avg += best->probs[i];
    avg = lv > 0 ? avg / nd.level : 0.0f;
    if (best->trailing > cfg_.num_trailing_blanks && avg >= nd.ac_threshold) {
      KwsDetection det;
      det.index = nd.phrase;
      det.keyword = phrases_.empty() ? std::string() : phrases_[nd.phrase];
      const int n = static_cast<int>(best->ys.size());
      const int take = std::min(nd.level, n);
      det.tokens.assign(best->ys.end() - take, best->ys.end());
      const int nt = static_cast<int>(best->timestamps.size());
      const int tk = std::min(nd.level, nt);
      const double step = 4.0 * fbank_.frameShift() / fbank_.config().sample_rate;  // 40 ms
      if (tk > 0) {
        det.start = best->timestamps[nt - tk] * step;
        det.end = best->timestamps[nt - 1] * step;
      }
      out.push_back(std::move(det));
      hyps_.clear();
      Hyp h;
      h.ys = {-1, kBlank};
      hyps_.push_back(h);
    }
  }
  ++frame_offset_;
}

std::vector<KwsDetection> KeywordSpotter::feed(const float* pcm, std::size_t n, int timeout_ms) {
  std::vector<KwsDetection> out;
  if (n == 0) return out;
  if (pcm == nullptr) throw Error(-1, "BCDL KeywordSpotter::feed: null pcm");
  if (phrases_.empty()) throw Error(-1, "BCDL KeywordSpotter: no keywords set");
  const int F = fbank_.config().num_mel_bins;
  // snip_edges = false framing: frame t is centred at t * shift + shift / 2,
  // so the first frame reaches len/2 - shift/2 samples before the stream and
  // Kaldi reflects them in. Prepending that reflection turns it into plain
  // snip-edges framing.
  const std::size_t pad = static_cast<std::size_t>(fbank_.frameLength() / 2 - fbank_.frameShift() / 2);
  if (!started_) {
    head_.insert(head_.end(), pcm, pcm + n);
    if (head_.size() < pad) return out;
    wav_.assign(head_.rend() - pad, head_.rend());
    wav_.insert(wav_.end(), head_.begin(), head_.end());
    head_.clear();
    started_ = true;
  } else {
    wav_.insert(wav_.end(), pcm, pcm + n);
  }
  std::vector<float> fr;
  const int nf = fbank_.compute(wav_.data(), wav_.size(), fr);
  if (nf > 0) {
    feats_.insert(feats_.end(), fr.begin(), fr.end());
    wav_.erase(wav_.begin(), wav_.begin() + static_cast<std::size_t>(nf) * fbank_.frameShift());
  }
  while (feats_.size() >= static_cast<std::size_t>(t_in_) * F) {
    encodeChunk(timeout_ms);
    for (int t = 0; t < out_frames_; ++t)
      decodeFrame(&enc_out_[static_cast<std::size_t>(t) * dim_], out);
  }
  return out;
}

std::vector<KwsDetection> KeywordSpotter::flush(int timeout_ms) {
  std::vector<KwsDetection> out;
  if (started_ || !head_.empty()) {
    // Enough silence for the lookahead of one more chunk plus the trailing
    // blanks: (T + shift) fbank frames, as sherpa-onnx pads at InputFinished.
    const std::size_t n = static_cast<std::size_t>(t_in_ + shift_) * fbank_.frameShift() +
                          static_cast<std::size_t>(fbank_.frameLength());
    const std::vector<float> zeros(n, 0.0f);
    out = feed(zeros.data(), zeros.size(), timeout_ms);
  }
  reset();
  return out;
}

std::vector<float> KeywordSpotter::encode(const float* pcm, std::size_t n, int& frames,
                                          int timeout_ms) {
  reset();
  std::vector<float> all;
  frames = 0;
  if (n == 0) return all;
  if (pcm == nullptr) throw Error(-1, "BCDL KeywordSpotter::encode: null pcm");
  const int F = fbank_.config().num_mel_bins;
  const std::size_t pad = static_cast<std::size_t>(fbank_.frameLength() / 2 - fbank_.frameShift() / 2);
  if (n < pad) return all;
  wav_.assign(std::reverse_iterator<const float*>(pcm + pad), std::reverse_iterator<const float*>(pcm));
  wav_.insert(wav_.end(), pcm, pcm + n);
  started_ = true;
  std::vector<float> fr;
  fbank_.compute(wav_.data(), wav_.size(), fr);
  feats_ = std::move(fr);
  while (feats_.size() >= static_cast<std::size_t>(t_in_) * F) {
    encodeChunk(timeout_ms);
    all.insert(all.end(), enc_out_.begin(), enc_out_.end());
    frames += out_frames_;
  }
  reset();
  return all;
}

}  // namespace bcdl
