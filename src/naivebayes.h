#pragma once

#include <vector>

namespace NB {
  typedef double number_t;
  typedef int label_t;

  const label_t SIG = 0;
  const label_t BKG = 1;
  const label_t ERR = -1;

  struct Event {
    std::vector<number_t> features;
  };

  const number_t gaussian(const number_t &, const number_t &, const number_t &);

  class NaiveBayes {
   private:
     const std::size_t _num_features;
     bool _trained;
     std::vector<number_t> _mu_s, _mu_b, _sigma_s, _sigma_b;

   public:
     NaiveBayes() = delete;
     NaiveBayes(const std::size_t num_features)
         : _num_features(num_features), _trained(false), _mu_s(num_features),
           _mu_b(num_features), _sigma_s(num_features),
           _sigma_b(num_features){};
     const std::size_t num_features() const { return _num_features; }
     const bool trained() const { return _trained; }
     void train(const std::vector<Event>&, const std::vector<label_t>&);
     const label_t predict(const Event&) const;
  };
}
