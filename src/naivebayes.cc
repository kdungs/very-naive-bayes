#include <cmath>
#include <iostream>

#include "naivebayes.h"
#include "runningstats.hpp"

namespace NB {
  
  const number_t gaussian(const number_t &x, const number_t &mu,
                        const number_t &sigma) {
    return 1. / std::sqrt(2 * M_PI) / sigma *
           std::exp(-.5 * std::pow((x - mu) / sigma, 2));
  }

  void NaiveBayes::train(const std::vector<Event> &events,
                         const std::vector<label_t> &labels) {
    if (events.size() != labels.size()) {
      std::cerr << "Error in " << __FILE__ << ":" << __LINE__ << std::endl
                << "Training sample and labels have to have same size."
                << std::endl;
      return;
    }

    std::vector<RunningStats<number_t>> s(_num_features), b(_num_features);
    for (std::size_t e = 0; e < events.size(); e++) {
      auto *c = &s;
      if (labels[e] != SIG) {
        c = &b;
      }
      for (std::size_t f = 0; f < _num_features; f++) {
        c->at(f).Push(events[e].features[f]);
      }
    }

    for (std::size_t f = 0; f < _num_features; f++) {
      _mu_s[f] = s[f].Mean();
      _sigma_s[f] = s[f].StdDev();
      _mu_b[f] = b[f].Mean();
      _sigma_b[f] = b[f].StdDev();
    }

    _trained = true;
  }

  const label_t NaiveBayes::predict(const Event &event) const {
    if (!_trained) {
      std::cerr << "Error in " << __FILE__ << ":" << __LINE__ << std::endl
                << "Leaner has to be trained before a prediction can be made."
                << std::endl;
      return ERR;
    }

    number_t prob_s(1), prob_b(1);
    for (std::size_t f = 0; f < _num_features; f++) {
      prob_s *= gaussian(event.features[f], _mu_s[f], _sigma_s[f]);
      prob_b *= gaussian(event.features[f], _mu_b[f], _sigma_b[f]);
    }
    return prob_s > prob_b ? SIG : BKG;
  }
}
