#include <fstream>
#include <iostream>
#include <vector>

#include "naivebayes.h"

std::tuple<std::vector<NB::Event>, std::vector<NB::label_t>>
load_data(const std::string &filename, const std::size_t num_features) {
  std::vector<NB::Event> events;
  std::vector<NB::label_t> labels;
  std::vector<NB::number_t> features(num_features);
  NB::label_t label;

  std::ifstream ifs(filename);
  while (ifs.good()) {
    for (std::size_t f = 0; f < num_features; f++) {
      ifs >> features[f];
    }
    ifs >> label;
    events.push_back(NB::Event{features});
    labels.push_back(label);
  }
  ifs.close();

  return std::make_tuple(events, labels);
}

int main(int argc, char *argv[]) {
  const size_t num_features = 4;
  NB::NaiveBayes nb(num_features);
  std::vector<NB::Event> events;
  std::vector<NB::label_t> labels;

  // Training
  std::tie(events, labels) = load_data("data/train.dat", num_features);
  nb.train(events, labels);

  // Testing (write results to file, actual evaluation is done via Python)
  std::tie(events, labels) = load_data("data/test.dat", num_features);
  std::ofstream ofs("predictions.dat");
  for (const auto &e : events) {
    ofs << nb.predict(e) << std::endl;
  }
  ofs.close();
}
