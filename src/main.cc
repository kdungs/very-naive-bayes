#include <fstream>
#include <iostream>
#include <string>
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
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " N train predict" << std::endl
              << "  N       - number of features" << std::endl
              << "  train   - path to the training data sample" << std::endl
              << "  predict - path to the data sample to be predicted"
              << std::endl;
    return EXIT_FAILURE;
  }

  const size_t num_features = std::stoul(argv[1]);
  const std::string filename_train = std::string(argv[2]),
                    filename_predict = std::string(argv[3]);

  NB::NaiveBayes nb(num_features);
  std::vector<NB::Event> events;
  std::vector<NB::label_t> labels;

  // Training
  std::tie(events, labels) = load_data(filename_train, num_features);
  nb.train(events, labels);

  // Prediction
  std::tie(events, labels) = load_data(filename_predict, num_features);
  for (const auto &e : events) {
    std::cout << nb.predict(e) << std::endl;
  }
}
