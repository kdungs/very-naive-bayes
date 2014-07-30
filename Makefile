CXX=clang++
CXXFLAGS=-O3 -Wall -Werror -pedantic -std=c++11

all: evaluation

main: main.cc naivebayes.cc

testfiles: scripts/maketestdata.py
	mkdir -p data
	scripts/maketestdata.py

predictions.dat: testfiles main
	./main

evaluation: predictions.dat scripts/evaluate.py
	scripts/evaluate.py

clean:
	rm -f main
	rm -rf data
	rm -f predictions.dat
