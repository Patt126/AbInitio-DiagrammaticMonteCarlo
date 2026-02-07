CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare -Wno-tautological-compare -Wno-reorder -Wno-unused-private-field

COMMON_SRC     := DMC.cpp feynmanDiagram.cpp saveSimulationToFile.cpp
COMMON_HEADERS := feynmanDiagram.h saveSimulationToFile.h

BENCH_SRC := Benchmark/benchmark.cpp
BREATH_SRC := Breathing_Coupling/breathing.cpp
SIGN_SRC := sign_oscillation/sign_oscillation.cpp

BENCH_BIN := Benchmark/benchmark_run
BREATH_BIN := Breathing_Coupling/breathing_run
SIGN_BIN := sign_oscillation/sign_oscillation_run

BENCH_OUT := Benchmark/results/benchmark.csv
BENCH_GREEN_OUT := Benchmark/results/benchmark_green.csv
BREATH_OUT := Breathing_Coupling/results/breathing.csv
BREATH_GREEN_OUT := Breathing_Coupling/results/breathing_green.csv
SIGN_ABS_OUT := sign_oscillation/results/sign_oscillation_abs.csv
SIGN_ABS_GREEN_OUT := sign_oscillation/results/sign_oscillation_abs_green.csv
SIGN_SIGNED_OUT := sign_oscillation/results/sign_oscillation_signed.csv
SIGN_SIGNED_GREEN_OUT := sign_oscillation/results/sign_oscillation_signed_green.csv

.PHONY: all prep benchmark breathing sign run-benchmark run-breathing run-sign run-sign-abs run-sign-signed run clean

all: benchmark breathing sign

prep:
	mkdir -p Benchmark/results Breathing_Coupling/results sign_oscillation/results

benchmark: prep $(BENCH_BIN)

breathing: prep $(BREATH_BIN)

sign: prep $(SIGN_BIN)

$(BENCH_BIN): $(BENCH_SRC) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(BENCH_SRC) $(COMMON_SRC)

$(BREATH_BIN): $(BREATH_SRC) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(BREATH_SRC) $(COMMON_SRC)

$(SIGN_BIN): $(SIGN_SRC) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SIGN_SRC) $(COMMON_SRC)

run-benchmark: benchmark
	./$(BENCH_BIN) --out=$(BENCH_OUT) --out-green=$(BENCH_GREEN_OUT)

run-breathing: breathing
	./$(BREATH_BIN) --out=$(BREATH_OUT) --out-green=$(BREATH_GREEN_OUT)

run-sign: sign
	@$(MAKE) run-sign-abs
	@$(MAKE) run-sign-signed

run-sign-abs: sign
	./$(SIGN_BIN) --variant=abs --out=$(SIGN_ABS_OUT) --out-green=$(SIGN_ABS_GREEN_OUT)

run-sign-signed: sign
	./$(SIGN_BIN) --variant=signed --out=$(SIGN_SIGNED_OUT) --out-green=$(SIGN_SIGNED_GREEN_OUT)

run: run-benchmark run-breathing

clean:
	rm -f *.o diagmc bench_paper
	rm -f $(BENCH_BIN) $(BREATH_BIN) $(SIGN_BIN)
