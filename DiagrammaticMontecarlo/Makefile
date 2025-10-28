CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare -Wno-tautological-compare -Wno-reorder -Wno-unused-private-field

TARGET   := diagmc

SRC      := DMC.cpp \
            feynmanDiagram.cpp \
            saveSimulationToFile.cpp \
            main.cpp

OBJ      := $(SRC:.cpp=.o)

.PHONY: all clean run prep

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# compile rules
DMC.o: DMC.cpp feynmanDiagram.h saveSimulationToFile.h
	$(CXX) $(CXXFLAGS) -c DMC.cpp

feynmanDiagram.o: feynmanDiagram.cpp feynmanDiagram.h
	$(CXX) $(CXXFLAGS) -c feynmanDiagram.cpp

saveSimulationToFile.o: saveSimulationToFile.cpp saveSimulationToFile.h
	$(CXX) $(CXXFLAGS) -c saveSimulationToFile.cpp

main.o: main.cpp feynmanDiagram.h saveSimulationToFile.h
	$(CXX) $(CXXFLAGS) -c main.cpp

# create output directory if missing
prep:
	mkdir -p results

# build + run full simulation
run: prep all
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
