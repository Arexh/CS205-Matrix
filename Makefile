all: compile main

compile:
	cmake -S . -B build
	cmake --build build

.PHONY: main
main:
	./build/main

.PHONY: test
test: compile
	cd build && ctest

clean:
	rm -r build