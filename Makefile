all: compile main

compile:
	cmake -S . -B build
	cmake --build build

.PHONY: main
main:
	./build/main

.PHONY: test
test:
	cd build && ctest

clean:
	rm -r build