CC=g++
TARGET=main
CFLAGS=-c -Wall
BUILD_PATH=build
OBJECTS_CPP=main Matrix
OBJECTS=$(patsubst %, $(BUILD_PATH)/%.o, $(OBJECTS_CPP))

build: build_cache main

.PHONY:main
main: $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

build_cache:
	mkdir -p $(BUILD_PATH)

$(BUILD_PATH)/%.o: %.cpp
	$(CC) $(CFLAGS) $< -o $@

.PHONY:clean
clean:
	rm -rf ${BUILD_PATH} $(TARGET)