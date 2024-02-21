FLAGS = -O3 -std=c++17 -Werror

benchmark: main.cc shaders.metallib
	clang++ main.cc ${FLAGS} -framework Foundation -framework Metal -o $@

shaders.metallib: shaders.metal
	xcrun metal shaders.metal -O3 -o $@

clean:
	rm -f benchmark shaders.metallib

.PHONY: clean
