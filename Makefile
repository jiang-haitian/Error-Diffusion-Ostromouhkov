CFLAGS = -O3
CFLAGS += -std=c++11
CFLAGS += -I/usr/include/opencv2
LFLAGS = -L/usr/lib/x86_64-linux-gnu
LFLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

error_diffusion: ostromoukhov.o main.o
	g++ $^ $(LFLAGS) -o $@

ostromoukhov.o: ostromoukhov.cpp ostromoukhov.hpp
	g++ -c $(CFLAGS) $< -o $@

main.o: main.cpp ostromoukhov.hpp
	g++ -c $(CFLAGS) $< -o $@

clean:
	-rm -rf error_diffusion ostromoukhov.o main.o
