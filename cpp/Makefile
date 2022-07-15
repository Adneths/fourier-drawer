.PHONY: clean debug

build: lib.o
	g++ -O3 -static -shared "-I${PYTHON_PATH}\include" "-L${PYTHON_PATH}\libs" -o lib.dll lib.o -lpython39

lib.o: lib.cpp
	g++ -c lib.cpp -o lib.o

debug: lib.cpp
	g++ -g "-I${PYTHON_PATH}\include" "-L${PYTHON_PATH}\libs" -o lib.exe lib.cpp -lpython39
	gdb lib.exe

clean:
	rm lib.o lib.dll lib.exe