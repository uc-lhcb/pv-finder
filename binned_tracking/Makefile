LD      = g++
CC      = gcc
FLAGS   = -Wall -O2
INCLUDE = -I .
ROOT_FLAGS      = $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOT_LIBS       = $(shell $(ROOTSYS)/bin/root-config --libs) -lMinuit
LIBS = ${ROOT_LIBS}
FLAGS += ${ROOT_FLAGS}

%: %.cc *.h
	$(LD) $(FLAGS) $(INCLUDE) -c -o $*.o $*.cc
	$(LD) $(FLAGS) $*.o $(LIBS) -o $*
