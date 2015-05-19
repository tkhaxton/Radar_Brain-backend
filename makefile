
COMPILER=g++
FLAGS=-lm -O3 -lboost_system  -lboost_filesystem

all : learn_minibatch generalize

learn_minibatch : learn_minibatch.cpp functions.cpp
	${COMPILER} ${FLAGS} -o learn_minibatch learn_minibatch.cpp functions.cpp

generalize : generalize.cpp functions.cpp
	${COMPILER} ${FLAGS} -o generalize generalize.cpp functions.cpp

clean:
	rm generalize
	rm learn_minibatch

