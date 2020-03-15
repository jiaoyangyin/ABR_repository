CFLAGS=-I/home/yinjiaoyang/anaconda3/envs/pensieve/include/python2.7 
LDFLAGS=-lpython2.7 -L/home/yinjiaoyang/anaconda3/envs/pensieve/lib/
CC=gcc
#CC=clang



all:runpython 

help:
	@echo "---------------------------------------------------"
	@echo "---------------------------------------------------"

runpython:main.o 
	$(CC) $(CFLAGS) -o runpython main.o  $(LDFLAGS)

main.o : main.c 
	$(CC) $(CFLAGS)  -c main.c 

clean :
	rm -rf *.o runpython
