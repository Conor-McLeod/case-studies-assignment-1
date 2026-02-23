# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O3
LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

# Target executable and source files
TARGET = q2_tsqr
SRCS = q2_tsqr.c
OBJS = $(SRCS:.c=.o)

# Default target
.PHONY: all
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
