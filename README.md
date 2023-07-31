# Proximity Criteria Checker

## Description

This project is a Proximity Criteria Checker for a set of points placed in a two-dimensional plane. The coordinates (x, y) of each point P are defined by given formulas. The project checks if there exist at least 3 points that satisfy the Proximity Criteria for a range of t values. The project is implemented using C, CUDA, MPI, and OpenMP.

## Installation

To install this project, follow these steps:
1. Clone the repository to your local machine.
2. Ensure that you have the necessary dependencies installed. These include a C compiler, CUDA toolkit, MPI, and OpenMP.
3. Build the project using the provided Makefile.

## Dependencies

1. **C Compiler**: A C compiler is required to compile the source code. GCC (GNU Compiler Collection) is a popular choice and can be installed on Ubuntu using the command `sudo apt install build-essential`.

2. **CUDA Toolkit**: CUDA Toolkit provides a development environment for creating high performance GPU-accelerated applications. You can download the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit). Please follow the installation instructions provided on the website.

3. **MPI (Message Passing Interface)**: MPI is a standardized and portable message-passing system designed to function on a wide variety of parallel computing architectures. You can install MPI using the command `sudo apt install mpich` or `sudo apt install openmpi-bin` on Ubuntu.

4. **OpenMP (Open Multi-Processing)**: OpenMP is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It's usually included with the compiler. If you're using GCC, OpenMP support is included.

Please note that these instructions are for Ubuntu. If you're using a different operating system, the commands might be different. Also, make sure to check the compatibility of your GPU with the CUDA version you're installing.

## Usage

To use this project:

1. Prepare your input file `input.txt` with the following format:
   N K D TCount \
   id x1 x2 a b \
   id x1 x2 a b \
   ... \
   id x1 x2 a b \
2. Run the program. The results will be written to the file `output.txt`.

## Running Instructions

1. After cloning the repository and ensuring all dependencies are installed, navigate to the project directory.
2. Run `make` to build the project. This will generate the executable.
3. Run `make run` to run the executable.

Please fill in the specific commands for your Makefile and executable here.

## Input and Output

The input file contains N in the first line - the number of points in the set, K – minimal number of points to satisfy the Proximity Criteria, distance D and TCount. The next N lines contain parameters for every point in the set.

The output file contains information about results found for points that satisfy the Proximity Criteria. For each t that 3 points satisfying the Proximity Criteria were found, it contains a line with the parameter t and ID of these 3 points. In case that the points were not found for any t, the program outputs: "There were no 3 points found for any t."
