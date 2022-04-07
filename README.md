# Simulation of Quantum Circuits on a laptop

## Description
This is my dissertation project for my 4th year of BSc (Hons) Computer Science & Physics in the University of Edinburgh.

## Files
<ul>
  <li> <a href="./Dissertation-PabloMiróRuiz.pdf">Dissertation</a> --> This pdf file contains my undergraduate thesis in the University of edinburgh.
  <li> <a href="./src">src</a> --> This directory contains the implementation of the algorithm, alongside several experiments.
  <li> <a href="./src/experiments.py">experiments.py</a> --> Python file to try the different approaches for the sampling algorithm. It also contains the methods for reproducing the figures from chapter 6 of the dissertation.
  <li> <a href="./results">results</a> --> This directory contains the experimental XEBs obtained while variying the order of the correlators. There are also some .pdf files containing several plots.
  <li> <a href="./img">img</a> --> This directory contains images such as the ones added in chapter 6 of the paper.
</ul>

## Getting Started

These subsections will help you get a copy of the project and understand how to run it on your local machine for development and testing purposes.
I will discuss how to clone this repository and set it up in any IDE of your choice. 

### How to Install

The first thing you should do is clone this repository into your local machine. You can do this with the following command:
```
git clone https://github.com/Paramiru/Simulations-on-Quantum-Circuits
```
Once you have cloned the repository, you should check your current version of Python. I used Python 3.9.7 for the project. You can check the version you are currently using running this command in the terminal.
```
python --version
```
In order to run the project with the same version I am using you can use the environment.yml file to import my conda environment using:
```
conda env create -f environment.yml
```
This will create an environment named <em>quantum</em> which you can use for running the code without having to install the dependencies on your system. Read <a href="https://realpython.com/python-virtual-environments-a-primer/">this article</a> to understand the reasons for why virtual environments are important in Python.

## Running the Project 

Once you have cloned the repository and have the conda environment (or another virtual environment with the required dependencies) you can then run the **experiments.py** file. Uncomment some of the lines in the file to try different experiments.
## Built With

* [Python3](https://www.python.org/downloads/)
* [Conda](https://docs.conda.io/en/latest/) - Dependency Management

## Python Dependencies

* [Numpy](https://numpy.org/doc/stable/index.html)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/stable/)
* [Cirq](https://quantumai.google/cirq)

## Authors

* **Pablo Miró** - [Paramiru](https://github.com/Paramiru)

