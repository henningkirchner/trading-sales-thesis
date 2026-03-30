# Pairs Trading Facharbeit

This repository contains the code used for my Thesis on pairs trading. The project examines whether a self-developed pairs trading strategy can generate profitable returns based on historical market data.

## Project Scope

The analysis includes:

- data preparation
- spread construction
- signal generation
- backtesting
- performance evaluation

## Objective

The main goal of this project is to test whether a rules-based pairs trading strategy can exploit temporary mispricing between two historically related stocks and generate positive returns over time.

## Repository Structure

- two csv input data files
- `outputs/` – exported results, charts, and performance summaries, created after executing the python script
- `*.py` – Python scripts for data processing, strategy logic, and evaluation

## Requirements

To execute the script all input files and the script need to be in the same folder.

Install the required packages with:

```bash
pip install -r requirements.txt
```
Alternatively the packages can be install one by one, by just typing the following commands into the consol.

pip install pandas

pip install numpy

pip instll matplotlib

## Notes

This repository was created as part of a school paper (Facharbeit).
The code is intended to document the empirical part of the analysis and support the results discussed in the paper.
