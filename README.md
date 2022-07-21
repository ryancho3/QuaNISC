# QuaNISC: Quantifier Negation Identification using SpaCy
QuaNISC was written for Ambilab, a research team part of the Language Science department at the University of California: Irvine.
Using spaCy, QuaNISC identifies potential quantifier-negation statements from corpora transcripts.

## Installation
Please install the following dependencies using your package manager of choice:
- spaCy
- cupy
- pandas
- argparse
- progress

*Important: spaCy requires more steps to install than your typical python library. Please refer to [this page](https://spacy.io/usage) for more information. Make sure you have the english model installed, and have the CUDA toolkit installed if you have an NVIDIA GPU.*

## Usage
`python QuaNISC.py -r [input_file_path] -q [quantifier] -c'
'-r' or '--read' (required): Designates the path to the input file. Accepted file type: .csv, .txt (not implemented yet).
'-q' or '--quantifier' (optional): Specifies which quantifier to look for. QuaNISC will look for all quantifiers (every, some, no) if not given an argument.
'-c' or '--cuda' (optional): Using this arguement will enable CUDA GPU acceleration. Only use if your system has an NVIDIA GPU.

'python QuaNISC.py -r [input_file_path]' will search your file for quantifier negations that include 'every, some, no'.

### Input File Format
**CSV**
Format your .csv file so that each individual sentence separated by a comma or is on another line.
