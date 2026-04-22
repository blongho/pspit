# PSPI: Prokaryote SP Identifier
A model using long short-term memory and (n,k)-mers to identify prokaryotic SPs

## Contents
Several demo files have been included in the folders
### datasets/
For training and testing purposes
```
datasets/training/positives: Place any positive training data here
datasets/training/negatives: Place any negative training data here
datasets/testing/positives: Place any positive testing data here
datasets/testing/negatives: Place any negative testing data here

```
### input_files/
The default location PSPI will search for files to process

### output_files/
Results will be found here

### model_default.pkl
The default model this program uses

### pspi.py
The exectuable itself

### README.md
This file

## Dependencies
```
python 3.11.5
keras 2.15.0
numpy 1.24.3
scikit-learn 1.3.0
```
## File Format requirements
input files must be in fasta format. The entire sequence must be contained on a single line.

This tool is designed to identify amino acid sequences of at-most 100 AAs. Including longer sequences may have undefined behaviors

## Execution
You must be in the PSPI folder to run the program. 
If you have multiple files you wish to run at once, place them all in the input_files directory.

The required command is
```
python3 pspi.py
```
This will cause the program to use model_default.pkl to identify SPs in all the files saved in input_files/ and save the results to output_files/results.csv

### Optional arguments
```
-m <model name> : Select a preexisting model to use instead of the default model
-t <model name> : Have the tool build a new model using the files in datasets/training and test the model using the files in datasets/testing. Argument requires a name for the newly trained model
-r <model name> : Test an existing model using the files in datasets/testing. Argument requires the name of the model to test
-i <input file path> : Provide a specific file. Input must be the filepath to the specific file and be in fasta format. If this option isn't used, PSPI will take every file in input_files as input
-o <output file path> : Define a specific name for the output file. Default name is results.csv
-l <float [0,1]> : Select the threshold value required to flag a sequence as a short protein. Value must be between 0 and 1. Default is 0.75
```
### Examples
```
Running with a custom model:
python3 pspi.py -m my_model.pkl -i my_input.fasta -o my_output.csv -l .35

building and testing a new model:
python3 pspi.py -t my_model.pkl

testing an existing model on the testing data:
python3 pspi.py -r my_model.pkl
```

## Training your own model
Training and testing data must match the format requirements for input files.

Place each training and testing datafile in its corresponding folder in datasets/
When you've set everything up, run the command
```
python3 pspi -t <model name>
```
pspi will automatically pull all the datafiles in datasets/training and use them to train a new model.

Once it has trained the model, it will run 12 tests where it randomly chooses 1000 positive and 1000 negative datapoints from the files in datasets/testing. If there are fewer than 1000 of either one, it will use all of them for testing.

The model will output the precision, sensitivity, specificity, f1, AUROC, and AUPR scores of each test followed by the average results.

Finally, it will prompt you if you want to save the model. Typing 'y' or 'yes' will create a binary file <model name>. Typing 'n' or 'no' will close the program without creating a file.