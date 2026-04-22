from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import pickle
import numpy as np
import argparse
import os
import sys
import random
import csv

timesteps = 1


def get_gap_dimer(seq):
	chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
	chardict = {}
	for i in range(0,20):
		chardict[chars[i]] = i
	count = [0] * 400
	motif_size = 4

	for i in range(0, len(seq) - 1):
		for j in range(i+1, min(i + motif_size, len(seq))):
			try:
				index = chardict[seq[i]] * 20 + chardict[seq[j]]
				count[index] += 1
			except:
				continue

	return count


def prepare_feature(sequences):

	protein_seq_dict = {}
	protein_index = 1
	for line in sequences:
		seq = line
		protein_seq_dict[protein_index] = seq
		protein_index = protein_index + 1


	aavectors = []
	gapDimer = []

	# get protein feature
	for i in protein_seq_dict:

		aavectors_feature = get_aavectors(protein_seq_dict[i])
		gapDimer_feature = get_gap_dimer(protein_seq_dict[i])

		if len(aavectors_feature) != 20 * 100:
			print(sys.stderr, "Warning: " + protein_seq_dict[i] + " is too long and will be discarded")
			continue
			
		aavectors.append(aavectors_feature)
		gapDimer.append(gapDimer_feature)
		protein_index = protein_index + 1

	X = np.concatenate((aavectors, gapDimer), axis=1) 


	return X

def get_aavectors(seq_temp):
	seq = seq_temp
	fea = []
	tem_vec =[]
	k = len(seq_temp)
	for i in range(k):
		if seq[i] =='A':
			tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='C':
			tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='D':
			tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='E':
			tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='F':
			tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='G':
			tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='H':
			tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='I':
			tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='K':
			tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='L':
			tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='M':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
		elif seq[i]=='N':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
		elif seq[i]=='P':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
		elif seq[i]=='Q':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
		elif seq[i]=='R':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
		elif seq[i]=='S':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
		elif seq[i]=='T':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
		elif seq[i]=='V':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
		elif seq[i]=='W':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
		elif seq[i]=='Y':
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
		else:
			tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

		fea = fea + tem_vec
	for i in range(k, 100):
		fea = fea + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	return fea

def calculate_performace(test_num, pred_y, labels):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for index in range(test_num):
		if labels[index] == 1:
			if labels[index] == pred_y[index]:
				tp = tp + 1
			else:
				fn = fn + 1
		else:
			if labels[index] == pred_y[index]:
				tn = tn + 1
			else:
				fp = fp + 1

	precision = float(tp) / (tp + fp)
	sensitivity = float(tp) / (tp + fn)
	specificity = float(tn) / (tn + fp)
	f1 = float(2*tp) / (2 * tp + fp + fn)
	return precision, sensitivity, specificity, f1

def transfer_label_from_prob(proba, threshold):
	if threshold != None:
		label = [1 if val >= threshold else 0 for val in proba]
	else:
		label = [1 if val >= .75 else 0 for val in proba]
	return label
def plot_roc_curve(labels, probality):
	fpr, tpr, thresholds = roc_curve(labels, probality)
	roc_auc = auc(fpr, tpr)

	return roc_auc

def getSeqs(dir):
    
	seq = []
	directory = os.listdir(dir)
	for file in directory:
		fileList = list(open(dir + file, "r"))
		for line in fileList:
			if line[0] != '>':
				seq.append(line[:-1])
	return seq

def buildModel():

	batch_size = 32   
	epochs = 30

	labeltraining = []
	print("Reading Training data")
	print("-" * 50)
	sequences = getSeqs("datasets/training/positives/")
	for i in range(0, len(sequences)):
		labeltraining.append(1)
	sequences += getSeqs("datasets/training/negatives/")
	for i in range(len(labeltraining), len(sequences)):
		labeltraining.append(0)
	print("Preparing features")
	print("-" * 50)
	aavectorstraining = prepare_feature(sequences)
	X_train = np.array(np.reshape(aavectorstraining, (len(aavectorstraining), timesteps, len(aavectorstraining[0]))))
	train_label = np.array(labeltraining)


	model = Sequential()
	model.add(LSTM(128, return_sequences=False,input_shape=(timesteps, len(aavectorstraining[0])), name='lstm1'))
	model.add(Dropout(0.25, name='dropout'))
	model.add(Dense(1, name='full_connect2'))

	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', 
		optimizer='adam',
		metrics=['accuracy'])
	print("Training the model")
	print("-" * 50)
	model.fit(X_train, train_label, batch_size=batch_size,epochs=epochs, verbose = 0)
 
	return model

def sortBysensitivity(a, b):
	return a[2] - b[2]

def testModel(model, threshold):
    print("Testing")
    print("-" * 50)
    pos_sequences = getSeqs("datasets/testing/positives/")
    neg_sequences = getSeqs("datasets/testing/negatives/")
    sums = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for j in range(1, 13):
        subsequence = []
        labeltesting = []
        y_pred = []
        all_prob = []
        if len(pos_sequences) > 1000:
            subsequence += random.sample(pos_sequences, 1000)
            labeltesting += [1] * 1000
        else:
            subsequence += pos_sequences
            labeltesting += [1] * len(pos_sequences)
        if len(neg_sequences) > 1000:
            subsequence += random.sample(neg_sequences, 1000)
            labeltesting += [0] * 1000
        else:
            subsequence += neg_sequences
            labeltesting += [0] * len(neg_sequences)

        aavectorstesting = prepare_feature(subsequence)

        X_test_all = np.array(
            np.reshape(
                aavectorstesting,
                (len(aavectorstesting), timesteps, len(aavectorstesting[0])),
            )
        )
        lstm_proba = model.predict(X_test_all, verbose=0)
        y_pred += transfer_label_from_prob(lstm_proba, threshold)
        all_prob += [val for val in lstm_proba]

        test_label = np.array(labeltesting)
        all_performance_lstm = []

        prec, sensitivity, specificity, f1 = calculate_performace(
            len(test_label), y_pred, test_label
        )

        roc_auc = plot_roc_curve(test_label, all_prob)
        all_performance_lstm = sorted(all_performance_lstm, key=lambda res: res[2])
        precision, recall, t = precision_recall_curve(test_label, all_prob)

        aupr = auc(recall, precision)

        ROUND_DIGITS = 4

        print("Results: Trial " + str(j) + "/12")
        print("precision\t", round(prec, ROUND_DIGITS))
        print("sensitivity\t", round(sensitivity, ROUND_DIGITS))
        print("specificity\t", round(specificity, ROUND_DIGITS))
        print("f1\t\t", round(f1, ROUND_DIGITS))
        print("AUROC\t\t", round(roc_auc, ROUND_DIGITS))
        print("AUPR\t\t", round(aupr, ROUND_DIGITS))
        sums += np.array([prec, sensitivity, specificity, f1, roc_auc, aupr])
        print("-" * 50)

    print("Results: Average")
    print("precision\t", round(sums[0] / 12, ROUND_DIGITS))
    print("sensitivity\t", round(sums[1] / 12, ROUND_DIGITS))
    print("specificity\t", round(sums[2] / 12, ROUND_DIGITS))
    print("f1\t\t", round(sums[3] / 12, ROUND_DIGITS))
    print("AUROC\t\t", round(sums[4] / 12, ROUND_DIGITS))
    print("AUPR\t\t", round(sums[5] / 12, ROUND_DIGITS))
    print("-" * 50)


def readFile(model, inFile, outfile, threshold):
	labels = []
	sequences = []
	for line in inFile:
		if line[0] == ">":
			labels.append(line[:-1])
		else:
			sequences.append(line[:-1])
		if len(sequences) == 1000:
			features = prepare_feature(sequences)
			X_test_all = np.array(np.reshape(features, (len(features), timesteps, len(features[0]))))
			lstm_proba = model.predict(X_test_all, verbose = 0)
			y_pred = transfer_label_from_prob(lstm_proba, threshold)
			for i in range(0, len(sequences)):
				outfile.writerow([labels[i], sequences[i], y_pred[i], lstm_proba[i][0]])
			labels=[]
			sequences = []
	features = prepare_feature(sequences)
	X_test_all = np.array(np.reshape(features, (len(features), timesteps, len(features[0]))))
	lstm_proba = model.predict(X_test_all, verbose = 0)
	y_pred = transfer_label_from_prob(lstm_proba, threshold)
	for i in range(0, len(sequences)):
		outfile.writerow([labels[i], sequences[i], y_pred[i], lstm_proba[i][0]])


def categorize(model, input, output, threshold):
	files = os.listdir("input_files")
	if output != None:
		outfile = csv.writer(open("output_files/" + output, 'w'))
	else:
		outfile = csv.writer(open("output_files/results.csv", 'w'))
	outfile.writerow(["Seq name", "seqeunce", "prediction", "confidence"])
	
	if input != None:
		with open(input , "r") as inFile:
			readFile(model, inFile, outfile, threshold)
	else:
		for file in files:
			with open("input_files/" + file, "r") as inFile:
				readFile(model, inFile, outfile, threshold)

def PSPI(model, train, test, file_input, output, threshold):

	if train != None:
		model = buildModel()
		testModel(model, threshold)
		yes_res = ['y', 'yes']
		no_res = ['n', 'no']
		while True:
			user_input = input('Save the model? y/n: ')

			if user_input.lower() in yes_res:
				with open(train, "wb") as f:
					pickle.dump(model, f)
				return
			elif user_input.lower() in no_res:
				return
			else:
				print('Response must be yes or no')


	if test != None:
		model = pickle.load(open(test, "rb"))
		testModel(model, threshold)
		return

	if model != None:
		model = pickle.load(open(model, "rb"))
	else:
		model = pickle.load(open("model_default.pkl", "rb"))


	categorize(model, file_input, output, threshold)


parser = argparse.ArgumentParser(description="PSPI: A long short-term memory model for identifying short proteins in prokaryotes")

parser.add_argument("-m", dest="model", type = str, required = False, help = "Optional. Select a preexisting model to use instead of the default model")
parser.add_argument("-t", dest="train", type = str, required = False, help = "Optional. Have the tool build a new model using the files in datasets/training and test the model using the files in datasets/testing. Argument requires a name for the newly trained model")
parser.add_argument("-r", dest="test", type = str, required = False, help = "Optional. Test an existing model using the files in datasets/testing. Argument requires the name of the model to test")
parser.add_argument("-i", dest="input", type = str, required = False, help = "Optional. Provide a specific file. Input must be the filepath to the specific file and be in fasta format. If this option isn't used, PSPI will take every file in input_files as input")
parser.add_argument("-o", dest="output", type = str, required = False, help = "Optional. Define a specific name for the output file. Default name is results.csv")
parser.add_argument("-l", dest="threshold", type = float, required = False, help = "Optional. Select the threshold value required to flag a sequence as a short protein. Value must be between 0 and 1. Default is 0.75")
args = parser.parse_args()


if args.threshold != None and ((args.threshold < 0) or (args.threshold > 1)):
	print(sys.stderr, "threshold value must be between 0 and 1\n")
	sys.exit(0)


PSPI(args.model, args.train, args.test, args.input, args.output, args.threshold)
