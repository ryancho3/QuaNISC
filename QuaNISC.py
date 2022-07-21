# QUAntifier Negation Identification Using SpaCy

from csv import reader, writer
import spacy
from spacy import displacy
import cupy
import pandas as pd
import argparse
from progress.bar import IncrementalBar
from os.path import exists

nlp = spacy.load("en_core_web_sm")
print('INFO: spaCy initialized successfully.')


def get_quantifier(sentence, quantifiers):
    doc = nlp(sentence)
    dep = ['det', 'poss', 'advmod', 'nmod', 'nsubj', 'nsubjpass', 'ROOT']
    for token in doc:
        for quantifier in quantifiers:
            if (quantifier in token.text.lower()) and (token.dep_ in dep):
                return token
    return None


def assoc_negation_exists(sentence, q_root):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == 'neg':
            if (token.head.text == q_root.text and token.head.i == q_root.i) or (
                    token.head.head.text == q_root.text and token.head.head.i == q_root.i):
                return True
    return False


def get_q_root(quantifier):
    case_1 = ['nsubj', 'nsubjpass']
    case_2 = ['det', 'poss', 'advmod', 'nmod']
    dep = quantifier.dep_

    q_head = quantifier.head
    if dep in case_1:
        if q_head.dep_ == 'nsubj' or q_head.dep_ == 'auxpass':
            return q_head.head
        else:
            return q_head
    elif dep in case_2:
        return q_head.head


def reversed_traversal(sentence, quantifiers):
    doc = nlp(sentence)
    negation = None
    for token in doc:
        if token.dep_ == 'neg' or token.dep_ == 'preconj':
            negation = token
    if negation == None:
        return False

    ancestor = negation
    while ancestor != ancestor.head:
        ancestor = ancestor.head

    for quantifier in quantifiers:

        if ancestor.dep_ == 'ROOT' and quantifier in ancestor.text:
            return True
        for token in doc:
            if token.head == ancestor and quantifier in token.text and token.i < ancestor.i:
                return True

    return False


def is_quantifier_negation(sentence, quantifiers):
    quantifier = get_quantifier(sentence, quantifiers)
    if quantifier is None:
        return False
    if reversed_traversal(sentence, quantifiers):
        return True
    q_root = get_q_root(quantifier)
    if assoc_negation_exists(sentence, q_root):
        return True

    return False


def find_quantifier_negation(sentences, quantifiers):
    print('INFO: Beginning search for quantifier + negation statements.')
    ret = []
    bar = IncrementalBar('Searching...', max=len(sentences))
    for sentence in sentences:
        if is_quantifier_negation(sentence, quantifiers):
            ret.append(sentence)
        bar.next()
    bar.finish()
    print('INFO: Search completed with ' + str(len(ret)) + ' potential quantifier + negations.')
    print("\n")
    return ret


def read_csv(csv):
    print('INFO: Reading ' + csv + '.')
    sentences = []
    with open(csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            sentences.append(row[0])
    print('INFO: Successfully imported ' + str(len(sentences)) + ' sentences from ' + csv + '.')
    return sentences


def write_csv(sentences):
    print('INFO: Writing potential quantifier + negation statements to output.csv.')
    with open('output', 'w+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerows([sentence] for sentence in sentences)

def banner():
    print("  ____              _   _ _____  _____  _____ ")
    print(" / __ \            | \ | |_   _|/ ____|/ ____|")
    print("| |  | |_   _  __ _|  \| | | | | (___ | |     ")
    print("| |  | | | | |/ _` | . ` | | |  \___ \| |     ")
    print("| |__| | |_| | (_| | |\  |_| |_ ____) | |____ ")
    print(" \___\_\\__,_|\__,_|_| \_|_____|_____/ \_____|")
    print("Quantifier Negation Identification using SpaCy")
    print("version 1.1")
    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QUAntifier Negation Identification Using SpaCy v1.0")
    parser.add_argument('-r', '--read', type=str, nargs=1, metavar='file_name', default=None,
                        help='Specify the CSV file to read.')
    parser.add_argument('-q', '--quantifier', type=str, nargs=1, metavar='quantifier', default=None,
                        help='Specify the quantifier to search for.')
    parser.add_argument('-c', '--cuda', action='store_true', default=False,
                        help='Use CUDA acceleration.')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()
    banner()
    if args.debug is not None:
        print(is_quantifier_negation("everything I've heard him say hasn't been right", ['no', 'some', 'every']))

    if args.read is None:
        print('ERROR: Please specify the .csv file.')
        print('**QuaNISC Terminated**')
        quit()
    elif not exists(args.read[0]):
        print('ERROR: ' + args.read[0] + ' does not exist.')
        print('**QuaNISC Terminated**')
        quit()

    if args.cuda:
        spacy.prefer_gpu()
        print('INFO: CUDA GPU Acceleration initialized successfully.')

    sentences = read_csv(args.read[0])

    if args.quantifier is not None:
        quantifiers = [args.quantifier[0]]
        print('Identifying potential "' + args.quantifier[0] + '" + negation statements in ' + args.read[0])
    else:
        quantifiers = ['every', 'some', 'no']
        print('Identifying all potential quantifier + negation statements in ' + args.read[0])
    print("\n")
    identifications = find_quantifier_negation(sentences, quantifiers)
    write_csv(identifications)

    print('**QuaNISC Terminated**')
