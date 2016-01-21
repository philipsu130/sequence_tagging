from BaselineNER import Interval
import math

class HMM:
    """
    A Hidden Markov Model that utilizes the Viterbi algorithm to determine the most likely
    sequence of tags for a Named Entity Recognition task. Tunable hyperparameters include:
        - Unknown token threshold
        - Smoothing amount
        - n value for N-gram model
    """
    # Mapping of each labels to a map of word : count
    words_labels_counts = {}

    START_OF_SENTENCE_TOKEN = "<START_OF_SENTENCE>"
    UNKNOWN_TOKEN = "<UNK>"
    UNKNOWN_TOKEN_THRESHOLD = 0
    SMOOTHING_VALUE = 0.1
    N_VALUE = 5
    # Total counts of labels
    label_counts = {'B-PER':0,
                       'I-PER':0,
                       'B-LOC':0,
                       'I-LOC':0,
                       'B-ORG':0,
                       'I-ORG': 0,
                       'B-MISC':0,
                       'I-MISC':0,
                       'O':0}

    # Map of specific tags to general tags, prevents sparsity
    label_type_map = {'B-PER': 'PER', 'I-PER': 'PER',
                      'B-LOC': 'LOC', 'I-LOC': 'LOC',
                      'B-ORG': 'ORG', 'I-ORG': 'ORG',
                      'B-MISC': 'MISC', 'I-MISC': 'MISC',
                      'O': 'O'}



    def __init__(self, test):
        """
        Initializes a baseline named entity recognition model. If test = True, then the model
        trains on the entire training set and tests on the test data, writing results in a
        file "hmm_results.csv". If test = False, then the model trains on "train_partial.txt"
        and validates on "validation_partial.txt".

        :param test: if the model should be tested or validated
        :param n: the n-gram model to u
        :return:
        """
        self.all_grams = Ngram(self.START_OF_SENTENCE_TOKEN)
        for label in self.label_type_map:
            self.words_labels_counts[label] = {}
            self.words_labels_counts[label][self.UNKNOWN_TOKEN] = 0
        if test:
            self.train("train.txt")
            self.test("test.txt")
        else:
            self.train("train_partial.txt")
            self.validate("validation_partial.txt")

    def train(self, train_data):
        """
        Train the Hidden Markov Model on training data. Extracts the tokens and tags and generates
        transition probabilities P(tag | previous tags) and emission probabilities P(word | tag).

        :param train_data: consisting of sentences, pos tags, and IOB tag.
        """
        with open(train_data, 'r') as train_data:
            while True:
                tokens = train_data.readline().split()
                pos = train_data.readline().split()
                labels = train_data.readline().split()
                if not tokens or not pos or not labels:
                    break
                # Generate transition probabilities
                for i in range(0, len(labels) - self.N_VALUE + 1):
                    self.add_label_sequence(labels[i:i + self.N_VALUE])
                # Generate lexical generation probabilities
                for i in range(0, len(tokens)):
                    token = tokens[i].lower()
                    label = labels[i]
                    self.add_word_tag(token, label)
        self.handle_unknowns()

    def handle_unknowns(self):
        """
        Aggregates all words in every label with a count less than UNKNOWN_TOKEN_THRESHOLD to
        an UNKNOWN_TAG.
        """
        for label in self.words_labels_counts:
            for word in self.words_labels_counts[label]:
                if self.words_labels_counts[label][word] <= self.UNKNOWN_TOKEN_THRESHOLD:
                    self.words_labels_counts[label][self.UNKNOWN_TOKEN] += self.words_labels_counts[label][word]
                    self.words_labels_counts[label][word] = 0

    def validate(self, validate_data):
        """
        Performs validation on a data set. Uses the Viterbi algorithm to determine most likely
        sequence of tags. Calculates the F-Score of the model

        :param validate_data: data to validate on
        """
        with open(validate_data, 'r') as validate_data:
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            result = {}
            for type in self.label_type_map:
                result[type] = []
            while True:
                tokens = validate_data.readline().split()
                pos = validate_data.readline().split()
                labels = validate_data.readline().split()
                if not tokens or not pos or not labels:
                    break
                # Classify all named entities in a sentence 85
                curr_results = self.viterbi(tokens)
                for i in range(0, len(labels)):
                    if curr_results[i] != 'O':
                        if labels[i] == 'O':
                            false_positive += 1 # Not 'O', but should be 'O'
                        else:
                            if self.label_type_map[labels[i]] == self.label_type_map[curr_results[i]]:
                                true_positive += 1 # Correct prediction
                    else:
                        if labels[i] == 'O':
                            true_negative += 1 # Correct prediction of 'O'
                        else:
                            false_negative += 1 # Predicted 'O', not 'O'
            # Calculate precision - TP / (TP + FP)
            precision = float(true_positive) / float(true_positive + false_positive)
            # Calculate recall - TP / (TP + FN)
            recall = float(true_positive) / float(true_positive + false_negative)
            # Calculate F-Score - 2 * P * R / (P + R)
            f_score = float(2*precision * recall) / float(precision + recall)
            print "Precision: " + str(precision)
            print "Recall: " + str(recall)
            print "F-score: " + str(f_score)


    def test(self, test_data):
        """
        Runs HMM on test data. Uses Viterbi algorithm to determine most likely sequence of tags.

        :param test_data: data to test on
        """
        with open(test_data, 'r') as test_data:
            results = {}
            for type in self.label_type_map:
                results[self.label_type_map[type]] = []
            while True:
                tokens = test_data.readline().split()
                pos = test_data.readline().split()
                indices = test_data.readline().split()
                if not tokens or not pos or not indices:
                    break
                curr_results = self.viterbi(tokens)
                intervals = self.extract_intervals(curr_results, indices)
                for type in intervals:
                    for interval in intervals[type]:
                        results[type].append(interval)
            self.write_results(results)

    def write_results(self, results):
        """
        Helper function to write results to a csv that is in a format suitable for
        Kaggle competition.

        :param results: from tag test data.
        """
        predictions = open('hmm_results.csv', 'w')
        predictions.write("Type,Prediction")
        for type in results:
            if type == 'O':
                continue
            predictions.write("\n" + str(type) + ",")
            for interval in results[type]:
                predictions.write(str(interval) + " ")
        predictions.close()

    def extract_intervals(self, labels, indices):
        """
        Extracts the intervals of a sequence of labels where labels are part of entities.

        :param labels: of a token sequence
        :param indices: global indices used by Kaggle
        :return: list of Intervals containing all entity tags within the sequence of labels
        """
        results = {}
        for type in self.label_type_map:
            results[self.label_type_map[type]] = []
        entity_index = 0
        while entity_index < len(indices):
            label = labels[entity_index]
            if label != 'O':
                type = self.label_type_map[label]
                interval = Interval(int(indices[entity_index]), int(indices[entity_index]))
                while entity_index < len(indices) and self.label_type_map[labels[entity_index]] == type:
                    entity_index += 1
                interval.end = int(indices[entity_index - 1])
                results[type].append(interval)
            else:
                entity_index += 1
        return results

    def viterbi(self, word_seq):
        """
        Runs the Viterbi algorithm for a given word sequence to determine the most probable
        sequence of tags.

        :param word_seq: to determine tag for
        :return: list of most likely sequence of tags
        """
        # Initialize scores
        scores = [{}]
        path = {}
        # Populate scores
        for i in range(0, len(word_seq)):
            for label in self.label_type_map:
                scores[i][label] = 0
            scores.append({})
        self.initialize(scores, word_seq, path)
        path = self.iterate(scores, word_seq, path)
        return self.identify(scores, word_seq, path)

    def initialize(self, scores, word_seq, path):
        """
        Initialize the probabilities of tags for a word sequence.

        :param scores: most likely probabilities for a word sequence at iteration 0
        :param word_seq: sequence of words to determine tags for
        :param path: the current paths
        """
        for label in self.label_type_map:
            label_prob = self.get_ngram_prob([label])
            lexical_generation_prob = self.get_lexical_generation_prob(word_seq[0], label)
            # scores[0][label] = label_prob * lexical_generation_prob
            scores[0][label] = math.log(lexical_generation_prob)
            path[label] = [label]

    def iterate(self, scores, word_seq, path):
        """
        Iterate through all tags to determine most likely sequence of tags using the Viterbi
        algorithm

        :param scores: most likely probabilities for a word sequence at each iteration
        :param word_seq: sequence of words to determine tags for
        :param path: the current paths
        :return: the most likely paths for each tag
        """
        for i in range(1, len(word_seq)):
            new_path = {}
            for label in self.label_type_map:
                prev_labels = path[label][:]
                # Conditioned on N previous terms
                if len(prev_labels) > self.N_VALUE:
                    prev_labels = prev_labels[-self.N_VALUE+1:]
                # Emission probability: P(w | t)
                lexical_generation_prob = self.get_lexical_generation_prob(word_seq[i], label)
                prob, state = float('-inf'), None
                # Iterate and find next state that is most probable
                for label0 in self.label_type_map:
                    # Transition probability P(t | t-1, t-2...t-n)
                    curr_label = prev_labels[:]
                    curr_label.append(label0)
                    label_prob = self.get_ngram_prob(curr_label)
                    # curr_prob, curr_state = scores[i-1][label0] * label_prob * lexical_generation_prob, label0
                    curr_prob, curr_state = scores[i-1][label0] + math.log(label_prob) + math.log(lexical_generation_prob), label0
                    # Update if a better state is found
                    if curr_prob > prob:
                        prob, state = curr_prob, curr_state
                scores[i][label] = prob
                new_path[label] = path[state] + [label]
            path = new_path
        return path

    def identify(self, scores, word_seq, path):
        """
        Identify the most likely path of tags of a given word sequence after running the
        Viterbi algorithm.

        :param scores: the final scores of the probability of the paths
        :param word_seq: sequence of words to determine tags for
        :param path: the final paths of most likely sequences of tags
        :return: the most likely sequence of tags.
        """
        n = len(word_seq) - 1
        (prob, state) = max((scores[n][label], label) for label in self.label_type_map)
        return path[state]

    def add_word_tag(self, token, label):
        """
        Adds a word associated with a label seen in training. Used to generate emission
        probabilities P(word | tag).

        :param token: word
        :param label: tag
        """
        # Add total count for label
        self.label_counts[label] += 1
        # Add count for word given label
        if token not in self.words_labels_counts[label]:
            self.words_labels_counts[label][token] = 1
        else:
            self.words_labels_counts[label][token] += 1

    def add_label_sequence(self, label_seq):
        """
        Adds a label sequence seen in training. Used to generate transition probabilities
        P(tag | previous tags)

        :param label_seq: sequence of labels
        """
        curr_ngram = self.all_grams
        for label in label_seq:
            curr_ngram.add_count()
            curr_ngram = curr_ngram.get_next_Ngram(label)
        # Add count for last label
        curr_ngram.add_count()


    def get_lexical_generation_prob(self, word, label):
        """
        Calculates the emission probability P(word | tag).

        :param word: word
        :param label: tag
        :return: probability of word given tag
        """
        word = word.lower()
        numer = self.SMOOTHING_VALUE
        if word in self.words_labels_counts[label] and self.words_labels_counts[label][word] != 0:
            numer += self.words_labels_counts[label][word]
        elif word in self.words_labels_counts[label]:
            numer += self.words_labels_counts[label][self.UNKNOWN_TOKEN]
        denom = self.label_counts[label] + self.SMOOTHING_VALUE * self.all_grams.get_count()
        return float(numer) / denom

    def get_ngram_prob(self, label_seq):
        """
        Calculates the transition probability P(tag | previous tags). The number of previous tags
        to use determines on the N_VALUE for n-grams.

        :param label_seq: sequence of labels
        :return: the n-gram probability
        """
        curr_ngram = self.all_grams
        for i in range(0, len(label_seq)):
            label = label_seq[i]
            if i == len(label_seq) - 1:
                denom = curr_ngram.get_count() + self.SMOOTHING_VALUE * 9
            curr_ngram = curr_ngram.get_next_Ngram(label)
        # For smoothing, just add self.SMOOTHING_VALUE
        numer = curr_ngram.get_count() + self.SMOOTHING_VALUE
        return float(numer) / denom


class Ngram:
    """
    A class representing Ngrams of labels. This class has a recursive implementation to allow for a
    varying n-value of ngrams. Each Ngram objects has mappings to other Ngram objects, n-1 number of
    times until n = 1 in which case the object contains a mapping to counts. The Ngram probability
    can be calculated by taking the nth Ngram object count dividing by the n-1th Ngram object.
    """
    def __init__(self, label):
        self.label = label
        # Mapping of label to Ngram object
        self.next_grams = {}
        self.count = 0

    def get_next_Ngram(self, label):
        """
        Get next Ngram object of a given label

        :param label:
        :return: next Ngram object
        """
        if label not in self.next_grams:
            self.next_grams[label] = Ngram(label)
        return self.next_grams[label]

    def has_next_Ngram(self, label):
        """
        Check if current Ngram object has the next Ngram object for a label.

        :param label: for next Ngram
        :return: true if this Ngram obejct has the next Ngram object
        """
        return label in self.next_grams

    def add_count(self):
        """
        Add count to current Ngram object.
        """
        self.count += 1

    def get_count(self):
        """
        Returns count of labels for a Ngram object
        :return: count of labels
        """
        return self.count

if __name__ == '__main__':
    HMM(False)
