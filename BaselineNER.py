__author__ = 'philipsu'


class BaselineNER:
    """
    Simple naive Baseline Named Entity Recognition system, memorizes training data
    and classifies test data based on training data
    """
    # Mapping of first words to list of entities and tags
    named_entities = {}
    all_types = ["PER", "LOC", "ORG", "MISC"]
    ENTITY_MAX_LEN = 2

    def __init__(self, test):
        """
        Initializes a baseline named entity recognition model. If test = True, then the model
        trains on the entire training set and tests on the test data, writing results in a
        file "baseline_results.csv". If test = False, then the model trains on "train_partial.txt"
        and validates on "validation_partial.txt".

        :param test: If the model should be tested or validated
        """
        if test:
            self.train("train.txt")
            self.test("test.txt")
        else:
            self.train("train_partial.txt")
            self.validate("validation_partial.txt")



    def extract_entities(self, tokens, labels):
        """
        Extracts named entities from training data, adding the word into the correct entity
        type in named_entities map.

        :param tokens: list of tokens
        :param labels: list of labels
        :return: mapping of all words to their types
        """
        entity_word = ""
        entity_type = ""
        entity_index = 0
        results = {}
        # Find first index of entity
        while entity_index < len(labels) and labels[entity_index][0] != 'B':
            entity_index += 1
        while entity_index < len(labels):
            # Continuation of named entity
            if labels[entity_index][0] == 'I':
                entity_word += " " + tokens[entity_index]
            # Other 2 tags require special handling, must end current entity
            elif labels[entity_index][0] == 'B' or labels[entity_index][0] == 'O':
                # Save previously created entity
                if len(entity_word) > 0:
                    results[entity_word] = entity_type
                # On 'B' tag, start of new entity
                if labels[entity_index][0] == 'B':
                    entity_word = tokens[entity_index]
                    entity_type = labels[entity_index][2:]
                # On 'O' tag, no entity here
                else:
                    entity_word = ""
                    entity_type = ""
            entity_index += 1
        # In case entity ends at end of sentence
        if len(entity_word) > 0:
            results[entity_word] = entity_type
        return results

    def train(self, train_data):
        """
        Train the baseline named entity recognition model on training data. Extracts the first
        word of all named entities and stores the named entity as well as the tag.

        :param train_data: consisting of sentences, pos tags, and IOB tag.
        """
        with open(train_data, 'r') as train_data:
            while True:
                tokens = train_data.readline().split()
                pos = train_data.readline().split()
                labels = train_data.readline().split()
                if not tokens or not pos or not labels:
                    break
                # Extract mapping of entities to types in sentence
                entity_type_map = self.extract_entities(tokens, labels)
                for entity in entity_type_map:
                    # Add each entity to a dictionary based on first word
                    first_word = entity.split()[0]
                    if first_word in self.named_entities:
                        self.named_entities[first_word].append((entity, entity_type_map[entity]))
                    else:
                        self.named_entities[first_word] = [(entity, entity_type_map[entity])]

    def validate(self, validate_data):
        """
        Performs validation on a data set. Calculates the accuracy (precision) of the model.

        :param validate_data: data to validate on
        """
        with open(validate_data, 'r') as validate_data:
            predictions = open('baseline_results_validation.csv', 'w')
            predictions.write("Type,Prediction")
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            result = {}
            for type in self.all_types:
                result[type] = []
            global_index = 0
            while True:
                tokens = validate_data.readline().split()
                pos = validate_data.readline().split()
                labels = validate_data.readline().split()
                if not tokens or not pos or not labels:
                    break
                # Classify all named entities in a sentence
                curr_results = self.classify_ner(tokens, None)
                for type in curr_results:
                    for interval in curr_results[type]:
                        start, end = interval.get_start(), interval.get_end()
                        if labels[start][2:] == type:
                            correct_count += 1
                        pred_count += 1
                        interval.set_start(global_index)
                        interval.set_end(global_index)
                        result[type].append(interval)
                global_index += len(tokens)
            # Write results
            for type in result:
                predictions.write("\n" + str(type) + ",")
                for interval in result[type]:
                    predictions.write(str(interval) + " ")
            predictions.close()
            print "Correct: " + str(correct_count) + ", Total: " + str(pred_count)
            print "Accuracy: " + str(float(correct_count) / pred_count)

    def test(self, test_data):
        """
        Runs BaselineNER on test data. Writes results to "baseline_rsults.csv" in a suitable format
        for Kaggle competition.

        :param test_data: data to test on
        """
        with open(test_data, 'r') as test_data:
            predictions = open('baseline_results.csv', 'w')
            predictions.write("Type,Prediction")
            result = {}
            for type in self.all_types:
                result[type] = []
            while True:
                tokens = test_data.readline().split()
                pos = test_data.readline().split()
                indices = test_data.readline().split()
                if not tokens or not pos or not indices:
                    break
                # Classify all named entities in a sentence
                curr_results = self.classify_ner(tokens, indices)
                for type in curr_results:
                    for interval in curr_results[type]:
                        result[type].append(interval)
            # Write results
            for type in result:
                predictions.write("\n" + str(type) + ",")
                for interval in result[type]:
                    predictions.write(str(interval) + " ")
            predictions.close()

    def classify_ner(self, tokens, indices):
        """
        Classifies a list of tokens into named entities. Returns a list of intervals that are
        grouped by label type.

        :param tokens: sentence of test or validation data
        :return list of intervals grouped by tag type.
        """
        entity_index = 0
        result = {}
        for type in self.all_types:
            result[type] = []
        while entity_index < len(tokens):
            entity_word = tokens[entity_index]
            # Check if first word has been seen
            if entity_word in self.named_entities:
                for data in self.named_entities[entity_word]:
                    entities, type = data
                    # See if any before seen named-entities match this
                    if self.matches(entities.split(), tokens[entity_index:]):
                        if indices:
                            interval = Interval(int(indices[entity_index]),
                                            int(indices[entity_index]) + len(entities.split()) - 1)
                        else:
                            interval = Interval(entity_index, entity_index + len(entities.split()) - 1)
                        result[type].append(interval)
                        entity_index += len(entities.split())
                        break

            entity_index += 1
        return result

    def matches(self, train_data, test_data):
        """
        Checks if 2 lists of words match
        """
        for i in range(0, len(train_data)):
            if i >= len(test_data):
                pass
            if i >= len(test_data) or test_data[i] != train_data[i]:
                return False
        return True

class Interval:
    """
    Simple wrapper class to create intervals for writing result values
    """
    start = -1
    end = -1
    def __init__(self, start=-1, end=-1):
        self.start = start
        self.end = end

    def __str__(self):
        return str(self.start) + "-" + str(self.end)

    def set_start(self, offset):
        self.start += offset

    def set_end(self, offset):
        self.end += offset

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

if __name__ == '__main__':
    NER = BaselineNER(False)
