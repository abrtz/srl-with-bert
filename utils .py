import pandas as pd


def read_data_as_sentence(file_path, output_path):
    """
    Parses the CoNNL-U Plus file and returns a dataframe of sentences.
    Extract features from the data and return a datarame.

    Returns a dataframe, where each row represents one sentence with its all words and all features of words (each columns is a list with lengh of number of words in sentence).

    file_path (str): The file path to the data to be preprocessed.
    output_path (str): The file path to the save processed dataframe.
    """

    sentences = []
    sentence = []  # Initialize an empty list for the current sentence
    with open(file_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip().split('\t')
            if line[0].startswith('#'):
                # If the line starts with '#', it's a comment, ignore it

                continue
            elif line[0].strip() != '':
                # Split the features string into a dictionary
                features = dict()
                for feature in line[5].split('|'):
                    key_value_pair = feature.split('=')

                    # Check if the split result is valid, if it is, add it to the dictionary
                    if len(key_value_pair) == 2:
                        key, value = key_value_pair
                        features[key] = value

                # Create a token if its ID does not contain a period
                if '.' not in line[0] and len(line) > 10:
                    token = {
                        'id': line[0],
                        'form': line[1],
                        'lemma': line[2],
                        'upos': line[3],
                        'xpos': line[4],
                        'features': features,
                        'head': line[6],
                        'dependency_relation': line[7],
                        'dependency_graph': line[8],
                        'miscellaneous': line[9],
                        'predicate': line[10],
                        'argument': line[11:]  # Store all remaining elements as arguments.
                    }
                    # Append the token to the sentence.
                    sentence.append(token)

            # A new line indicates the end of a sentence.
            elif line[0].strip() == '':
                # Append the completed sentence to the sentences list.
                sentences.append(sentence)
                # Reset sentence for the next sentence.
                sentence = []

    # Iterate over all sentences. Create copies of sentences for each predicate.
    tokens_list = []
    expanded_sentences = []
    for sentence in sentences:
        # Find all predicates in the sentence.
        predicates = [token['predicate'] for token in sentence if token['predicate'] != '_']

        # for every predicate, create a copy of the sentence.
        for index, predicate in enumerate(predicates):
            sentence_copy = [token.copy() for token in sentence]
            for token in sentence_copy:
                # Keep only this predicate.
                if token['predicate'] != predicate:
                    token['predicate'] = predicate.split('.')[0]#'_'

                # Keep only the relevant argument for this predicate. Overwrite 'V' with '_'.
                token['argument'] = token['argument'][index] if token['argument'][index] != 'V' else '_'
            tokens_list.append(sentence_copy)
            expanded_sentences.append(sentence_copy)
    # Create a list to store each sentence.
    final_list = []
    # Iterate over all sentences after copy sentences for each predicate.
    for sentence in expanded_sentences:
        # Create empty lists for eah feature of data.
        id_list =[]
        form_list =[]
        lemma_list =[]
        upos_list =[]
        xpos_list =[]
        featues_list =[]
        head_list =[]
        dependency_relation_list =[]
        dependency_graph_list =[]
        miscellaneous_list =[]
        predicate_list =[]
        argument_list =[]
        # For each word in sentence append features in their list
        # each list has all its feature for all words in sentence
        for word in sentence:
            id_list.append(word['id'])
            form_list.append(word['form'])
            lemma_list.append(word['lemma'])
            upos_list.append(word['upos'])
            xpos_list.append(word['xpos'])
            featues_list.append(word['features'])
            head_list.append(word['head'])
            dependency_relation_list.append(word['dependency_relation'])
            dependency_graph_list.append(word['dependency_graph'])
            miscellaneous_list.append(word['miscellaneous'])
            predicate_list.append(word['predicate'])
            argument_list.append(word['argument'])
        # After all words in a sentence processed, append all list of sentence to final list as a dict.
        final_list.append({'id': id_list,
                        'input_form': '[CLS] '+' '.join(lemma_list)+' [SEP] '+str(predicate_list[0])+' [SEP]',
                        'form': form_list,
                        'lemma': lemma_list,
                        'upos': upos_list,
                        'xpos': xpos_list,
                        'features': featues_list,
                        'head': head_list,
                        'dependency_relation': dependency_relation_list,
                        'dependency_graph': dependency_graph_list,
                        'miscellaneous': miscellaneous_list,
                        'predicate': predicate_list,
                        'argument': argument_list})
    # Convert list to pandas dataframe
    df = pd.DataFrame(final_list)
    # Save Dataframe to output_path
    df.to_csv(output_path)
    # return Dataframe
    return df
