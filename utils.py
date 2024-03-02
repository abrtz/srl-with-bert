import pandas as pd
import numpy as np

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
            # If the line starts with '#', it's a comment, ignore it
            if line[0].startswith('#'):
                continue
            elif line[0].strip() != '':
                # Create a token if its ID does not contain a period
                if '.' not in line[0] and len(line) > 10:
                    token = {
                        'form': line[1],
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
                    token['predicate'] = predicate.split('.')[0]

                # Keep only the relevant argument for this predicate. Overwrite 'V' with '_'.
                token['argument'] = token['argument'][index] if token['argument'][index] != 'V' else '_'
            expanded_sentences.append(sentence_copy)

    # Create a list to store each sentence.
    final_list = []
    # Iterate over all sentences after copy sentences for each predicate.
    for sentence in expanded_sentences:
        # Create empty lists for form and argument of sentence.
        form_list =[]
        argument_list =[]
        # For each word in sentence append features in their list
        # each list has all its feature for all words in sentence
        for word in sentence:
            form_list.append(word['form'])
            argument_list.append(word['argument'])
        # After all words in a sentence processed, append all list of sentence to final list as a dict.
        # Create input_form of each sentence like: sentence [SEP] predicate
        # Also argument is list of all argument in sentence (labels) 
        final_list.append({
                        'input_form': ' '.join(form_list)+' [SEP] '+str(sentence[0]['predicate']).split('.')[0],
                        'argument': argument_list})
    # Convert list to pandas dataframe
    df = pd.DataFrame(final_list)
    # Save Dataframe to output_path
    df.to_csv(output_path)
    # return Dataframe
    return df


#mapping labels to numbers
def map_labels_to_numbers(label_list,label_map):
    """Map a list of labels to corresponding numerical values based on a pre-defined label mapping.
    Return a list containing the mapped numerical values corresponding to the input labels.

    Parameters:
    - label_list (list): a list of labels to be mapped to numerical values.
    - label_map (dict): A dictionary mapping labels to numerical values.
    """
    
    mapped_labels = [label_map[label] for label in label_list if label in label_map]
    return mapped_labels


def map_labels_in_dataframe(df):
    """
    Explode the lists in a specified column of a DataFrame to obtain all labels, then map them to numerical values.
    Add a new column to the DataFrame with the mapped labels.
    Return a new DataFrame with an additional column containing the mapped numerical labels.
    
    Parameters:
    - df (DataFrame): input pandas DataFrame containing a column with lists of labels.
    """
    
    #exploding the lists in the column to get all the arguments
    exploded_df = df.explode('argument')

    #getting unique labels for all arguments
    labels = exploded_df['argument'].unique()
    labels.sort() #sorting the labels

    #moving '_' item to position 0
    labels = np.concatenate((['_'], np.delete(labels, np.where(labels == '_'))))

    #mapping the argument labels to a number
    label_map = {label: index for index, label in enumerate(labels)}

    #adding new column with mapped labels to the df
    df['mapped_labels'] = df['argument'].apply(lambda x: map_labels_to_numbers(x,label_map))
    
    return df


def tokenize_and_align_labels(tokenizer, dataset, label_all_tokens=True):
    """
    preprocess data and tokenize it using model tokenizer.
    Extract features from the data and return a datarame.

    Return Tokenized data.

    tokenizer (transformers AutoTokenizer): tokenizer of pretrained model.
    dataset (dataframe): dataframe of list of words and list of labels of each sentence.
    label_all_tokens (boolean): Taked from tutorial if True all tokens have thier own label (some words maybe splited to more than one token).    
    """

    # From tutorial with some changes to work with our data
    tokenized_inputs = tokenizer(dataset['input_form'].tolist(), truncation=True, is_split_into_words=True)

    labels = []
    # for i, label in enumerate(examples['argument']):
    for i, label in enumerate(dataset['mapped_labels'].tolist()):
        # print('label:', label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
