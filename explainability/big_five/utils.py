# Test that the model is correct
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW
from datasets import load_metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
auc_metric = load_metric("roc_auc", trust_remote_code=True)

from transformers import BertTokenizerFast, RobertaTokenizerFast
from typing import Optional, List, Union
from transformers.tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    EncodedInput,
    TensorType
)
from transformers.utils import PaddingStrategy, add_end_docstrings, logging

# Define mbti words
mbti_words = {'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ENFJs', 'ENFPs', 'ENTJs', 'ENTPs',
              'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'ESFJs', 'ESFPs', 'ESTJs', 'ESTPs',
              'INFJ', 'INFP', 'INTJ', 'INTP', 'INFJs', 'INFPs', 'INTJs', 'INTPs',
              'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ISFJs', 'ISFPs', 'ISTJs', 'ISTPs',
              'INFX', 'INFx', 'INFJness', 'ISFX', 'ISFx', 'ESxx', 'Ixxx', 'ISTJish', 'EXFJ',
              'E', 'N', 'F', 'J', 'P',
              'Es', 'Ns', 'Ss', 'Fs', 'Ts', 'Js', 'Ps',
              'EN', 'FJ', 'FP', 'TJ', 'TP',
              'ENs', 'FJs', 'FPs', 'TJs', 'TPs',
              'NE', 'NI', 'SI', 'TE',
              'NEs', 'NIs', 'SIs', 'TEs',
              'TI', 'FI',
              'TIs', 'FIs',
              'NF', 'NT', 'SF', 'ST',
              "SE", "FE", "PE", "JE",
              "SEs", "FEs", "PEs", "JEs",
              'NFs', 'NTs', 'TJs', 'SFs', 'STs',
              'ENF', 'ENT', 'ESF', 'EST',
              'ENFs', 'ENTs', 'ESFs', 'ESTs',
              'NFJ', 'NFP', 'NTJ', 'NTP',
              'NFJs', 'NFPs', 'NTJs', 'NTPs',
              'SFJ', 'SFP', 'STJ', 'STP',
              'SFJs', 'SFPs', 'STJs', 'STPs',
              'INF', 'INT', 'ISF', 'IST',
              'INFs', 'INTs', 'ISFs', 'ISTs'}

class MyBertTokenizerFast(BertTokenizerFast):

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, truncation=True, max_length=512, **kwargs).tokens()

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = 512,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

class MyRobertaTokenizerFast(RobertaTokenizerFast):

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, truncation=True, max_length=512, **kwargs).tokens()

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = 512,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

def getTokenizer(model_name):
    if model_name == "bert-base-uncased":
        return MyBertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    elif model_name == "FacebookAI/roberta-base":
        return MyRobertaTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    else:
        return AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

def compute_metrics(eval_pred):
    # Calculate accuracy
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    conf_matrix = confusion_matrix(labels, predictions)

    # Calculate AUC
    roc_auc = auc_metric.compute(prediction_scores=predictions, references=labels)["roc_auc"]

    return {"accuracy": accuracy, "roc_auc": roc_auc, "conf_matrix": conf_matrix}

def create_partition_from_split(train, test, x_colname, y_colname):
    X_train = train[x_colname]
    X_test = test[x_colname]
    y_train = train[y_colname]
    y_test = test[y_colname]
    return X_train, X_test, y_train, y_test

def create_datasets(tokenizer, X_train, X_test, y_train, y_test):
    # Convert input dataframes into lists
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    # Tokenize the text
    train_tokens = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    valid_tokens = tokenizer(X_test, truncation=True, padding=True, max_length=512)

    class MakeTorchData(torch.utils.data.Dataset):
        def __init__(self, tokens, labels):
            self.tokens = tokens
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.tokens.items()}
            item["labels"] = torch.tensor([self.labels[idx]])
            return item

        def __len__(self):
            return len(self.labels)

    # Convert our tokenized data into a torch Dataset
    train_dataset = MakeTorchData(train_tokens, y_train)
    valid_dataset = MakeTorchData(valid_tokens, y_test)

    return train_dataset, valid_dataset

def get_first_n_words(original_string, n):
    words = original_string.split()
    first_n_words = ' '.join(words[:n])
    return first_n_words

def truncate_text(tokenizer, text, max_length=512):
    length = len(text.split())
    while len(tokenizer.encode(get_first_n_words(text, length))) > max_length:
        length = int(length/1.25)
    if length < len(text.split()):
        print("WARNING: text needed to be truncated from", len(text.split()), "words to", length)
    return get_first_n_words(text, length)

def get_preproc_text(df, row_index, tokenizer, max_length, text_colname="TEXT"):
    text = df.loc[row_index, text_colname]  # Assuming 'TEXT' is the correct column name
    preproc_text = truncate_text(tokenizer, text, max_length)
    #preproc_text = text
    return preproc_text

def get_all_word_attr_fname_essays(trait):
    return 'all_word_attributions_essays_' + trait.replace("/", "_") + '.pkl'

def get_all_sum_attr_fname_essays(trait):
    return 'all_sum_attributions_essays_' + trait.replace("/", "_") + '.pkl'

def get_all_word_attr_fname_mbti(trait):
    return 'all_word_attributions_mbti_' + trait.replace("/", "_") + '.pkl'

def get_all_sum_attr_fname_mbti(trait):
    return 'all_sum_attributions_mbti_' + trait.replace("/", "_") + '.pkl'

def get_all_word_attr_fname_masked_mbti(trait):
    return 'all_word_attributions_masked_mbti_' + trait.replace("/", "_") + '.pkl'

def get_all_sum_attr_fname_masked_mbti(trait):
    return 'all_sum_attributions_masked_mbti_' + trait.replace("/", "_") + '.pkl'

def get_word_freqs(attributions, pred_class=None):
    freq = {}
    for attributions, true_class, predicted_class in attributions:
        if pred_class is None or pred_class==predicted_class:
            for word, score in attributions:
                if word in freq:
                    freq[word] = freq[word] + 1
                else:
                    freq[word] = 1
    return freq

def filter_attributions(attributions, sign, label, force_correct_classif=False):
    def correct_classif(predicted_class, true_class):
        modif_pred_class = 1
        if predicted_class == "LABEL_0":
            modif_pred_class = 0
        return modif_pred_class == true_class

    attrs = []
    for attributions, true_class, predicted_class in attributions:
        if predicted_class == label:
            if not force_correct_classif or (force_correct_classif and correct_classif(predicted_class, true_class)):
                for word, score in attributions:
                    if sign == '+':
                        if score > 0:
                            attrs.append((word, score))
                    else:
                        if score < 0:
                            attrs.append((word, abs(score)))
    return attrs

def filter_stop_words(stopwords, word_score_pairs):
    return [(word, score) for word, score in word_score_pairs if word not in stopwords]

def filter_words(valid_words, stopwords, word_freq, word_score_pairs,
                 min_word_freq=5, min_word_len=4, words_to_keep=[]):

    # Convert words to keep to lowercase
    words_to_keep_lower = {word.lower() for word in words_to_keep}

    result = []
    for word, score in word_score_pairs:
        if word.lower() in words_to_keep_lower:
            result.append((word, score))
        else:
            if word not in stopwords and \
            (valid_words is None or word in valid_words) and \
            word_freq[word] >= min_word_freq and len(word)>=min_word_len:
                result.append((word, score))
    return result

def calc_freqs(word_score_pairs):
    result = {}
    for word, score in word_score_pairs:
        if word in result:
            result[word] = result[word] + 1
        else:
            result[word] = 1
    return result

def accum_scores(word_score_pairs):
    result = {}
    for word, score in word_score_pairs:
        if word in result:
            result[word] = result[word] + score
        else:
            result[word] = score
    return result

def avg_scores(word_score_pairs):
    accum = {}
    freq = {}
    result = {}
    for word, score in word_score_pairs:
        if word in result:
            accum[word] = accum[word] + score
            freq[word] = freq[word] + 1
        else:
            accum[word] = score
            freq[word] = 1

    for word in accum:
        result[word] = accum[word]/freq[word]

    return result

def max_scores(word_score_pairs):
    result = {}
    for word, score in word_score_pairs:
        if word in result:
            if score > result[word]:
                result[word] = score
        else:
            result[word] = score
    return result

def geom_mean_acc_max_scores(word_score_pairs):
    acc_scrs = accum_scores(word_score_pairs)
    max_scrs = max_scores(word_score_pairs)
    result = {}
    for word, _ in word_score_pairs:
        geom_mean = math.sqrt(acc_scrs[word] * max_scrs[word])
        result[word] = geom_mean
    return result

def summarize_scores(word_score_pairs, technique="accum"):
    if technique == "accum":
        return accum_scores(word_score_pairs)
    elif technique == "freq":
        return calc_freqs(word_score_pairs)
    elif technique == "max":
        return max_scores(word_score_pairs)
    elif technique == "avg":
        return avg_scores(word_score_pairs)
    elif technique == "gmean":
        return geom_mean_acc_max_scores(word_score_pairs)
    else:
        return None

def filter_sum_attributions(sum_attributions, true_class=None, pred_class=None, get_attr_plus_idx=False):
    filtered_sum_attributions = []
    for i in range(len(sum_attributions)):
        sum_attr = sum_attributions[i]
        curr_attr, curr_true_class, curr_pred_class = sum_attr[0], sum_attr[1], sum_attr[2]
        if (pred_class is None or pred_class==curr_pred_class) and (true_class is None or true_class==curr_true_class):
            if get_attr_plus_idx:
                filtered_sum_attributions.append((sum_attr[0],i))
            else:
                filtered_sum_attributions.append(sum_attr[0])
    return filtered_sum_attributions

def get_max_word_score(word_of_interest, word_attributions):
    word_scr_pairs = word_attributions[0]
    max_score = None
    for word, score in word_scr_pairs:
        if word_of_interest == word:
            if max_score is None or max_score < score:
                max_score = score
    return max_score

def get_max_word_scores(word_of_interest, all_word_attributions, pred_class = None):
    result = []
    for i in range(len(all_word_attributions)):
        word_attributions, true_class, predicted_class = all_word_attributions[i]
        if pred_class is None or pred_class == predicted_class:
            max_score = get_max_word_score(word_of_interest, all_word_attributions[i])
            if max_score is not None:
                result.append((max_score, i))
    return result

# Function to plot horizontal bar plot with top n significant words
def plot_bar(words_scores, title, n=15, filename = None):
    # Sort the words_scores based on scores
    sorted_words_scores = sorted(words_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top n significant words and scores
    top_n_words_scores = sorted_words_scores[:n]
    bottom_n_words_scores = sorted_words_scores[-n:]  # Select the bottom n significant words and scores

    # Extract top words and scores
    top_words = [word for word, score in top_n_words_scores]
    top_scores = [score for word, score in top_n_words_scores]

    # Extract bottom words and scores
    bottom_words = [word for word, score in bottom_n_words_scores]
    bottom_scores = [score for word, score in bottom_n_words_scores]

    fig = plt.figure(figsize=(14, 8))

    # Plot top words
    plt.subplot(1, 2, 1)
    plt.barh(top_words, top_scores, color='skyblue')
    plt.xlabel('Scores')
    plt.ylabel('Words')
    plt.title(f'Top {n} Significant Words - {title}')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest scores at the top

    # Plot bottom words
    plt.subplot(1, 2, 2)
    plt.barh(bottom_words, bottom_scores, color='lightcoral')
    plt.xlabel('Scores')
    plt.ylabel('Words')
    plt.title(f'Bottom {n} Significant Words - {title}')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest scores at the top

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi = 300, bbox_inches = "tight")
        plt.close(fig)
        print(f"Saved {filename}")
    else:
        plt.show()
