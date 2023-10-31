import torch

def remove_unncessary_characters(sentence):
    sentence = sentence.replace('�', '')
    return sentence


def remove_consecutive_words(sentence):
    # Split the sentence into words
    words = sentence.split()
    
    # Initialize variables to keep track of consecutive occurrences
    current_word = None
    consecutive_count = 0
    
    # Initialize a list to store the modified sentence
    modified_sentence = []
    
    for word in words:
        # Convert the word to lowercase for case-insensitive comparison
        # word = word.lower()
        
        if word == current_word:
            consecutive_count += 1
        else:
            if consecutive_count >= 3:
                modified_sentence.append(current_word)
            else:
                modified_sentence.extend([current_word] * consecutive_count)
            current_word = word
            consecutive_count = 1
    
    # Add the last consecutive occurrences to the modified sentence
    if current_word is not None:
        if consecutive_count >= 3:
            modified_sentence.append(current_word)
        else:
            modified_sentence.extend([current_word] * consecutive_count)
    
    # Join the modified sentence back into a string
    modified_sentence = ' '.join(modified_sentence)
    
    return modified_sentence


def fix_repetition(text, max_count):
    uniq_word_counter = {}
    words = text.split()
    for word in text.split():
        if word not in uniq_word_counter:
            uniq_word_counter[word] = 1
        else:
            uniq_word_counter[word] += 1

    for word, count in uniq_word_counter.items():
        if count > max_count:
            words = [w for w in words if w != word]
    text = " ".join(words)
    return text


def remove_long_words(sentence, max_length=15):
    # Split the sentence into words
    words = sentence.split()
    
    # Filter out words that exceed the maximum length
    filtered_words = [word for word in words if len(word) <= max_length]
    
    # Join the filtered words back into a sentence
    modified_sentence = ' '.join(filtered_words)
    
    return modified_sentence


def punctuate(text, models, tokenizer):
    PUNCT_WEIGHTS = [[1.0, 1.4, 1.0, 0.8]]
    
    input_ids = tokenizer(text).input_ids
    with torch.no_grad():
        model = models[0]
        logits = torch.nn.functional.softmax(
            model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1],
            dim=1).cpu()
        for model in models[1:]:
            logits += torch.nn.functional.softmax(
                model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1],
                dim=1).cpu()
        logits = logits / len(models)
        logits *= torch.FloatTensor(PUNCT_WEIGHTS)
        label_ids = torch.argmax(logits, dim=-1)

        tokens = tokenizer(text, add_special_tokens=False).input_ids
        punct_text = ""
        for index, token in enumerate(tokens):
            token_str = tokenizer.decode(token)
            if '##' not in token_str:
                punct_text += " " + token_str
            else:
                punct_text += token_str[2:]
            punct_text += ['', '।', ',', '?'][label_ids[index].item()]

    punct_text = punct_text.strip()
    return punct_text


def postprocess_text(sentence):
    postprocessed_text = remove_consecutive_words(remove_unncessary_characters(remove_long_words(sentence)))
    return fix_repetition(postprocessed_text, 3)


if __name__ == "__main__":
    # Example usage:
    sentence = "This is a simple example. This this example demonstrates word word word frequency counting."

    modified_sentence = remove_consecutive_words(sentence)
    print(modified_sentence)