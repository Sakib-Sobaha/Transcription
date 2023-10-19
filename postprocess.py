

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


def remove_long_words(sentence, max_length=15):
    # Split the sentence into words
    words = sentence.split()
    
    # Filter out words that exceed the maximum length
    filtered_words = [word for word in words if len(word) <= max_length]
    
    # Join the filtered words back into a sentence
    modified_sentence = ' '.join(filtered_words)
    
    return modified_sentence


def postprocess_text(sentence):
    return remove_consecutive_words(remove_unncessary_characters(remove_long_words(sentence)))



if __name__ == "__main__":
    # Example usage:
    sentence = "This is a simple example. This this example demonstrates word word word frequency counting."

    modified_sentence = remove_consecutive_words(sentence)
    print(modified_sentence)