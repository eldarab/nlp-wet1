
def add_or_append(dictionary, item):
    if item not in dictionary:
        dictionary[item] = 1
    else:
        dictionary[item] += 1


def parse_lower(word_tag):
    word, tag = word_tag.split('_')
    return word.lower(), tag


def get_words_arr(line):
    words_tags_arr = line.split(' ')
    if len(words_tags_arr) == 0:
        return
    words_tags_arr[-1] = words_tags_arr[-1][:-1]  # removing \n from end of line
    return words_tags_arr


def has_digit(word):
    for char in word:
        if char.isdigit():
            return True


def has_hyphen(word):
    for char in word:
        if char == '-':
            return True
