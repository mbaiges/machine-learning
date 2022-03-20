import re

from sklearn import pipeline

def lower(w):
    return w.lower()

ACCENT_REPL = {"á":"a", "é":"e", "í":"i", "ó":"o", "ú":"u"}
def remove_accent(w):
    for l in ACCENT_REPL:
        w = w.replace(l, ACCENT_REPL[l])
    return w

PUNCT = [",", ".", ";", "?", "¿", ":", "º", "!", "¡", "|", "\\", "/", "-", "+", "*", "=", "(", ")", "_", "&", "\"", "'", "[", "]", "¦", "\x9d", "\u00f1"]
def remove_punctuation(w):
    for l in PUNCT:
        w = w.replace(l, "")
        if w == "":
            break
    return w

NUMBER_PATTERN = "[0-9]"
def remove_numbers(w):
    w = "" if re.match(NUMBER_PATTERN, w) else w
    return w

SAVED = ["$", "%"]
def save_characters(w):
    for c in SAVED:
        if c in w:
            w = c
            break
    return w

REDUNDANT_WORDS = set({"los", "las", "ellos", "ellas", "les", "con", "una", "que", "por", "como", "del", "era", "hay", "sin", "dos", "para", "tres", "sus", "mis", "tus", "muy", "son", "asi", "mil", "esta", "mas", "nos", "otra", "otro", "eso", "aun", "eran", "cual", "otro"})
def remove_redundant_words(w):
    if w.isalpha() and len(w) <= 2:
        w = "" 
    else:
        if w in REDUNDANT_WORDS:
            w = ""
    return w


def normalize(text):
    words = []

    pipeline = [
        lower,
        remove_accent,
        remove_punctuation,
        save_characters,
        remove_numbers,
        remove_redundant_words
    ]

    for w in text.split(" "):
        nw = w
        for f in pipeline:
            if nw == "":
                break
            nw = f(nw)
        if nw != "":
            words.append(nw)

    return words