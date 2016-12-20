import nltk

hypothesis = ["Clearing", "items", "find", "controls", "in", "c", "#"]
reference = ["c", "#", "WPF", "How", "to", "change", "only", "the", "content", "?"]

#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=[1, ])
print BLEUscore

#print 5.0 / 50 * 10
