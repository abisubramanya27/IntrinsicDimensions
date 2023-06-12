import matplotlib.pyplot as plt
import numpy as np

'''
Plot for Ideal Pruning fraction identification
'''
# x = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
# y = np.array([80.998, 80.699, 79.242, 78.443, 43.952])

# plt.figure(0)
# plt.title('test acc vs prune fraction of mbert-base-256')
# plt.xlabel('prune fraction')
# plt.ylabel('test accuracy')
# plt.plot(x,y)
# plt.show()

'''
Plot for correlation between ID and Corpus size
'''
corpus = {
    'en': 14,
    'ar': 10,
    'bg': 8,
    'de': 12,
    'el': 8,
    'es': 12,
    'fr': 12,
    'hi': 7,
    'ru': 12,
    'sw': 5,
    'th': 8,
    'tr': 9,
    'ur': 7,
    'vi': 9,
    'zh': 11
}

baseline_acc = {
    'ar': 70.988,
    'hi': 66.148,
    'th': 65.709,
    'de': 75.369,
    'en': 81.637,
    'sw': 64.431,
}

ID = {
    'ar': 18000,
    'hi': 18000,
    'th': 35000,
    'de': 19500,
    'en': 22000,
    'sw': 41000,
}

ID_sw = {
    'hi': 1350,
    'ar': 950,
    'de': 3500,
}

ID_en = {
    'hi': 450,
    'ar': 650,
    'th': 6500,
    'de': 40,
}

ID_hi = {
    'ar': 270,
    'de': 70,
}

ID_ar = {
    'de': 100,
}

ID_th = {
    'de': 3400,
}

ID_50_en = {
    'ar': 500,
}

corpus = {key: corpus[key] for key in baseline_acc.keys()}

plt.figure(0)
# plt.title('test acc vs prune fraction of mbert-base-256')
plt.ylabel(r'ID ($d_{90}$) in log scale')
plt.xlabel(r"corpus size (Wikisize)")
# plt.xticks(rotation=90)
# ax = plt.gca()
# plt.setp(ax.xaxis.get_minorticklabels(), rotation=50)
plt.semilogx(ID.values(), corpus.values(), 'bo')
# for text in ax.get_xminorticklabels():
#     text.set_rotation(50)

for txt in ID.keys():
    plt.annotate(txt, (ID[txt], corpus[txt]))

plt.show()

# print(list(ID_50_test.values()), type(ID_50_test.values()))
# print(np.corrcoef(np.array(np.log(list(ID_50_test.values()))), np.array(list(baselines.values()))))
# print(np.corrcoef(np.array(list(corpus.values())), np.array(list(ID.values()))))
print(np.corrcoef(np.log(np.array(list(ID.values()))), np.array(list(corpus.values()))))