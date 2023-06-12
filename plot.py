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
ID_3 = {
    'en': 22000,
    # 'ar': 12000,
    # 'bg': 18000,
    'de': 17000,
    # 'el': 25000,
    'es': 17000,
    'fr': 14500,
    # 'hi': 7000,
    # 'ru': 10000,
    # 'sw': 46000,
    # 'th': 150000,
    # 'tr': 19000,
    # 'ur': 6500,
    # 'vi': 13000,
    # 'zh': 13500
}

ID_2 = {
    'en': 36000,
    'ar': 12000,
    'bg': 18000,
    'de': 11000,
    'el': 25000,
    'es': 14000,
    'fr': 15000,
    'hi': 11000,
    'ru': 16000,
    'sw': 40000,
    'th': 500000,
    'tr': 21000,
    'ur': 6500,
    'vi': 12700,
    'zh': 13500
}

ID_1 = {
    'en': 18000,
    'ar': 12000,
    'bg': 15000,
    'de': 12000,
    'el': 15000,
    'es': 14000,
    'fr': 16000,
    'hi': 4000,
    'ru': 14000,
    'sw': 41000,
    'th': 150000,
    'tr': 14500,
    'ur': 4000,
    'vi': 12000,
    'zh': 14500
}

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

baselines = {
    'en': 81.637,
    'ar': 64.032,
    'bg': 68.124,
    'de': 70.06,
    'el': 66.607,
    'es': 73.653,
    'fr': 73.613,
    'hi': 60.12,
    'ru': 67.944,
    'sw': 50.539,
    'th': 53.513,
    'tr': 62.595,
    'ur': 58.583,
    'vi': 69.721,
    'zh': 68.922
}

ID_50_test = {
    'en': 500,
    'ar': 2000,
    'bg': 2000,
    'de': 500,
    'el': 2000,
    'es': 1000,
    'fr': 1000,
    'hi': 5000,
    'ru': 2000,
    # 'sw': 500000,
    # 'th': 200000,
    'tr': 10000,
    'ur': 5000,
    'vi': 500,
    'zh': 1000
}

pruning = {
    'en': 96.09,
    'ar': 90.46,
    'bg': 90.54,
    'de': 94.33,
    'el': 89.87,
    'es': 94.61,
    'fr': 93.98,
    'hi': 92.26,
    'ru': 92.59,
    # 'sw': 83.37,
    # 'th': 78.03,
    'tr': 87.09,
    'ur': 92.37,
    'vi': 93.87,
    'zh': 94.53
}

pruned_baselines = {
    'en': 78.443,
    'ar': 57.924,
    'bg': 90.54,
    'de': 94.33,
    'el': 89.87,
    'es': 94.61,
    'fr': 93.98,
    'hi': 92.26,
    'ru': 92.59,
    'sw': 83.37,
    'th': 78.03,
    'tr': 87.09,
    'ur': 92.37,
    'vi': 93.87,
    'zh': 94.53
}

ID_pruned = {
    'en': 35000,
    'ar': 5000,
    'bg': 14000,
    'de': 35000,
    'el': 12000,
    'es': 19000,
    'fr': 20000,
    'hi': 5000,
    'ru': 17000,
    'sw': 500,
    'th': 5000,
    'tr': 12000,
    'ur': 7000,
    'vi': 14000,
    'zh': 11000
}

ID_ar = {
    'en': 500000,
    'ar': 150000,
    'bg': 300000,
    'de': 400000,
    'el': 35000,
    'es': 200000,
    'fr': 35000,
    'hi': 12000,
    'ru': 200000,
    'sw': 20000,
    'th': 300000,
    'tr': 27000,
    'ur': 1700,
    'vi': 350000,
    'zh': 34000
}

ID_hi = {
    'en': 20000,
    'ar': 19000,
    'bg': 19000,
    'de': 35000,
    'el': 8000,
    'es': 20000,
    'fr': 40000,
    'hi': 19000,
    'ru': 18000,
    'sw': 7000,
    'th': 18000,
    'tr': 14000,
    'ur': 11000,
    'vi': 14000,
    'zh': 11000
}

ID_2_en = {
    'hi': 450,
    'de': 50,
    'ar': 650,
    'th': 6500,
}

ID_np = np.array(list(ID_3.values()))
ID_pruned_np = np.array(list(ID_pruned.values()))
corpus_np = np.array(list(corpus.values()))
baselines_np = np.array(list(baselines.values()))

plt.figure(0)
# plt.title('test acc vs prune fraction of mbert-base-256')
plt.xlabel(r'ID ($d_{90}$) in log scale')
plt.ylabel(r"corpus size (Wikisize)")
# plt.xticks(rotation=90)
# ax = plt.gca()
# plt.setp(ax.xaxis.get_minorticklabels(), rotation=50)
plt.semilogx(ID_pruned.values(), corpus_np, 'bo')
# for text in ax.get_xminorticklabels():
#     text.set_rotation(50)

for txt in corpus.keys():
    plt.annotate(txt, (ID_pruned[txt], corpus[txt]))

plt.show()

# print(list(ID_50_test.values()), type(ID_50_test.values()))
# print(np.corrcoef(np.array(np.log(list(ID_50_test.values()))), np.array(list(baselines.values()))))
# print(np.corrcoef(np.array(list(corpus.values())), np.array(list(ID.values()))))
print(np.corrcoef(corpus_np, np.log(np.array(list(ID_pruned.values())))))