import matplotlib.pyplot as plt
import numpy as np

loss = [515.97705078125, 453.982177734375, 409.95001220703125, 362.23272705078125, 316.176025390625, 282.418701171875, 242.54017639160156, 206.873046875, 175.29727172851562, 141.1794891357422, 109.41294860839844, 80.58006286621094, 52.360252380371094, 24.59552764892578, 0.0]

plt.figure(0)
plt.plot(list(range(1,16)), loss)
plt.show()

"""
(7 clusters using KMeans)
en 5                                                                                                                  [1/1972]
ar 4                                                                                                                          
bg 3
de 2
el 0
es 1
fr 1
hi 4
ru 3
sw 4
th 6
tr 2
ur 4
vi 4
zh 2

(5 clusters using KMeans)
en 3                                                                                                                  [1/1989]
ar 3
bg 2
de 1
el 4
es 3
fr 3
hi 0
ru 2
sw 0
th 0
tr 1
ur 0
vi 0
zh 1

(6 clusters KMeans)
en 5                                                                                                                  [1/1806]
ar 0                                                                                                                          
bg 4
de 1
el 2
es 0
fr 0
hi 2
ru 4
sw 2
th 3
tr 1
ur 2
vi 2
zh 1

(6 clusters Spectral rbf)
en 5                                                                                                                 [61/1965]
ar 3                                                                                                                          
bg 3                                                                                                                          
de 4                                                                                                                          
el 1                                                                                                                          
es 3                                                                                                                          
fr 3                                                                                                                          
hi 2                                                                                                                          
ru 3                                                                                                                          
sw 3                                                                                                                          
th 0                                                                                                                          
tr 4                                                                                                                          
ur 3                                                                                                                          
vi 3
zh 4
"""