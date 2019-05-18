import probabilities
import frequencies
import directions
import fremen
import draw_arrows
import numpy as np
from time import clock






"""
structure = [2, [], []]
path  = '../data/two_weeks_days_nights0_weekends.txt'
clusters = 4
edges_of_cell = np.array([1200.0, 0.5, 0.5])
#dirs = probabilities.Directions(18, [4, [1.0, 1.0], [86400.0, 604800.0]], '../data/two_weeks_days_nights_weekends_with_dirs.txt')
freqs = frequencies.Frequencies(clusters, edges_of_cell, structure, path)
frem = fremen.Fremen()
P = frem.find_prominent(path, freqs)
print(P)
err = freqs.rmse(path, plot_it=False)
print(err)
structure[1].append(1.0)
structure[2].append(P)

freqs = frequencies.Frequencies(clusters, edges_of_cell, structure, path)
frem = fremen.Fremen()
P = frem.find_prominent(path, freqs)
print(P)
err = freqs.rmse(path, plot_it=False)
print(err)
structure[1].append(1.0)
structure[2].append(P)

freqs = frequencies.Frequencies(clusters, edges_of_cell, structure, path)
frem = fremen.Fremen()
P = frem.find_prominent(path, freqs)
print(P)
err = freqs.rmse(path, plot_it=False)
print(err)
structure[1].append(1.0)
structure[2].append(P)

freqs = frequencies.Frequencies(clusters, edges_of_cell, structure, path)
frem = fremen.Fremen()
P = frem.find_prominent(path, freqs)
print(P)
err = freqs.rmse(path, plot_it=False)
print(err)
structure[1].append(1.0)
structure[2].append(P)
"""

"""
start = clock()
#dirs4 = directions.Directions(3, [2, [1.0, 1.0, 1.0, 1.0], [21600.0, 43200.0, 86400.0, 604800.0]], '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
dirs3 = directions.Directions(3, [2, [1.0, 1.0, 1.0], [21600.0, 43200.0, 86400.0]], '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
print(clock() - start)

X_t, target_t = dirs3.transform_data('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
y_t = dirs3.predict(X_t)
pomer = np.sum(target_t)/np.sum(y_t)
print('pomer :' + str(pomer))


#X_s, target_s = dirs3.transform_data('/home/tom/pro/my/WHyTeS/data/sergi_s_output/all_from_testing_times/two_test_days_24_25_may_1m_10min.txt')
X_n, target_n = dirs3.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
X_d, target_d = dirs3.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_days_with_angles_plus_reversed.txt')

y = dirs3.predict(X_n)
#np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/nights_WHyTeS_3clusters_3periods.txt', y)
print('WHyTe, nights, 3 periods')
print(np.sum(target_n)/np.sum(y))
print(np.sqrt(np.mean((y - target_n) ** 2.0)))
print(np.sqrt(np.mean((y*pomer - target_n) ** 2.0)))
print(np.sqrt(np.mean((y/pomer - target_n) ** 2.0)))

y = dirs3.predict(X_d)
#np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/days_WHyTeS_3clusters_3periods.txt', y)
print('WHyTe, days, 3 periods')
print(np.sum(target_d)/np.sum(y))
print(np.sqrt(np.mean((y - target_d) ** 2.0)))
print(np.sqrt(np.mean((y*pomer - target_d) ** 2.0)))
print(np.sqrt(np.mean((y/pomer - target_d) ** 2.0)))

#y = dirs3.predict(X_s)
#np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/1m_10min_WHyTeS_3clusters_3periods.txt', y)
"""



"""
dirs0 = directions.Directions(3, [2, [], []], '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
dirs2 = directions.Directions(3, [2, [1.0, 1.0], [86400.0, 604800.0]], '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')

X_s0, target_s0 = dirs0.transform_data('/home/tom/pro/my/WHyTeS/data/sergi_s_output/two_test_days_24_25_may_1m_10min.txt')
X_n0, target_n0 = dirs0.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
X_d0, target_d0 = dirs0.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_days_with_angles_plus_reversed.txt')
X_s2, target_s2 = dirs2.transform_data('/home/tom/pro/my/WHyTeS/data/sergi_s_output/two_test_days_24_25_may_1m_10min.txt')
X_n2, target_n2 = dirs2.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
X_d2, target_d2 = dirs2.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_days_with_angles_plus_reversed.txt')

y = dirs0.predict(X_s0)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/1m_10min_WHyTeS_3clusters_0periods.txt', y)

y = dirs2.predict(X_s2)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/1m_10min_WHyTeS_3clusters_2periods.txt', y)

y = dirs0.predict(X_n0)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/nights_WHyTeS_3clusters_0periods.txt', y)
print('WHyTe, nights, 0 periods')
print(np.sqrt(np.mean((y - target_n0) ** 2.0)))

y = dirs2.predict(X_n2)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/nights_WHyTeS_3clusters_2periods.txt', y)
print('WHyTe, nights, 2 periods')
print(np.sqrt(np.mean((y - target_n2) ** 2.0)))

5y = dirs0.predict(X_d0)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/days_WHyTeS_3clusters_0periods.txt', y)
print('WHyTe, days, 0 periods')
print(np.sqrt(np.mean((y - target_d0) ** 2.0)))

y = dirs2.predict(X_d2)
np.savetxt('/home/tom/pro/my/WHyTeS/data/my_output/days_WHyTeS_3clusters_2periods.txt', y)
print('WHyTe, days, 2 periods')
print(np.sqrt(np.mean((y - target_d2) ** 2.0)))
"""

#start = clock()
##structure = [2, [1.0, 1.0, 1.0, 1.0], [21600.0, 43200.0, 86400.0, 604800.0]]
#for j in xrange(1, 11):
#    structure = [2, [], []]
#    for i in xrange(7):
#        #start = clock()
#        dirs = directions.Directions(j, structure, '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
#        #finish = clock()
#        #print('\ntime to calculate model: ' + str(finish - start))
#        #print(i)
#        X, target = dirs.transform_data('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
#        y = dirs.predict(X)
#        #print('sum target: ' + str(np.sum(target)))
#        #print('sum y: ' + str(np.sum(y)))
#        #err = dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
#        #print('default rmse: ' + str(err))
#        pomer = np.sum(target)/np.sum(y)
#        #print('pomer: ' + str(pomer))
#        #print('calculated rmse: ' + str(np.sqrt(np.mean((y - target) ** 2.0))))
#        rmse_tst = np.sqrt(np.mean(((y * pomer) - target) ** 2.0))
#        #print('updated rmse test: ' + str(np.sqrt(np.mean(((y * pomer) - target) ** 2.0))))
#        #X, target = dirs.transform_data('/home/tom/pro/my/WHyTeS/data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt')
#        #y = dirs.predict(X)
#        #pomer = np.sum(target)/np.sum(y)
#        #rmse_trn = np.sqrt(np.mean(((y * pomer) - target) ** 2.0))
#        print(str(j) + ' ' + str(rmse_tst) + ' ' + str(clock() - start) + ' ' + str(structure[2])) 
#        #print('updated rmse training: ' + str(np.sqrt(np.mean(((y * pomer) - target) ** 2.0))))
#        frem = fremen.Fremen(structure)
#        P = frem.find_prominent('../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt', dirs)
#        #print(P)
#        structure[1].append(1.0)
#        structure[2].append(P)
#


#X, target = dirs.transform_data('../data/wednesday_thursday_days_with_angles_plus_reversed.txt')
#y = dirs.predict(X)
#print(np.sqrt(np.mean((y - target) ** 2.0)))
#err = dirs.rmse('../data/wednesday_thursday_days_with_angles_plus_reversed.txt')
#print(err)
#
#X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
#y = dirs.predict(X)
#print(np.sqrt(np.mean((y - target) ** 2.0)))
#err = dirs.rmse('../data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
#print(err)




#X, target = dirs.transform_data('../data/wednesday_thursday_nights_with_dirs.txt')
#X, target = dirs.transform_data(path)

#y = dirs.predict(X)
#print(y)

#err = dirs.rmse('../data/wednesday_thursday_nights.txt', plot_it=True)
#err = dirs.rmse('../data/wednesday_thursday_nights_with_dirs.txt', plot_it=True)

#print(err)
#print(np.sqrt(np.mean((y - target) ** 2.0)))
#print(np.sqrt(np.mean((np.zeros_like(target) - target) ** 2.0)))


#window = 3600
#for T in xrange(43200, 43201, window):
#    dirs.vykresli(T, window)

"""
edges_of_cell = np.array([300.0, 1.0, 1.0])

graph = draw_arrows.DrawArrows(probabilities.Directions(27, [4, [1.0, 1.0], [86400.0, 604800.0]], '../data/two_weeks_days_nights_weekends_with_dirs.txt'),
                               frequencies.Frequencies(9, edges_of_cell, [2, [1.0, 1.0], [86400.0, 604800.0]], '../data/two_weeks_days_nights_weekends_without_dirs.txt'),
                               edges_of_cell)

for counter, time in enumerate(xrange(0, 86401, int(edges_of_cell[0]))):
    start = clock()
    graph.make_png(time, counter)
    finish = clock()
    print('png created in [seconds]: ' + str(finish - start))
"""


edges_of_cell = np.array([60.0, 1.0, 1.0])
structure1 = [2, [1.0, 1.0, 1.0, 1.0], [21600.0, 43200.0, 86400.0, 604800.0]]
structure2 = [2, [1.0, 1.0], [86400.0, 604800.0]]
graph = draw_arrows.DrawArrows(directions.Directions(9, structure1, '../data/two_weeks_days_nights_weekends_with_angles_plus_reversed.txt'),
                               frequencies.Frequencies(9, edges_of_cell, structure2, '../data/two_weeks_days_nights_weekends_without_dirs.txt'),
                               edges_of_cell)

for counter, time in enumerate(xrange(32100, 68201, int(edges_of_cell[0]))):
    start = clock()
    graph.make_png(time, counter)
    finish = clock()
    print('png created in [seconds]: ' + str(finish - start))

