from collections import Counter

import pandas as pd
import pyximport

pyximport.install()

df = pd.read_csv('WDI_csv/WDIData.csv', index_col='Country Name')
df = df.iloc[:, :-1]

Adolescent fertility rate (births per 1,000 women ages 15-19)
Net official development assistance and official aid received (current US$)
Mortality rate, infant (per 1,000 live births)
Mortality rate, under-5 (per 1,000 live births)
Rural population (% of total population)


International migrant stock, total
Merchandise exports (current US$)
Merchandise imports (current US$)
GDP (current US$)
GDP per capita (current US$)
GNI (current US$)
Exports of goods and services (% of GDP)
External balance on goods and services (% of GDP)
Imports of goods and services (% of GDP)
Trade (% of GDP
Exports of goods and services (current US$)
Imports of goods and services (current US$)
Inflation, consumer prices (annual %)

sums = 0.
for _, b in social:
    sums += b
print(sums)
for a, b in social:
    print((a, b / sums))
# df.fillna(0, inplace=True)

# def distance_matrix(X, f='correlation'):
#     return pdist(X, metric=f)
#
#
# def get_interpoled_data(datas):
#     # TODO: make it work
#     drop_list = []
#     for i in range(datas.shape[0]):
#         original_row = datas.iloc[i, :]
#         curated_row = original_row.dropna()
#
#         if len(curated_row) < 2:
#             drop_list.append(datas.index[i])
#             continue
#             # raise ValueError("Need at least 2 not nane values!")
#
#         indexes = original_row.index.values
#         indexes = indexes.astype(np.float)
#
#         x = curated_row.index.values
#         x = x.astype(np.float)
#         y = curated_row.values
#         f = interp1d(x, y, fill_value='extrapolate')
#
#         for n in range(len(indexes)):
#             if math.isnan(datas.iat[i, n]):
#                 calcul = f(indexes[n])
#                 datas.iat[i, n] = calcul if calcul > 0 else 0
#     datas = datas.drop(drop_list)
#     return datas
#
#
# def get_data_by_name(name, name_list):
#     data = df[df['Indicator Name'] == name]
#     data = data.iloc[:, 3:]
#     data = get_interpoled_data(data)
#     return data, data.index
#
#
# def pretty_plot(ds, datas, tmp):
#     # fig, ax = plt.subplots(figsize=(13, 15))
#     t0 = time()
#     mds = manifold.TSNE(2, random_state=0, metric='precomputed')
#     Y = mds.fit_transform(ds)
#     t1 = time()
#     color = [tmp[i] for i in range(len(Y))]
#     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
#     plt.title("TSNE (%.2g sec)" % (t1 - t0))
#
#     for i, txt in enumerate(datas):
#         plt.annotate(txt, (Y[i, 0], Y[i, 1]))
#     plt.show()
#
#
def get_most_common_n(n):
    a = df[df['1960'].notnull()]['Indicator Name']
    test = Counter()
    for b, c in zip(a, a.index):
        test[b] += 1
    return test.most_common(n)


set(df.loc['Afghanistan']['Indicator Name'])

T = get_most_common_n(500)
test_pool = []
for a in range(0, 200, 1):
    # if not ('population' in T[a][0].lower() or 'survival' in T[a][0].lower()):
    print(T[a][0])
        # test_pool.append(T[a][0])
#
# in_list = []
# name_list = set(df.index)
# for name, scale in indicator_name:
#     try:
#         frame, names = get_data_by_name(name, name_list)
#         in_list.append((frame, scale))
#         # print(len(names))
#         name_list &= set(names)
#     except ValueError:
#         print(name)
#         continue
# print(len(name_list))
# out_list = []
# name_list = sorted(list(name_list))
# for pool, scale in in_list:
#     out_list.append((minmax_scale(pool.loc[name_list].values, axis=0), scale))
# result = None
# np.random.seed(3345)
# scale = np.random.rand(len(out_list))
# for i, X in out_list:
#     if i == 0:
#         result = i * distance_matrix(X, f=frechet)
#     else:
#         result += i * distance_matrix(X, f=frechet)
# dist = ward(result)
# dendrogram(dist, labels=name_list, leaf_font_size=10)
#
# plt.show()
# tmp = fcluster(dist, t=0.4)
# print(name_list)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(tmp))]
# pretty_plot(squareform(result), name_list, [colors[i] for i in tmp])
