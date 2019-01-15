import math
from collections import Counter
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyximport
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from sklearn.preprocessing import minmax_scale

pyximport.install()
from frechet import frechet

df = pd.read_csv('WDI_csv/WDIData.csv', index_col='Country Name')
df = df.iloc[:, :-1]

indicator_name = ['International migrant stock (% of population)',
                  'International migrant stock, total',
                  'Net migration',
                  'Real effective exchange rate index (2010 = 100)',
                  'Official exchange rate (LCU per US$, period average)',
                  'DEC alternative conversion factor (LCU per US$)',
                  'GDP deflator: linked series (base year varies by country)',
                  'Inflation, GDP deflator: linked series (annual %)',
                  'Strength of legal rights index (0=weak to 12=strong)',
                  'Public credit registry coverage (% of adults)',
                  'Private credit bureau coverage (% of adults)',
                  'Depth of credit information index (0=low to 8=high)',
                  'Domestic credit to private sector (% of GDP)',
                  'Domestic credit provided by financial sector (% of GDP)',
                  'Claims on other sectors of the domestic economy (% of GDP)',
                  'Claims on central government, etc. (% GDP)',
                  'Risk premium on lending (lending rate minus treasury bill rate, %)',
                  'Real interest rate (%)',
                  'Interest rate spread (lending rate minus deposit rate, %)',
                  'Lending interest rate (%)',
                  'Deposit interest rate (%)',
                  'Wholesale price index (2010 = 100)',
                  'Inflation, consumer prices (annual %)',
                  'Consumer price index (2010 = 100)',
                  'Broad money growth (annual %)',
                  'Broad money to total reserves ratio',
                  'Broad money (% of GDP)',
                  'Broad money (current LCU)',
                  'Claims on private sector (annual growth as % of broad money)',
                  'Net foreign assets (current LCU)',
                  'Net domestic credit (current LCU)',
                  'Claims on other sectors of the domestic economy (annual growth as % of broad money)',
                  'Claims on central government (annual growth as % of broad money)',
                  'Total reserves minus gold (current US$)',
                  'Total reserves in months of imports',
                  'Total reserves (% of total external debt)',
                  'Total reserves (includes gold, current US$)',
                  'Bank liquid reserves to bank assets ratio (%)',
                  'Domestic credit to private sector by banks (% of GDP)',
                  'Depositors with commercial banks (per 1,000 adults)',
                  'Borrowers from commercial banks (per 1,000 adults)',
                  'Commercial bank branches (per 100,000 adults)',
                  'Bank capital to assets ratio (%)',
                  'Automated teller machines (ATMs) (per 100,000 adults)',
                  'Bank nonperforming loans to total gross loans (%)',
                  'Portfolio investment, bonds (PPG + PNG) (NFL, current US$)',
                  'Stocks traded, turnover ratio of domestic shares (%)',
                  'Stocks traded, total value (% of GDP)',
                  'Stocks traded, total value (current US$)',
                  'Listed domestic companies, total',
                  'Market capitalization of listed domestic companies (% of GDP)',
                  'Market capitalization of listed domestic companies (current US$)',
                  'S&P Global Equity Indices (annual % change)',
                  'Personal remittances, received (% of GDP)',
                  'Personal remittances, received (current US$)',
                  'Personal transfers, receipts (BoP, current US$)',
                  'Portfolio equity, net inflows (BoP, current US$)',
                  'Primary income on FDI, payments (current US$)',
                  'Foreign direct investment, net inflows (% of GDP)',
                  'Foreign direct investment, net inflows (BoP, current US$)',
                  'Portfolio investment, net (BoP, current US$)',
                  'Foreign direct investment, net (BoP, current US$)',
                  'Personal remittances, paid (current US$)',
                  'Foreign direct investment, net outflows (% of GDP)',
                  'Foreign direct investment, net outflows (BoP, current US$)']

indicator_name2 = ['International migrant stock, total',
                   'Merchandise exports (current US$)',
                   'Merchandise imports (current US$)',
                   'GDP (current US$)',
                   'GDP per capita (current US$)',
                   'GNI (current US$)',
                   'Exports of goods and services (% of GDP)',
                   'External balance on goods and services (% of GDP)',
                   'Imports of goods and services (% of GDP)',
                   'Trade (% of GDP)',
                   'Exports of goods and services (current US$)',
                   'Imports of goods and services (current US$)',
                   'Military expenditure (% of GDP)',
                   'Gross national expenditure (current US$)',
                   'Inflation, consumer prices (annual %)'
                   ]


# df.fillna(0, inplace=True)

def distance_matrix(X, f='correlation'):
    return pdist(X, metric=f)


def get_interpoled_data(datas):
    # TODO: make it work
    drop_list = []
    for i in range(datas.shape[0]):
        original_row = datas.iloc[i, :]
        curated_row = original_row.dropna()

        if len(curated_row) < 2:
            drop_list.append(datas.index[i])
            continue
            # raise ValueError("Need at least 2 not nane values!")

        indexes = original_row.index.values
        indexes = indexes.astype(np.float)

        x = curated_row.index.values
        x = x.astype(np.float)
        y = curated_row.values
        f = interp1d(x, y, fill_value='extrapolate')

        for n in range(len(indexes)):
            if math.isnan(datas.iat[i, n]):
                calcul = f(indexes[n])
                datas.iat[i, n] = calcul if calcul > 0 else 0
    datas = datas.drop(drop_list)
    return datas


def get_data_by_name(name, name_list):
    data = df[df['Indicator Name'] == name]
    data = data.iloc[:, 3:]
    data = get_interpoled_data(data)
    return data, data.index


def pretty_plot(ds, datas, tmp):
    # fig, ax = plt.subplots(figsize=(13, 15))
    t0 = time()
    mds = manifold.TSNE(2, random_state=0, metric='precomputed')
    Y = mds.fit_transform(ds)
    t1 = time()
    color = [tmp[i] for i in range(len(Y))]
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("TSNE (%.2g sec)" % (t1 - t0))

    for i, txt in enumerate(datas):
        plt.annotate(txt, (Y[i, 0], Y[i, 1]))
    plt.show()


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
    if not ('population' in T[a][0].lower() or 'survival' in T[a][0].lower()):
        print(T[a][0])
        # test_pool.append(T[a][0])

in_list = []
name_list = set(df.index)
for name in indicator_name2:
    try:
        frame, names = get_data_by_name(name, name_list)
        in_list.append(frame)
        # print(len(names))
        name_list &= set(names)
    except ValueError:
        print(name)
        continue
print(len(name_list))
out_list = []
name_list = sorted(list(name_list))
for pool in in_list:
    out_list.append(minmax_scale(pool.loc[name_list].values, axis=0))
result = None
np.random.seed(3345)
scale = np.random.rand(len(out_list))
for i, X in enumerate(out_list):
    if i == 0:
        result = scale[i] * distance_matrix(X, f=frechet)
    else:
        result += scale[i] * distance_matrix(X, f=frechet)
dist = ward(result)
dendrogram(dist, labels=name_list, leaf_font_size=10)

plt.show()
tmp = fcluster(dist, t=0.4)
# res = DBSCAN(metric='precomputed', eps=0.2, min_samples=3).fit(squareform(result))
# uniq = len(set(res.labels_))
# print(uniq)
print(name_list)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(tmp))]
pretty_plot(squareform(result), name_list, [colors[i] for i in tmp])
