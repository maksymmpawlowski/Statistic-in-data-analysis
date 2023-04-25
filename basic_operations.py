import numpy as np
import math
import sklearn
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.stats import skew
import statistics as stts
import pandas as pd

xies = [1,2,3,4,5,6,7,8,9]
yies = [2,4,6,8,10,12,14,16,18]


#Arithmetic average:
def average(list):
    arithmetic_average = sum(list)/len(list)
    return arithmetic_average
#print(average(xies))
'''
print(statistics.mean(xies))
print(np.mean(xies))
series = pd.Series(xies)
print(series.mean())'''


#Modes
def modes(list, num):
    counts = {}
    for x in list:
        if x in counts:
            counts[x] += 1
        else: 
            counts[x] = 1 
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:num]
    for item in sorted_items:
        print(f'"{item[0]}" - {item[1]}')
#modes(xies, 3)
'''
print(statistics.mode(xies))
print(pd.Series(xies).mode())'''


#Median:
def median(list):
    list.sort()
    centre = len(list)/2
    if len(list) % 2 == 0:
        p1 = list[int(centre)]
        p2 = list[int(centre-1)]
        pp = (p1+p2)/2
        return pp
    else:
        return list[int(centre)]
#print(median(xies))
'''
print(statistics.median(xies))
print(np.median(xies))
print(pd.Series(xies).median())'''


#Variance
def variance(list):
    result = []
    for i in list:
        result.append((i - (average(list)))**2)
    return (sum(result))/(len(list)-1)
#print(variance(xies))
'''
print(np.var(xies))
print(statistics.variance(xies))
print(stats.tvar(xies))'''


#Standard deviation
def deviation(list):
    devi = np.sqrt(variance(list))
    return devi
#print(deviation(xies))
'''
print(np.std(xies))
print(pd.Series(xies).std())'''


#Mean absolute deviation (MAD)
def mad(list):
    devs = [abs(x - average(list)) for x in list]
    mad = sum(devs)/len(list)
    return mad
#print(mad(xies))
'''print(np.mean(np.abs(list - np.mean(list))))'''


#Pooled variance
#sum[(n_i - 1)*var_i] / sum[n]- 2
def pldvar(*lists):
    if len(lists)>2:
        return "Pooled variance is only for 2 variables"
    else:
        vars = []
        #left = sum[len(list)-1*variance(list) for list in lists]
        for list in lists:
            vars.append((len(list)-1) * variance(list))
        left = sum(vars)
        right = sum((len(list)for list in lists))-2
        return left/right
#print(pldvar(xies,yies))


#Skewness
def skewness(list):
    skew = 3*(average(list) - median(list))/deviation(list)
    return skew
#print(skewness(xies))
'''
print(skew(xies))
print(pd.Series(xies).skew())'''


#Kurtosis
def kurtosis(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    kurtosis = (m4 / variance**2) - 3
    return kurtosis
#print(kurtosis(xies))
'''
from scipy.stats import kurtosis
print(kurtosis(xies))
print(pd.Series(xies).kurtosis())
print(np.kurtosis(xies))'''


#Slope
def slope(x1,x2,y1,y2):
    slope = (y2-y1)/(x2-x1)
    return slope
#print(slope(xies[2],xies[3],yies[3],yies[4]))
'''
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(xies, yies)
print(slope)'''


#Correlation_Pearson
#(n∑XY - ∑X∑Y) / sqrt([n∑X^2 - (∑X)^2][n∑Y^2 - (∑Y)^2])
def pearson_r(list1, list2):
    n = len(list1)
    sum_x = sum(list1)
    sum_y = sum(list2)
    sum_x_sq = sum([x**2 for x in list1])
    sum_y_sq = sum([y**2 for y in list2])
    sum_xy = sum([list1[i]*list2[i] for i in range(n)])
    up = n*sum_xy - sum_x*sum_y
    down = math.sqrt((n*sum_x_sq - sum_x**2) * (n*sum_y_sq - sum_y**2))
    if down == 0:
        return 0
    else:
        return up/down
#print(pearson_r(xies,yies))
'''
print(np.corrcoef(xies, yies)[0,1])
print(pearsonr(xies, yies)[0])
print(pd.Series(xies).corr(pd.Series(yies)))
df = pd.DataFrame({'x': xies, 'y': yies})
corr_coef = df['x'].corr(df['y'])
print(corr_coef)
from scipy.stats import pearsonr
corr_coef, p_value = pearsonr(xies, yies)
print("Pearson r:", corr_coef)
print("p-value:", p_value)'''


#Correlation_Spearman
def spearman_r(list1, list2):
    sorted_lst1 = sorted(set(list1))
    ranks1 = [sorted_lst1.index(value) + 1 for value in list1]
    sorted_lst2 = sorted(set(list2))
    ranks2 = [sorted_lst2.index(value) + 1 for value in list2]
    
    n = len(ranks2)   
    up = 6 * sum((ranks1[i] - ranks2[i])**2 for i in range(n))
    down = n * (n**2 -1)
    rs = 1 - up/down
    return rs
#print(spearman_r(xies,yies))
'''
from scipy.stats import spearmanr
print(spearmanr(xies, yies)[0])
print(pd.Series(xies).corr(pd.Series(yies), method='spearman'))
from scipy.stats import spearmanr
print(spearmanr(xies, yies))
df = pd.DataFrame({'x': xies, 'y': yies})
rho = df.corr(method='spearman').iloc[0, 1]
print("Spearman correlation coefficient:", rho)
rho = np.corrcoef(xies, yies, rowvar=False)[0, 1]
print("Spearman correlation coefficient:", rho)
'''


#Covariance
def covariance(list1, list2):
    av1 = average(list1)
    av2 = average(list2)
    x = [(a-av1) * (b-av2) for a, b in zip(list1, list2)]
    return sum(x)/(len(list1)-1)
#print(covariance(xies,yies))
'''
print(np.cov(np.stack((xies, yies), axis=0))[0, 1])
df = pd.DataFrame({"x":xies, "y":yies})
print(df.cov().loc["x","y"])'''


#T-Value
#(mean1 - mean2) / sqrt( pldvar/n1 + pldvar/n2 )
def tvalue(list1,list2):
    left = average(list1)-average(list2)
    pooledvar = pldvar(list1,list2)
    right = np.sqrt(pooledvar/len(list1)+pooledvar/len(list2))
    return left/right
#print(tvalue(xies,yies))


#Regression
def reg(X,list1,list2):
    regression_coefficient = covariance(list1,list2)/variance(list1)
    Y = regression_coefficient * X
    return Y 
#print(reg(20,xies,yies))
'''
np.polyfit(x, y, deg=1) 
print(scipy.stats.linregress(xies,yies))
sklearn.linear_model.LinearRegression() 
statsmodels.api.OLS(y, X) 
'''