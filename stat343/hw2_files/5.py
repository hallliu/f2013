import numpy as np
import numpy.linalg as linalg
import scipy.stats

X = np.matrix('10 9 9 11 11 10 10 12; 15 14 13 15 14 14 16 13').T
y = np.matrix('82 79 74 83 80 81 84 81').T

(q,r) = linalg.qr(X)

beta = linalg.solve(r, q.T * y)

print beta

predicted = X * beta
residuals = y - predicted
rss = np.dot(residuals.T, residuals)
df = y.shape[0] - X.shape[1]

sigmasq = (rss / df)[0,0]
print sigmasq

beta_cov = linalg.inv(X.T * X) * sigmasq
beta1_se = np.sqrt(beta_cov[0,0])

tdist = scipy.stats.t(6)
t_crit_val = tdist.ppf(0.975)
import ipdb;ipdb.set_trace()
print 'CI:({0}, {1})'.format(beta[0] - beta1_se*t_crit_val, beta[0] + beta1_se*t_crit_val)
