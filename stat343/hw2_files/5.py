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

import ipdb;ipdb.set_trace()
tdist = scipy.stats.t(6)
t_crit_val = tdist.ppf(0.975)
print 'CI:({0}, {1})'.format(beta[0] - beta1_se*t_crit_val, beta[0] + beta1_se*t_crit_val)

R = np.matrix("1 1")
xtxinv = linalg.inv(X.T * X)
s_1= 1 / ((R * xtxinv * R.T)[0,0])

rhb = beta.sum()
sigmahat = np.sqrt(sigmasq)
f_dist = scipy.stats.f(1,6)
f_crit = f_dist.ppf(0.95)
half_width = sigmahat * np.sqrt(f_crit / s_1)
print "CI for b_1+b_2: ({0}, {1})".format(rhb - half_width, rhb + half_width)

tval = (beta[1,0] - 3)/np.sqrt(beta_cov[1,1])
p_val = 1 - tdist.cdf(tval)
print tval, p_val

R1 = np.matrix("1 -1")
s_2 = 1 / ((R1 * xtxinv * R1.T)[0,0])
rhb1 = (R1 * beta)[0,0]
f_stat = rhb1 * rhb1 * s_2 / sigmasq
print f_stat, 1 - f_dist.cdf(f_stat)
