findRSS = function(rho, model) {
  sig = getSigma(rho, model$dims$N)
  f = chol(sig)
  tr_resids = f %*% model$residuals
  return(t(tr_resids) %*% tr_resids)
}

getSigma = function(rho, n) {
  sig = diag(n)
  sig = rho^abs(row(sig) - col(sig))
  return(sig)
}

years = 1:100
model2 = gls(Nile ~ years, method="ML", correlation=corAR1(form=~years))
rho = coef(model2$modelStruct$corStruct, unconstrained=FALSE)
syy = findRSS(rho, gls(Nile ~ 1, method="ML", correlation=corAR1(value=rho, form=~years, fixed=TRUE)))
rss = findRSS(rho, model2)
ssreg = syy-rss
msreg = ssreg / (model2$dims$p - 1)
f = msreg / (rss / (100-model2$dims$p))
p = pf(f, 1, 100-model2$dims$p)