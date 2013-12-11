getSigma = function(rho, n) {
  sig = diag(n)
  sig = rho^abs(row(sig) - col(sig))
  return(sig)
}

findRSS = function(rho, model) {
  sig = getSigma(rho, model$dims$N)
  f = chol(sig)
  tr_resids = f %*% model$residuals
  return(t(tr_resids) %*% tr_resids)
}

model = gls(Nile ~ years + I(years^2), method="ML", correlation=corAR1(form=~years))
rho = coef(model$modelStruct$corStruct, unconstrained=FALSE)
x0 = c(1, 110, 110^2)
years = 1:100
x = matrix(0, 100,3)
x = years^(col(x)-1)
ss = getSigma(rho, 100)
s = chol(ss)
shat = sqrt(findRSS(rho, model) / 97)
sepred = shat * sqrt(1+t(x0)%*%solve(t(x)%*%t(s)%*%solve(s)%*%x)%*%x0)
#years = 1:100
#ext_years = 1:110
#model = gls(Nile ~ years + I(years^2), method="ML", correlation=corAR1(form=~years))
#rho = coef(model$modelStruct$corStruct, unconstrained=FALSE)
#sig = getSigma(rho, 110)
#s11 = sig[1:100,1:100]
#s12 = sig[1:100,101:110]
#s21 = sig[101:110,1:100]
#s22 = sig[101:110,101:110]
#mean = -s21%*%solve(s11)%*%model$residuals
#var = s22 - s21%*%solve(s11)%*%s12






