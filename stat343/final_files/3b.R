pca_function = function(response, predictors, testSize, nComp) {
  totalLen = length(response)
  training_resp = response[1:testSize-1]
  test_resp = response[testSize:totalLen]
  
  training_pred = predictors[1:testSize-1,]
  test_pred = predictors[testSize:totalLen,]
  
  pcaM = prcomp(training_pred)
  model = lm(training_resp ~ pcaM$x[,1:nComp])
  print(summary(model))
  means = apply(training_pred, 2, mean)
  test_minus_means = as.matrix(sweep(test_pred, 2, means))
  tr_test_preds = test_minus_means %*% pcaM$rot[,1:nComp]
  preds = cbind(1, tr_test_preds) %*% model$coef
  print(rmse(training_resp, model$fit))
  return(rmse(preds, test_resp))
}

