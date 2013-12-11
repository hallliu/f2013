cv_lasso = function(X, y, k) {
  fold_len = length(y) / k
  for (i in 0:(k-1)) {
    test_data = X[i*fold_len+1:(i+1)*fold_len+1,]
    test_resp = y[i*fold_len+1:(i+1)*fold_len+1]
    
    tr_data = X[-(i*fold_len+1:(i+1)*fold_len+1),]
    tr_resp = y[-(i*fold_len+1:(i+1)*fold_len+1)]
    
    lasso_model = lars(tr_data, tr_resp)
    preds = predict(lasso_model, test_data)
    print(dim(preds))
  }
}