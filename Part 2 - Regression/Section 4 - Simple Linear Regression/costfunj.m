function j = costfun(X, y, theta)
m = size(X,1)
pred = X*theta
prederror = pred-y
j = 0.5/m * sum(prederror .^2);
