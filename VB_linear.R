logdet <- function(m) {
  return(2*sum(log(diag(chol(m)))))
}

#### he function expects the arguments
# - X: N x D matrix of training input samples, one per row
# - y: N-element column vector of corresponding output samples
# - a0, b0 (optional): scalar prior parameters of the noise precision
# - c0, d0 (optional): scalar hyper-prior shrinkage parameters
# If not given, the prior/hyper-prior parameters default to a0 = 1e-2,
# b0 = 1e-4, c0 = 1e-2, and d0 = 1e-4, resulting in an uninformative prior.
#
#### It returns
# - w: posterior weight D-element mean vector
# - V: posterior weight D x D covariance matrix
# - invV, logdetV: inverse of V, and its log-determinant
# - an, bn: scalar posterior parameter of noise precision
# - E_a: mean vector E(alpha) of shrinkage hyper-posterior
# - L: variational bound, lower-bounding the log-model evidence p(y | X)
#

vb_linear_arc <- function(X,y,a0=1e-2,b0=1e-4,c0=1e-2,
                          d0=1e-4,convergenceTol=0.00001){
  ## pre-process data
  n <- nrow(X)
  D <- ncol(X)
  X_corr <- t(X)%*%X
  Xy_corr <- t(X)%*%y
  an <- a0 + n/2
  gammaln_an <- lgamma(an)
  cn <- c0 + 1/2
  D_gammaln_cn <- D*lgamma(cn)
  
  ## iterate to find hyperparameters
  L_last <- -.Machine$double.xmax
  max_iter <- 500
  E_a <- c0/d0*rep(1,D)
  for (iter in 1:max_iter) {
    # covariance and weight of linear model
    inV <- diag(E_a) + X_corr
    V <- solve(inV)
    logdetV <- -logdet(inV)
    w <- V%*%Xy_corr
    
    # parameters of noise model (an remains constant)
    sse <- sum((y-X%*%w)^2)
    bn <- b0 + 0.5*(sse + sum((as.vector(w))^2*E_a))
    E_t <- an/bn
    
    # parameters of covariance prior (cn remains constant)
    dn <- d0 + 0.5*(E_t*((as.vector(w))^2) + diag(V))
    E_a <- cn/dn
    
    # variational bound, ignoring constant terms for now
    L <- -0.5*(E_t*sse + sum(X*(X%*%V))) + 0.5*logdetV -
      b0*E_t + gammaln_an - an*log(bn) + an + 
      D_gammaln_cn - cn*sum(log(dn))
    
    # variational bound must grow!
    if(L_last>L) {
      warning(sprintf("Last bound %f, current bound %f", 
                      L_last, L))
    }
    if(abs(L_last-L)<abs(convergenceTol*L))
       break
    
    L_last <- L
    show(sprintf("Iteration %d; Lower bound=%f", iter, L))
  }
  
  ## augment variational bound with constant terms
  L <- L - 0.5*(n*log(2*pi)-D) - lgamma(a0) + a0*log(b0) +
    D*(-lgamma(c0) + c0*log(d0))
  
  return(list(w=w,V=V,inV=inV,logdetV=logdetV,
              an=an,bn=bn,E_a=E_a,L=L))
}


vb_linear_pred <- function(X, w, V, an, bn){
  ans = matrix(0, nrow=dim(X)[1],ncol=2)
  for (i in 1:dim(X)[1]) {
    x <- as.matrix(X[i,])
    mu <- t(w)%*%x
    lamda <- (an/bn)/(1+t(x)%*%V%*%x)
    ans[i,] <-c(mu,lamda)
  }
  nu <- 2*an
  return(list(fit=ans,nu=nu))
}
