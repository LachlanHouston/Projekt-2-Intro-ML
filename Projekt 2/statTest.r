wr <- read.table("nyeerrors.csv", header=TRUE, sep=",", as.is=TRUE)

base <- wr[ ,1 ]
linreg <- wr[ ,2 ]
ANN <- wr[ ,3 ]

(n <- length(base))
(nu <- n-1)
(alpha <- 0.05)

# Base vs Linreg
z <- base - linreg
(zMean <- 1/n * sum(z))
sigma <- sd(z)
tobs <- (zMean - 0) / (sigma / sqrt(n))
(pvalue <- 2 * (1-pt(abs(tobs), df=n-1))) # 2.2e-16
(CI <- zMean + c(-1,1)*qt(0.975, n-1)*sigma/sqrt(n))

# Base vs ANN
z <- base - ANN
(zMean <- 1/n * sum(z))
sigma <- sd(z)
tobs <- (zMean - 0) / (sigma / sqrt(n))
(pvalue <- 2 * (1-pt(abs(tobs), df=n-1)))
(CI <- zMean + c(-1,1)*qt(0.975, n-1)*sigma/sqrt(n))

# ANN vs Linreg
z <- linreg - ANN
(zMean <- 1/n * sum(z))
sigma <- sd(z)
tobs <- (zMean - 0) / (sigma / sqrt(n))
(pvalue <- 2 * (1-pt(abs(tobs), df=n-1)))
(CI <- zMean + c(-1,1)*qt(0.975, n-1)*sigma/sqrt(n))

# Verifying that R's t-tests give the same answers
t.test(base - linreg)
t.test(base - ANN)
t.test(linreg-ANN)


# Average losses
1/n * sum(base)
1/n * sum(linreg)
1/n * sum(ANN)

