## ----eval=FALSE---------------------------------------------------------------
#  answer_3_4 <- function(n, sigma){
#  
#    # 定义使用逆变换法生成Rayleigh(σ)分布随机样本的函数
#    simulate_rayleigh <- function(n, sigma) {
#      u <- runif(n)
#      return(sigma * sqrt(-2 * log(u)))
#    }
#    rayleigh_samples <- simulate_rayleigh(n, sigma)
#  
#    # 绘制Rayleigh(σ)分布随机样本的直方图
#    hist(rayleigh_samples, prob = TRUE, breaks = 30, col = "lightblue",
#       main = paste("Rayleigh(σ =", sigma, ") Distribution"))
#  
#    # 将Rayleigh(σ)理论密度函数叠加在直方图上
#    x_vals <- seq(0, max(simulate_rayleigh(n, sigma)), length.out = 100)
#    rayleigh_pdf <- function(x, sigma) {
#      (x / sigma^2) * exp(-x^2 / (2 * sigma^2))
#    }
#    lines(x_vals, rayleigh_pdf(x_vals, sigma), lwd = 2)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_3_11 <- function(n, p1, mean1, sd1, mean2, sd2){
#  
#    # 根据p1, 1-p1的概率混合两个正态分布
#    mixing_prob <- p1
#    choices <- runif(n)
#    mixing_samples <- ifelse(choices < mixing_prob, rnorm(n, mean1, sd1),
#        rnorm(n, mean2, sd2))
#  
#    # 绘制混合分布随机样本的直方图
#    hist(mixing_samples, prob = TRUE, breaks = 40, col = "lightblue",
#        main = paste("Mixing Probability (p1 =", p1, ")"))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_3_20 <- function(t, lambda, alpha, beta){
#  
#    # 定义模拟复合泊松-伽马过程的函数
#    simulate_compound_poisson_gamma <- function(t, lambda, alpha, beta) {
#      N_t <- rpois(1, lambda * t)
#      if (N_t == 0) {
#        return(0) # 返回X(t) = 0
#      } else {
#        Y_i <- rgamma(N_t, alpha, beta)
#        return(sum(Y_i)) # 返回复合泊松-伽马过程过程X(t)
#      }
#    }
#  
#    # 模拟1000次得到复合泊松-伽马过程{X(t), t ≥ 0}, 计算结果
#    X_t <- replicate(1000, simulate_compound_poisson_gamma(t, lambda, alpha, beta))
#    mean_estimated <- mean(X_t)
#    var_estimated <- var(X_t)
#    mean_theoretical <- lambda * t * alpha / beta
#    var_theoretical <- lambda * t * alpha * (alpha + 1) / (beta^2)
#  
#    # 输出均值和方差的估计值和理论值
#    cat("Estimated Mean of X(10):", mean_estimated,
#        "Estimated Variance of X(10):", var_estimated, "\n")
#    cat("Theoretical Mean of X(10):", mean_theoretical,
#        "Theoretical Variance of X(10):", var_theoretical, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_5_4 <- function(n, alpha, beta){
#    x_vals <- seq(0.1, 0.9, by=0.1)
#    # 计算蒙特卡洛估计、pbeta的F(x)值
#    monte_carlo_cdf <- sapply(x_vals, function(x){
#      uniform_samples <- runif(n, 0, x)
#      mean(dbeta(uniform_samples, alpha, beta)) * x
#    })
#    pbeta_cdf <- pbeta(x_vals, alpha, beta)
#    # 输出结果
#    result <- data.frame(
#      x = x_vals,
#      MonteCarloCDF = monte_carlo_cdf,
#      PbetaCDF = pbeta_cdf,
#      Difference = abs(monte_carlo_cdf - pbeta_cdf))
#    print(result)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_5_9 <- function(n, sigma){
#    # 使用逆变换法生成对偶随机数、两个独立的随机数
#    u <- runif(n)
#    X <- sigma * sqrt(-2 * log(u))
#    X_duiou <- sigma * sqrt(-2 * log(1 - u))
#    X1 <- sigma * sqrt(-2 * log(runif(n)))
#    X2 <- sigma * sqrt(-2 * log(runif(n)))
#    # 输出结果比较(X+X′)/2、(X1+X2)/2的方差
#    var_anti <- var((X + X_duiou) / 2)
#    var_ind <- var((X1 + X2) / 2)
#    cat("Variance of (X + X') / 2 : ", var_anti,
#        "Variance of (X1 + X2) / 2 : ", var_ind,
#        "percent reduction: ", abs(var_anti - var_ind) / var_ind, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_5_13 <- function(n){
#    # 定义g(x)函数
#    g <- function(x){
#      x^2 / sqrt(2 * pi) * exp(-x^2 / 2)
#    }
#    # f1, f2分别选择Exp(1), Ga(3, 1)，且均向右移动一个单位长度
#    f1_samples <- rexp(n, 1) + 1
#    f1_estimate <- mean(g(f1_samples) / dexp(f1_samples - 1, 1))
#    f1_var <- var(g(f1_samples) / dexp(f1_samples - 1, 1)) / n
#    f2_samples <- rgamma(n, 3, 1) + 1
#    f2_estimate <- mean(g(f2_samples) / dgamma(f2_samples - 1, 3, 1))
#    f2_var <- var(g(f2_samples) / dgamma(f2_samples - 1, 3, 1)) / n
#    # 输出结果比较使用Exp(1), Ga(3, 1)的方差
#    cat("Estimate using Exp(1):", f1_estimate,
#        "Variance using Exp(1):", f1_var, "\n")
#    cat("Estimate using Ga(3, 1):", f2_estimate,
#        "Variance using Ga(3, 1):", f2_var, "\n")
#    # 作出g(x)、f1、f2的函数图像
#    x_vals <- seq(1, 10, length.out = 1000)
#    plot(x_vals, g(x_vals), type = "l", col = "black", lwd = 2,
#         ylim = c(0, max(g(x_vals), dexp(x_vals-1,1), dgamma(x_vals-1,3,1))),
#         xlab = "x", ylab = "Density")
#    lines(x_vals, dexp(x_vals - 1, 1), col = "blue", lwd = 2)
#    lines(x_vals, dgamma(x_vals - 1, 3, 1), col = "pink", lwd = 2)
#    legend("topright", col = c("black", "blue", "pink"), lwd = 2,
#           legend = c("g(x)", "f1", "f2"))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  fast_sorting_experiment <- function(n_values){
#    fast_sort <- function(arr){
#      # 递归终止条件：如果数组长度不大于1则返回该数组
#      if (length(arr) <= 1) {
#        return(arr)
#      }else{
#        # 随机选择一个基准数，以此划分数组
#        pivot <- arr[sample(1:length(arr), 1)]
#        equal <- arr[arr == pivot]
#        left_sorted <- fast_sort(arr[arr < pivot])
#        right_sorted <- fast_sort(arr[arr > pivot])
#        # 返回递归结果
#        return(c(left_sorted, equal,right_sorted))
#        }
#    }
#    # a_n用来储存平均计算时间，嵌套循环得到不同n的平均时间
#    a_n <- numeric(length(n_values))
#    for (i in seq_along(n_values)){
#      times <- numeric(100)
#      for (j in 1:100){
#        times[j] <- system.time(fast_sort(sample(1:n_values[i])))[3]
#      }
#      a_n[i] <- mean(times)
#    }
#    # 回归分析画图
#    t_n <- n_values * log(n_values)
#    model <- lm(a_n ~ t_n)
#    plot(t_n, a_n, xlab = "tn=nlog(n)", ylab = "an")
#    abline(model, col = "black", lwd = 2)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_6_6 <- function(n, num_sim) {
#    # 生成数据返回偏度
#    data_generation <- function(n, num_sim) {
#      skewness <- numeric(num_sim)
#      for (i in 1:num_sim) {
#        sample_data <- rnorm(n)
#        m3 <- mean((sample_data - mean(sample_data))^3)
#        m2 <- mean((sample_data - mean(sample_data))^2)
#        skewness[i] <- m3 / m2^(3/2)
#      }
#      return(skewness)
#    }
#    # 计算偏度分位数的蒙特卡洛估计量
#    stat_inference <- function(skewness) {
#      quantiles_mc <- quantile(skewness, probs = c(0.025, 0.05, 0.95, 0.975))
#      se <- sd(skewness) / sqrt(length(skewness))
#      return(list(quantiles_mc = quantiles_mc, se = se))
#    }
#    # 计算偏度在大样本下近似分布的分位数，进行结果比较
#    result_report <- function(quantiles_mc, n) {
#      quantiles_ls <- qnorm(c(0.025, 0.05, 0.95, 0.975),
#                                mean = 0, sd = sqrt(6/n))
#      result <- data.frame(
#        "Quantiles" = c(0.025, 0.05, 0.95, 0.975),
#        "Monte Carlo Estimate" = quantiles_mc,
#        "Large Sample Approx" = quantiles_ls)
#      print(result)
#    }
#    # 调用函数输出结果
#    skewness <- data_generation(n, num_sim)
#    quantiles_mc <- stat_inference(skewness)$quantiles_mc
#    result_report(quantiles_mc, n)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_6_b <- function(n, num_sim, rho, alpha) {
#    # (X, Y)分别是多元正态分布、多项式依赖关系
#    bivariate_generation <- function(n, rho, dist_type) {
#      if (dist_type == "ind") {
#        Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
#        data <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
#      } else if (dist_type == "d") {
#        X <- rnorm(n)
#        Y <- X^2 + X + rnorm(n, sd = 2)
#        data <- cbind(X, Y)
#      }
#      return(data)
#    }
#    # 使用cor.test()计算Pearson, Spearman, Kendall检验的p-value
#    cor_test <- function(data) {
#      pearson_test <- cor.test(data[, 1], data[, 2], method = "pearson")
#      spearman_test <- cor.test(data[, 1], data[, 2], method = "spearman")
#      kendall_test <- cor.test(data[, 1], data[, 2], method = "kendall")
#      return(list(
#        pearson_pval = pearson_test$p.value,
#        spearman_pval = spearman_test$p.value,
#        kendall_pval = kendall_test$p.value))
#    }
#    # 通过num_sim次实验得到的p-value计算不同检验的power
#    power_empir <- function(n, num_sim, rho, alpha, dist_type) {
#      data_xy <- replicate(num_sim, bivariate_generation(n, rho, dist_type), simplify = FALSE)
#      pearson_result <- sapply(data_xy, function(x) cor_test(x)$pearson_pval <= alpha)
#      spearman_result <- sapply(data_xy, function(x) cor_test(x)$spearman_pval <= alpha)
#      kendall_result <- sapply(data_xy, function(x) cor_test(x)$kendall_pval <= alpha)
#      return(c(
#          pearson_power = mean(pearson_result),
#          spearman_power = mean(spearman_result),
#          kendall_power = mean(kendall_result)))
#    }
#    # 调用函数输出结果
#    ind_power <- power_empir(n, num_sim, rho, alpha, dist_type = "ind")
#    d_power <- power_empir(n, num_sim, rho, alpha, dist_type = "d")
#    cat("normal power : ", ind_power, "\n")
#    cat("alternative power : ", d_power, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  hypothesis_test <- function(n, p1, p2) {
#    method1 <- rbinom(n, size = 1, p1)
#    method2 <- rbinom(n, size = 1, p2)
#    paired_t_result <- t.test(method1, method2, paired = TRUE)
#    print(paired_t_result)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  multiple_test_problem <- function(N, n_null, alpha, m) {
#    results_matrix <- matrix(0, nrow = m, ncol = 6)
#    for (i in 1:m) {
#      pvals_null <- runif(n_null)
#      pvals_alt <- rbeta(N - n_null, 0.1, 1)
#      pvals <- c(pvals_null, pvals_alt)
#      true_status <- c(rep(0, n_null), rep(1, N - n_null))
#      # 计算Bonferroni correction的FWER, FDR, TPR
#      pval_bonferroni <- p.adjust(pvals, method = "bonferroni")
#      reject_bonferroni <- pval_bonferroni < alpha
#      FWER_bonferroni <- sum(reject_bonferroni & true_status == 0) > 0
#      FDR_bonferroni <- sum(reject_bonferroni & true_status == 0) / max(1, sum(reject_bonferroni))
#      TPR_bonferroni <- sum(reject_bonferroni & true_status == 1) / sum(true_status == 1)
#      # 计算B-H correction的FWER, FDR, TPR
#      pval_bh <- p.adjust(pvals, method = "BH")
#      reject_bh <- pval_bh < alpha
#      FWER_bh <- sum(reject_bh & true_status == 0) > 0
#      FDR_bh <- sum(reject_bh & true_status == 0) / max(1, sum(reject_bh))
#      TPR_bh <- sum(reject_bh & true_status == 1) / sum(true_status == 1)
#      results_matrix[i, ] <- c(FWER_bonferroni, FDR_bonferroni, TPR_bonferroni,
#                               FWER_bh, FDR_bh, TPR_bh)
#    }
#    # 将每轮结果处理成3 × 2 table
#    avg_results <- apply(results_matrix, 2, mean)
#    output_table <- matrix(avg_results, nrow = 3, byrow = TRUE,
#                           dimnames = list(c("FWER", "FDR", "TPR"),
#                                           c("Bonferroni correction", "B-H correction")))
#    print(output_table)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_7_4 <- function(R) {
#    data <- c(3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487)
#    lambda_mle <- 1 / mean(data)
#    # 定义计算MLE的函数
#    mle_lambda <- function(data, indices) {
#      sample_data <- data[indices]
#      return(1 / mean(sample_data))
#    }
#    # 进行bootstrap得到bias and standard error
#    boot_lambda <- boot::boot(data, mle_lambda, R)
#    boot_bias <- mean(boot_lambda$t) - lambda_mle
#    boot_se <- sd(boot_lambda$t)
#    # 打印结果
#    cat("MLE of the hazard rate λ :", lambda_mle, "\n",
#        "bootstrap bias :", boot_bias, "\n",
#        "bootstrap standard error :", boot_se, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_7_5 <- function(R) {
#    data <- c(3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487)
#    # 定义计算mean time的函数
#    mean_time <- function(data, indices) {
#      sample_data <- data[indices]
#      return(mean(sample_data))
#    }
#    # 进行bootstrap得到standard normal, basic, percentile, and BCa methods的CI
#    boot_time <- boot::boot(data, mean_time, R)
#    ci_normal <- boot::boot.ci(boot_time, type = "norm")
#    ci_basic <- boot::boot.ci(boot_time, type = "basic")
#    ci_percentile <- boot::boot.ci(boot_time, type = "perc")
#    ci_bca <- boot::boot.ci(boot_time, type = "bca")
#    # 打印结果
#    cat("Standard normal method: ", ci_normal$normal[2:3], "\n",
#        "Basic method: ", ci_basic$basic[4:5], "\n",
#        "Percentile method: ", ci_percentile$perc[4:5], "\n",
#        "BCa method: ", ci_bca$bca[4:5], "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_7_8 <- function(k) {
#    scor <- bootstrap::scor
#    # 定义计算参数θ的函数
#    theta <- function(x, i) {
#      lambda <- eigen(cov(x[i,]))$values
#      return (lambda[1]/sum(lambda))
#    }
#    n <- nrow(scor)
#    theta_hat <- theta(scor, 1:n)
#    # 通过刀切法得到估计的偏差和标准差
#    theta_jack <- numeric(n)
#    for (i in 1:n) {
#      theta_jack[i] <- theta(scor, (1:n)[-i]) }
#    bias_jack <- (n-1)*(mean(theta_jack)-theta_hat)
#    se_jack <- sqrt((n-1)*mean((theta_jack-theta_hat)^2))
#    # 输出结果
#    cat("bias:", round(bias_jack, k), "\n",
#        "standard error:", round(se_jack, k), "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_7_10 <- function(k) {
#    # 加载数据并定义四种模型的函数
#    ironslag <- DAAG::ironslag
#    X <- ironslag$chemical
#    Y <- ironslag$magnetic
#    linear <- function(X, Y) { lm(Y ~ X) }
#    quadratic <- function(X, Y) { lm(Y ~ poly(X, 2)) }
#    exponential <- function(X, Y) { lm(log(Y) ~ X) }
#    cubic <- function(X, Y) { lm(Y ~ poly(X, 3)) }
#    n <- length(Y)
#    # 定义计算loocv error的函数
#    loocv_error <- function(model) {
#      error <- numeric(n)
#      for (i in 1:n) {
#        fit <- model(X[-i], Y[-i])
#        prediction <- predict(fit, newdata = data.frame(X = X[i]))
#        error[i] <- (Y[i] - prediction)^2 }
#      return (mean(error))
#    }
#    # 定义计算adjusted R2的函数
#    adjusted_R2 <- function(model) {
#      return(summary(model)$adj.r.squared)
#    }
#    # 输出结果
#    cat("linear loocv error:", round(loocv_error(linear), k), "\n",
#        "quadratic loocv error:", round(loocv_error(quadratic), k), "\n",
#        "exponential loocv error:", round(loocv_error(exponential), k), "\n",
#        "cubic loocv error:", round(loocv_error(cubic), k), "\n")
#    cat("linear adjusted R2:", round(adjusted_R2(lm(Y ~ X)), k), "\n",
#        "quadratic adjusted R2:", round(adjusted_R2(lm(Y ~ poly(X, 2))), k), "\n",
#        "exponential adjusted R2:", round(adjusted_R2(lm(log(Y) ~ X)), k), "\n",
#        "cubic adjusted R2:", round(adjusted_R2(lm(Y ~ poly(X, 3))), k), "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_8_1 <- function(num_perm) {
#    X <- c(158, 171, 193, 199, 230, 243, 248, 248, 250, 267, 271, 316, 327, 329)
#    Y <- c(141, 148, 169, 181, 203, 213, 229, 244, 257, 260, 271, 309)
#    cram_test <- cramer::cramer.test(X, Y)$statistic
#    # 定义permutation test的函数
#    perm_test <- function(x, y, num_perm) {
#      perm <- numeric(num_perm)
#      for (i in 1:num_perm) {
#        j <- sample(1:length(c(x, y)), size = length(x), replace = FALSE)
#        perm_x <- c(x, y)[j]
#        perm_y <- c(x, y)[-j]
#        perm[i] <- cramer::cramer.test(perm_x, perm_y)$statistic }
#      return (mean(c(cram_test, perm) >= cram_test))
#    }
#    # 输出结果
#    p_value <- perm_test(X, Y, num_perm)
#    cat("Cramer-von Mises test p-value:", cramer::cramer.test(X, Y)$p.value, "\n",
#        "permutation test p-value:", p_value, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_8_2 <- function(num_perm) {
#    # (X, Y)是独立的标准正态分布
#    X <- rnorm(10)
#    Y <- rnorm(10)
#    cor_init <- cor(X, Y, method = "spearman")
#    # 定义permutation test的函数
#    perm_test <- function(x, y, num_perm) {
#      perm <- numeric(num_perm)
#      for (i in 1:num_perm) {
#        j <- sample(1:length(c(x, y)), size = length(x), replace = FALSE)
#        perm_x <- c(x, y)[j]
#        perm_y <- c(x, y)[-j]
#        perm[i] <- cor(perm_x, perm_y, method = "spearman") }
#      return (mean(c(cor_init, perm) >= cor_init))
#    }
#    # 输出结果
#    p_value <- perm_test(X, Y, num_perm)
#    cat("permutation test p-value:", p_value, "\n")
#    cor.test(X, Y, method = "spearman")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_9_3 <- function(n, discard) {
#    # 初始化
#    samples <- numeric(n)
#    samples[1] <- 0
#    # Metropolis-Hastings采样
#    for (i in 2:n) {
#      proposal <- samples[i - 1] + rnorm(1, mean = 0, sd = 1)
#      acceptance_ratio <- dcauchy(proposal) / dcauchy(samples[i - 1])
#      if (runif(1) < acceptance_ratio) {
#        samples[i] <- proposal
#      } else { samples[i] <- samples[i - 1] }
#    }
#    # 计算样本分位数和标准Cauchy分位数
#    samples <- samples[(discard + 1):n]
#    sample_deciles <- quantile(samples, probs = seq(0.1, 0.9, by = 0.1))
#    theoretical_deciles <- qcauchy(seq(0.1, 0.9, by = 0.1))
#    # 输出结果
#    cat("deciles of generated observations:", sample_deciles, "\n")
#    cat("deciles of standard Cauchy distribution:", theoretical_deciles, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_9_8 <- function(n, discard, a, b, n_val) {
#    # 初始化
#    x_samples <- numeric(n)
#    y_samples <- numeric(n)
#    x_samples[1] <- 5
#    y_samples[1] <- 0.5
#    # Gibbs采样
#    for (i in 2:n) {
#      # 更新x的条件分布Binomial(n, y)
#      y_current <- y_samples[i - 1]
#      x_samples[i] <- rbinom(1, size = n_val, prob = y_current)
#      # 更新y的条件分布Beta(x + a, n − x + b)
#      x_current <- x_samples[i]
#      y_samples[i] <- rbeta(1, shape1 = x_current + a, shape2 = n_val - x_current + b)
#    }
#    # 绘制联合分布图
#    plot(x_samples[(discard + 1):n], y_samples[(discard + 1):n], pch = 20, main = "joint density f(x,y)", xlab = "x", ylab = "y")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  converge_9_3 <- function(n, discard, n_chain) {
#    # 初始化
#    chains <- matrix(NA, nrow = n, ncol = n_chain)
#    # 运行多条马尔可夫链
#    for (j in 1:n_chain) {
#      chain <- numeric(n)
#      chain[1] <- 0
#      for (i in 2:n) {
#        proposal <- chain[i - 1] + rnorm(1, mean = 0, sd = 1)
#        acceptance_ratio <- dcauchy(proposal) / dcauchy(chain[i - 1])
#        if (runif(1) < acceptance_ratio) {
#          chain[i] <- proposal
#        } else { chain[i] <- chain[i - 1] }
#      }
#      chains[, j] <- chain
#    }
#    # 转换为mcmc.list对象
#    mcmc_chains <- coda::mcmc.list(lapply(1:n_chain, function(j) mcmc(chains[(discard + 1):n, j])))
#    # Gelman-Rubin收敛诊断
#    gelman_diag <- coda::gelman.diag(mcmc_chains)$psrf[, "Point est."]
#    cat("convergence of chains:", gelman_diag, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  converge_9_8 <- function(n, discard, a, b, n_val, n_chain) {
#    # 初始化
#    x_chains <- matrix(NA, nrow = n, ncol = n_chain)
#    y_chains <- matrix(NA, nrow = n, ncol = n_chain)
#    # 运行多条马尔可夫链
#    for (j in 1:n_chain) {
#      x_chain <- numeric(n)
#      y_chain <- numeric(n)
#      x_chain[1] <- 5
#      y_chain[1] <- 0.5
#      for (i in 2:n) {
#        y_current <- y_chain[i - 1]
#        x_chain[i] <- rbinom(1, size = n_val, prob = y_current)
#        x_current <- x_chain[i]
#        y_chain[i] <- rbeta(1, shape1 = x_current + a, shape2 = n_val - x_current + b)
#      }
#      x_chains[, j] <- x_chain
#      y_chains[, j] <- y_chain
#    }
#    # 转换为mcmc.list对象
#    x_mcmc_chains <- coda::mcmc.list(lapply(1:n_chain, function(j) mcmc(x_chains[(discard + 1):n, j])))
#    y_mcmc_chains <- coda::mcmc.list(lapply(1:n_chain, function(j) mcmc(y_chains[(discard + 1):n, j])))
#    # Gelman-Rubin收敛诊断
#    gelman_diag_x <- coda::gelman.diag(x_mcmc_chains)$psrf[, "Point est."]
#    gelman_diag_y <- coda::gelman.diag(y_mcmc_chains)$psrf[, "Point est."]
#    cat("convergence of x_chains:", gelman_diag_x, "\n")
#    cat("convergence of y_chains:", gelman_diag_y, "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_11_3 <- function(a, d, max_k) {
#    # compute the kth term
#    compute_term <- function(a, d, k) {
#      norm_a <- sqrt(sum(a^2))
#      term <- ((-1)^k / (factorial(k) * 2^k)) *
#            (norm_a^(2 * k + 2) / ((2 * k + 1) * (2 * k + 2))) *
#            (gamma((d + 1) / 2) * gamma(k + 3 / 2) / gamma(k + d / 2 + 1))
#      return(term)
#    }
#    # computes and returns the sum
#    compute_sum <- function(a, d, max_k) {
#      summation <- sum(sapply(0:max_k, function(k) compute_term(a, d, k)))
#      return(summation)
#    }
#    # 输出结果
#    cat("the sum when a = (1, 2):", compute_sum(a, d, max_k), "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_11_5 <- function(k_values) {
#    # function of the integral
#    ratio <- function(k) {
#      return(2 * gamma((k + 1) / 2) / (sqrt(pi * k) * gamma(k / 2)))
#    }
#    ck <- function(a, k) {
#      return(sqrt((a^2 * k) / (k + 1 - a^2)))
#    }
#    integral_left <- function(a, k) {
#      result <- integrate(function(u) (1 + u^2 / (k - 1))^(-k / 2), lower = 0, upper = ck(a, k - 1))
#      return(result$value)
#    }
#    integral_right <- function(a, k) {
#      result <- integrate(function(u) (1 + u^2 / k)^(-(k + 1) / 2), lower = 0, upper = ck(a, k))
#      return(result$value)
#    }
#    # solve the equation for a
#    find_a <- function(k) {
#      f <- function(a) {
#        left_term <- ratio(k - 1) * integral_left(a, k)
#        right_term <- ratio(k) * integral_right(a, k)
#        return(abs(left_term - right_term))
#      }
#      solution <- optimize(f, interval = c(0.01, sqrt(k)))$minimum
#      return(solution)
#    }
#    # 输出结果
#    a_values <- sapply(k_values, find_a)
#    print(data.frame(k = k_values, a = a_values))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  em_algo <- function(epsilon, max_iter) {
#    Y <- c(0.54, 0.48, 0.33, 0.43, 1.00, 1.00, 0.91, 1.00, 0.21, 0.85)
#    tau <- 1
#    lambda <- 1
#    for (iter in 1:max_iter) {
#      # E-step
#      uncensor <- Y[Y < tau]
#      censor <- Y[Y == tau]
#      n_uncensor <- length(uncensor)
#      n_censor <- length(censor)
#      E_T_censor <- tau + 1 / lambda
#      # M-step
#      lambda_iter <- (n_uncensor + n_censor) / (sum(uncensor) + n_censor * E_T_censor)
#      if (abs(lambda_iter - lambda) < epsilon) {
#        lambda <- lambda_iter
#        break }
#      lambda <- lambda_iter
#    }
#    # 输出结果
#    cat("the E-M algorithm to estimate λ:", lambda, "\n")
#    cat("the observed data MLE:", 1 / mean(Y), "\n")
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_11_7 <- function() {
#    f.obj <- c(4, 2, 9)
#    f.con <- matrix(c(2, 1, 1, 1, -1, 3), nrow = 2, byrow = TRUE)
#    f.dir <- c("<=", "<=")
#    f.rhs <- c(2, 3)
#    # 使用lp函数求解最小化问题
#    solution <- lpSolve::lp("min", f.obj, f.con, f.dir, f.rhs)
#    # 输出结果
#    solution$objval
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_204_3 <- function() {
#    datasets::data(mtcars)
#    formulas <- list(
#      mpg ~ disp,
#      mpg ~ I(1 / disp),
#      mpg ~ disp + wt,
#      mpg ~ I(1 / disp) + wt
#    )
#    # for loops to fit linear models
#    lm_for <- list()
#    for (i in 1:length(formulas)) {
#      lm_for[[i]] <- lm(formulas[[i]], data = mtcars)
#    }
#    # lapply() to fit linear models
#    lm_lapply <- lapply(formulas, function(formula) lm(formula, data = mtcars))
#    # 输出拟合结果
#    lm_for
#    lm_lapply
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_204_4 <- function() {
#    datasets::data(mtcars)
#    bootstraps <- lapply(1:10, function(i) {
#      rows <- sample(1:nrow(mtcars), replace = TRUE)
#      mtcars[rows, ]
#    })
#    # using a for loop
#    boot_for <- list()
#    for (i in 1:length(bootstraps)) {
#      boot_for[[i]] <- lm(mpg ~ disp, data = bootstraps[[i]])
#    }
#    # using lapply()
#    boot_lapply <- lapply(bootstraps, function(data) lm(mpg ~ disp, data = data))
#    # 输出拟合结果
#    boot_for
#    boot_lapply
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_204_5 <- function() {
#    datasets::data(mtcars)
#    formulas <- list(
#      mpg ~ disp,
#      mpg ~ I(1 / disp),
#      mpg ~ disp + wt,
#      mpg ~ I(1 / disp) + wt
#    )
#    lm_for <- list()
#    for (i in 1:length(formulas)) {
#      lm_for[[i]] <- lm(formulas[[i]], data = mtcars)
#    }
#    lm_lapply <- lapply(formulas, function(formula) lm(formula, data = mtcars))
#    bootstraps <- lapply(1:10, function(i) {
#      rows <- sample(1:nrow(mtcars), replace = TRUE)
#      mtcars[rows, ]
#    })
#    boot_for <- list()
#    for (i in 1:length(bootstraps)) {
#      boot_for[[i]] <- lm(mpg ~ disp, data = bootstraps[[i]])
#    }
#    boot_lapply <- lapply(bootstraps, function(data) lm(mpg ~ disp, data = data))
#    # extract R2 using the function below
#    rsq <- function(mod) summary(mod)$r.squared
#    rsq_lm_for <- sapply(lm_for, rsq)
#    rsq_lm_lapply <- sapply(lm_lapply, rsq)
#    rsq_boot_for <- sapply(boot_for, rsq)
#    rsq_boot_lapply <- sapply(boot_lapply, rsq)
#    # 输出 R² 值
#    rsq_lm_for
#    rsq_lm_lapply
#    rsq_boot_for
#    rsq_boot_lapply
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_213_3 <- function() {
#    trials <- replicate(
#      100,
#      t.test(rpois(10, 10), rpois(7, 10)),
#      simplify = FALSE
#    )
#    # Use sapply() and an anonymous function
#    p_val <- sapply(trials, function(x) x$p.value)
#    p_val
#    # Extra challenge
#    p_val_ex <- sapply(trials, `[[`, "p.value")
#    p_val_ex
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_214_6 <- function() {
#    parallel_lapply <- function(..., FUN, FUN.VALUE = NA) {
#      results_list <- Map(FUN, ...)
#      vapply(results_list, FUN.VALUE = FUN.VALUE, FUN = identity)
#    }
#    x <- 1:5
#    y <- 6:10
#    result <- parallel_lapply(x, y, FUN = function(a, b) a + b, FUN.VALUE = numeric(1))
#    result
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_365_4 <- function() {
#    # Make a faster version of chisq.test()
#    fast_chisq_test <- function(x, y) {
#      if (any(is.na(x)) || any(is.na(y)) || length(x) != length(y)) {
#        stop("The input must be two numeric vectors with no missing values.")
#      }
#      table_xy <- table(x, y)
#      row_totals <- margin.table(table_xy, 1)
#      col_totals <- margin.table(table_xy, 2)
#      grand_total <- sum(table_xy)
#      expected_counts <- outer(row_totals, col_totals, FUN = "*") / grand_total
#      chisq_stat <- sum((table_xy - expected_counts)^2 / expected_counts)
#      return(chisq_stat)
#    }
#    # Example
#    x <- c(1, 1, 2, 2, 3, 3, 4, 4)
#    y <- c(1, 2, 3, 4, 1, 2, 3, 4)
#    fast_chisq_test(x, y)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_365_5 <- function() {
#    # Make a faster version of table()
#    fast_table <- function(x, y) {
#      factor_x <- as.integer(factor(x))
#      factor_y <- as.integer(factor(y))
#      n_x <- length(unique(factor_x))
#      n_y <- length(unique(factor_y))
#      table_result <- matrix(0, nrow = n_x, ncol = n_y)
#      for (i in 1:length(x)) {
#        table_result[factor_x[i], factor_y[i]] <- table_result[factor_x[i], factor_y[i]] + 1
#      }
#      rownames(table_result) <- levels(factor(x))
#      colnames(table_result) <- levels(factor(y))
#      return(table_result)
#    }
#    # Example
#    x <- c(1, 1, 2, 2, 3, 3, 4, 4)
#    y <- c(1, 2, 3, 4, 1, 2, 3, 4)
#    fast_table(x, y)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  answer_9_8_Rccp <- function(N, n, a, b) {
#    # 设定 Rcpp 环境
#    Rcpp::cppFunction('
#    NumericMatrix gibbsRcpp(int N, int n, double a, double b) {
#        NumericMatrix result(N, 2);
#        double y = R::runif(0, 1); // 初始 y 值
#        for (int i = 0; i < N; i++) {
#            // 更新 x | y ~ Binomial(n, y)
#            int x = R::rbinom(n, y);
#            // 更新 y | x ~ Beta(x + a, n - x + b)
#            y = R::rbeta(x + a, n - x + b);
#            result(i, 0) = x;
#            result(i, 1) = y;
#        }
#        return result;
#    }')
#    # R 语言的 Gibbs 采样函数
#    gibbsR <- function(N, n, a, b) {
#        result <- matrix(NA, nrow = N, ncol = 2)
#        y <- runif(1)  # 初始 y 值
#        for (i in 1:N) {
#            # 更新 x | y ~ Binomial(n, y)
#            x <- rbinom(1, n, y)
#            # 更新 y | x ~ Beta(x + a, n - x + b)
#            y <- rbeta(1, x + a, n - x + b)
#            result[i, ] <- c(x, y)
#        }
#        return(result)
#    }
#    # 使用 Rcpp 和 R 函数生成样本
#    samples_rcpp <- gibbsRcpp(N, n, a, b)
#    samples_r <- gibbsR(N, n, a, b)
#    # Q-Q 图比较
#    par(mfrow = c(1, 2))
#    qqplot(samples_rcpp[, 1], samples_r[, 1], main = "Q-Q plot for X", xlab = "Rcpp", ylab = "R")
#    abline(0, 1, col = "black")
#    qqplot(samples_rcpp[, 2], samples_r[, 2], main = "Q-Q plot for Y", xlab = "Rcpp", ylab = "R")
#    abline(0, 1, col = "black")
#    # 计算性能比较
#    benchmark_result <- microbenchmark::microbenchmark(
#        Rcpp = gibbsRcpp(N, n, a, b),
#        R = gibbsR(N, n, a, b)
#    )
#    print(benchmark_result)
#  }

