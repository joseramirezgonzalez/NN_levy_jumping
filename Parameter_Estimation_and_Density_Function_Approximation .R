###############################################################
#                                                             #
#  ------ Code for estimating the density function of ------  #
#  ------ X_{t+Δt} given 𝔽(X_t), and inference of ------      #
#  ------       the parameters λ and γ                 ------ #
#                                                             #
###############################################################

#-----Libraries-----
library(torch)  # Loads the 'torch' library for neural networks and tensor computation
library(MASS)   # Loads the 'MASS' library for statistical functions, including 'mvrnorm'

#-----Clear memory-----
rm(list = ls())  # Removes all objects from the current R environment

###############################################################
#                                                             #
#       ------ Drift and Diffusion coefficients ------        #
#                                                             #
###############################################################

#------Drift coefficient f-----------
f <- function(x) {
  return(0.17*(x-x^3))  # Drift function: f(x) = sin(x)
}
#------Diffusion coefficient g--------
g <- function(x) {
  return(0.76*(1+cos(x)))  # Diffusion function: g(x) = 0.35x + 0.2
}

#------Derivative of g---------------
d_g <- function(x) {
  return(-0.76*sin(x))  # Derivative of g(x) with respect to x
}

###############################################################
#                                                             #
#          ------ Simulation trajectories ------              #
#                                                             #
###############################################################


#--------Function to simulate trajectories of X_{t + Δt} given 𝔽(X_t)--------
simulate_sample <- function(m, Delta_t, t, X_t, g, d_g, f_X_std, gamma,lambda) {
  # Simulación del número de eventos Poisson para cada muestra
  sim_pois_sample <- rpois(m, lambda*Delta_t)
  # Create a list structure to store the jump times
  times_sample <- vector("list", m)
  for (i in 1:m) {
    if (sim_pois_sample[i] > 0) {
      times_sample[[i]] <- sort(runif(sim_pois_sample[i], t, t + Delta_t))
    } else {
      times_sample[[i]] <- numeric(0)
    }
  }
  # Precompute constant values
  g_X_t <- g(X_t)
  d_g_X_t <- d_g(X_t)
  constant_term <- 0.5 * g_X_t * d_g_X_t
  adjusted_X_t <- X_t + (f_X_std - constant_term) * Delta_t
  # Simulate sample paths
  sample_sim <- numeric(m)
  for (i in 1:m) {
    if (sim_pois_sample[i] == 0) {
      aux <- rnorm(1, 0, sqrt(Delta_t))
      sample_sim[i] <- aux * g_X_t + 
                       constant_term * aux^2 + 
                       adjusted_X_t
    } else {
      # Time differences
      diff_times <- diff(c(t, times_sample[[i]], t + Delta_t))
      Sigma <- diag(diff_times)
      # Process simulation
      simula_W_diff <- mvrnorm(1, mu = rep(0, sim_pois_sample[i] + 1), Sigma = Sigma)
      simula_z <- runif(sim_pois_sample[i], -0.1, 0.1)     # Simulation of the jumps
      # Compute accumulated values for g
      g_evaluated <- g(X_t + gamma * simula_z)
      g_accumulated <- c(0, cumsum(g_evaluated - g_X_t))
      aux_vec <- g_X_t + g_accumulated
      #---Simulation--
      aux_1 <- sum(simula_W_diff * aux_vec)
      aux_2 <- constant_term * sum(simula_W_diff)^2
      aux_3 <- gamma * sum(simula_z) + adjusted_X_t
      sample_sim[i] <- aux_1 + aux_2 + aux_3
    }
  }
  return(sample_sim)
}


###############################################################
#                                                             #
# ------ Functions to construct the approximate density ----- #
#  ------   of X_{t+Δt} given \mathcal{F}(X_t)          ------#
#                                                             #
###############################################################

#------Construction of the density function via Yang and the characteristic function of X_{t+Δt} given \mathcal{F}(X_t)
density_function <- function(u, sample, M, h) {
  # Inicializar tensor para almacenar la densidad
  density <- torch_zeros(u$size(1), dtype = torch_cfloat())
  pi_torch <- torch_tensor(pi, dtype = torch_float32())  # Definir pi como tensor Torch
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float32()), torch_tensor(1, dtype = torch_float32()))  # Definir 1i
  
  # Calcular densidad para cada u[i]
  for (i in seq_len(u$size(1))) {
    aux <- torch_zeros(1, dtype = torch_cfloat())  # Inicializar acumulador
    for (j in seq(-M, M)) {
      idx <- j + M + 1  # Índice para sample
      # Calcular términos y acumular
      term_exp <- torch_exp(-j * h * u[i] * one_i)
      aux <- aux + term_exp * sample[idx]*h
    }
    
    # Asignar valor a density
    density[i] <- aux / (2 * pi_torch)
  }
  
  return(density)  
}

# Get epsilon for maximum precision (float64 in PyTorch)
epsilon <- torch_finfo(torch_float64())$eps


#-------Characteristic function of X_{t+Δt} given \mathcal{F}(X_t) without jumps-----
Aproximation_density_red_W <- function(X_t_next, u, t, Delta_t, X_t, M, h, predicted_f, predicted_g) {
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float32()), torch_tensor(1, dtype = torch_float32())) 
  two_i <- 2 * one_i  
  predic_g_X <- torch_tensor(predicted_g(X_t), dtype = torch_float32())
  predic_f_X <- torch_tensor(predicted_f(X_t), dtype = torch_float32())
  predic_f_X_s<-predic_f_X/(1+Delta_t*predic_f_X^2)
  d_g_X_t<-torch_tensor(d_g(X_t), dtype = torch_float32())
  c <- 0.5 * predic_g_X * d_g_X_t
  X_shift <- X_t + (predic_f_X_s - c) * Delta_t
  Delta_t<-torch_squeeze(Delta_t, dim = 1)
  t<-torch_squeeze(t, dim = 1)
  tam_u <- u$size(1)
  aux <- two_i * c * u / (1 - two_i * c * u * Delta_t)
  phi <- torch_zeros(tam_u, dtype = torch_cfloat())
  u_squared <- torch_abs(u)^2
  common_factor <- 1 / torch_sqrt(1 - two_i * u * c * Delta_t)
  common_exp <- torch_exp(-0.5 * u_squared * (predic_g_X^2) *(Delta_t+(Delta_t^2)*aux))
  exp_X_shift <- torch_exp(one_i * u * X_shift)
  phi <- exp_X_shift * common_factor * common_exp
  density <- density_function(X_t_next, phi, M, h)
  return(torch_abs(torch_real(density)))
}


#-------Characteristic function of X_{t+Δt} given \mathcal{F}(X_t) with jumps-----
Aproximation_density_red_2 <- function(X_t_1, u, n, t, Delta_t, X_t, M, h, predicted_f, predicted_g,gamma,lambda) {
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float32()), torch_tensor(1, dtype = torch_float32())) 
  two_i <- 2 * one_i  # 2i como tensor complejo
  gamma <- torch_tensor(gamma, dtype = torch_float32())
  predic_g_X <- torch_tensor(predicted_g(X_t), dtype = torch_float32())
  predic_f_X <- torch_tensor(predicted_f(X_t), dtype = torch_float32())
   predic_f_X_s<-predic_f_X/(1+Delta_t*predic_f_X^2)
  d_g_X_t<-torch_tensor(d_g(X_t), dtype = torch_float32())
  c <- 0.5 * predic_g_X * d_g_X_t
  X_shift <- X_t + (predic_f_X_s - c) * Delta_t
  Delta_t<-torch_squeeze(Delta_t, dim = 1)
  t<-torch_squeeze(t, dim = 1)
  sim_pois <- rpois(n, lambda*as.numeric(Delta_t))
  times <- vector("list", n) # Create a list structure to store the jump times
  for (i in seq_len(n)) {
      if (sim_pois[i] > 0) {
         times[[i]] <- sort(runif(sim_pois[i], as.numeric(t), as.numeric(t) + as.numeric(Delta_t)))
       } else {
            times[[i]] <- numeric(0)  
       }
  }
  tam_u <- u$size(1)
  aux <- two_i * c * u / (1 - two_i * c * u * Delta_t)
  phi <- torch_zeros(n, tam_u, dtype = torch_cfloat())
  u_squared <- torch_abs(u)^2
  common_factor <- 1 / torch_sqrt(1 - two_i * u * c * Delta_t)
  common_exp <- torch_exp(-0.5 * u_squared * (predic_g_X^2) *(Delta_t+(Delta_t^2)*aux))
  exp_X_shift <- torch_exp(one_i * u * X_shift)
  for (i in seq_len(n)) {
    if (sim_pois[i] == 0) {
      phi[i, ] <- exp_X_shift * common_factor * common_exp
    
    } else {
    	  simula_z<-torch_tensor(runif(sim_pois[i],-0.1,0.1), dtype = torch_float32()) # Simulation of the jumps
      values <-torch_tensor(predicted_g(X_t + gamma * simula_z), dtype = torch_float32())
      values_subtracted <- values - predic_g_X
      g_accumulated <- torch_cumsum(values_subtracted, dim = 1)
      g_accumulated <- torch_squeeze(g_accumulated, dim = 1)
      a <- torch_cat(list(torch_zeros(1, dtype = torch_float32()), g_accumulated),dim=1)+predic_g_X
      times_i_tensor <- torch_tensor(times[[i]], dtype = torch_float32())
      times_concat_1 <- torch_cat(list(times_i_tensor, t + Delta_t), dim = 1)  # c(times[[i]], t + Delta_t)
      times_concat_2 <- torch_cat(list(t, times_i_tensor), dim = 1)  # c(t, times[[i]])
      diff_times <- times_concat_1 - times_concat_2
      v<-a*diff_times
      # Calcular M_1 y M_2
      M_1 <- torch_sum(diff_times * a^2)
      M_2 <- torch_sum(v)^2
      phi[i, ] <- common_factor * exp_X_shift*torch_exp(one_i*u*gamma*torch_sum(simula_z)- 0.5 *u_squared * (M_1 + aux * M_2))
    }
  }
  sample <- torch_mean(phi, dim = 1)
  density <- density_function(X_t_1, sample, M, h)
  return(torch_abs(torch_real(density)))
}

# ================= Estimation and simulation of a random sample =================

#-----Parameters
gamma<-0.8
lambda<-0.94
M<-2000 
h<-0.05 
n<-150 
h_2<-0.01
x<-2.3
Delta<-0.5 
t<-0
x_next<-f(x)/(1+(Delta)*f(x)^2)*Delta+x

# Create Torch tensors to evaluate the approximated density function
x_t_a <- torch_tensor(matrix(x, ncol = 1))#
x_next_a <- torch_tensor(matrix(x_next, ncol = 1))#
delta_t_a <- torch_tensor(matrix(Delta, ncol = 1))#
t_a <- torch_tensor(matrix(t, ncol = 1))
reg_test<-seq(0,3,by=h_2)

#-----The approximated density function
with_no_grad({
aux_reg<-as.numeric(Aproximation_density_red_2(
       torch_tensor(reg_test, dtype = torch_float32()),  
      torch_tensor(seq(-M * h, M * h, by = h), dtype = torch_float32()),  
      n, 
       t_a[1, , drop = FALSE], 
      delta_t_a[1, , drop = FALSE],  
      x_t_a[1, , drop = FALSE],  
      M,  
      h, 
      f,  
     g,   
     gamma,
     lambda
    ))
  })

#-------Simulations
m<-100000#
X<-rep(0,m)
f_X_std<-f(x)/(1+(Delta)*f(x)^2)
for(i in 1:m){
	X[i]<-simulate_sample(1,Delta, t, x, g, d_g, f_X_std, gamma,lambda)#
}


library(ggplot2)
library(latex2exp)


###############################################################
#                                                             #
#                                                             #
#  X        : vector of simulated data                        #
#  x_next   : conditional expectation                         #
#  reg_test : points at which the approximate density is      #
#             evaluated                                       #
#  aux_reg  : values of the approximated density              #
###############################################################

#------Data frames and legends-------
df_hist <- data.frame(X = X)
df_curve <- data.frame(x = reg_test, y = aux_reg)
df_line <- data.frame(x = x_next)
# String of parameter expressions for use with latex2exp 
param_text <- TeX(paste(
  "$f(x) = 0.17(x-x^3),$",
  "$g(x) = 0.76(1 + \\cos(x)),$",#
  "$\\Delta t = 0.5,$",
  "$t = 0,$",#
  "$X_t = 2.3,$",
  "$h = 0.05,$",
  "$M = 2000,$",
  "$a = 0,$",
  "$\\gamma = 0.25,$",
  "$\\lambda = 0.81$",
  sep = "\n"
))


###############################################################
#   Histogram of the simulated sample vs the estimated        #
#   approximated density                                      #
###############################################################
ggplot() +
  geom_histogram(data = df_hist, aes(x = X, y = ..density.., fill = "a"), 
                 bins = round(sqrt(length(X))), color = "black", alpha = 0.3) +
  geom_line(data = df_curve, aes(x = x, y = y, color = "b"), size = 1.2) +
  geom_vline(data = df_line, aes(xintercept = x, color = "c"), 
             linewidth = 1.2, linetype = "dashed") +
  scale_color_manual(values = c("b" = "blue", "c" = "red"),
                     labels = c("b" = TeX("$f_{t,\\Delta t}^{M,h,a}(x)$"),
                                "c" = TeX("$E(X_{t+\\Delta t}\\,|\\,F(X_t))$"))) +
  scale_fill_manual(values = c("a" = "blue"),#
                    labels = c("a" = "Simulated Data")) +
  geom_label(data = data.frame(x = 0.15, y = 2.7),
             aes(x = x, y = y), label = param_text,
             hjust = 0, vjust = 1, size = 4.2,
             fill = "white", color = "black", parse = TRUE) +
  labs(title = "Density Estimation via Fourier Transform", x = TeX("$X_{t+\\Delta t}$"), y = "Density") +
  theme_minimal(base_size = 14) +
  theme(legend.title = element_blank()) +
  coord_cartesian(xlim = c(0, 3), ylim = c(0, 2.8)) +
  guides(fill = guide_legend(override.aes = list(alpha = 0.3)))

#------


###############################################################
#                   Inference of the parameters γ and λ       #
###############################################################


#-------Simulated data-------




###############################################################
#                                                             #
#       ------ Drift and Diffusion coefficients ------        #
#                                                             #
###############################################################

#------Drift coefficient f-----------
f <- function(x) {
  return(sin(x))  # Drift function: f(x) = sin(x)
}
#------Diffusion coefficient g--------
g <- function(x) {
  return(0.35*x+0.2)  # Diffusion function: g(x) = 0.35x + 0.2
}

#------Derivative of g---------------
d_g <- function(x) {
  return(0.35)  # Derivative of g(x) with respect to x
}



#-------Characteristic function of X_{t+Δt} given \mathcal{F}(X_t) with jumps-----
Aproximation_density_red_2 <- function(X_t_1, u, n, t, Delta_t, X_t, M, h, predicted_f, predicted_g,gamma,lambda) {
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float32()), torch_tensor(1, dtype = torch_float32())) 
  two_i <- 2 * one_i  # 2i como tensor complejo
  gamma <- torch_tensor(gamma, dtype = torch_float32())
  predic_g_X <- torch_tensor(predicted_g(X_t), dtype = torch_float32())
  predic_f_X <- torch_tensor(predicted_f(X_t), dtype = torch_float32())
   predic_f_X_s<-predic_f_X/(1+Delta_t*predic_f_X^2)
  d_g_X_t<-torch_tensor(d_g(X_t), dtype = torch_float32())
  c <- 0.5 * predic_g_X * d_g_X_t
  X_shift <- X_t + (predic_f_X_s - c) * Delta_t
  Delta_t<-torch_squeeze(Delta_t, dim = 1)
  t<-torch_squeeze(t, dim = 1)
  sim_pois <- rpois(n, lambda*as.numeric(Delta_t))
  times <- vector("list", n) # Create a list structure to store the jump times
  for (i in seq_len(n)) {
      if (sim_pois[i] > 0) {
         times[[i]] <- sort(runif(sim_pois[i], as.numeric(t), as.numeric(t) + as.numeric(Delta_t)))
       } else {
            times[[i]] <- numeric(0)  
       }
  }
  tam_u <- u$size(1)
  aux <- two_i * c * u / (1 - two_i * c * u * Delta_t)
  phi <- torch_zeros(n, tam_u, dtype = torch_cfloat())
  u_squared <- torch_abs(u)^2
  common_factor <- 1 / torch_sqrt(1 - two_i * u * c * Delta_t)
  common_exp <- torch_exp(-0.5 * u_squared * (predic_g_X^2) *(Delta_t+(Delta_t^2)*aux))
  exp_X_shift <- torch_exp(one_i * u * X_shift)
  for (i in seq_len(n)) {
    if (sim_pois[i] == 0) {
      phi[i, ] <- exp_X_shift * common_factor * common_exp
    
    } else {
    	  simula_z<-torch_tensor(runif(sim_pois[i],-0.5,0.5), dtype = torch_float32()) # Simulation of the jumps
      values <-torch_tensor(predicted_g(X_t + gamma * simula_z), dtype = torch_float32())
      values_subtracted <- values - predic_g_X
      g_accumulated <- torch_cumsum(values_subtracted, dim = 1)
      g_accumulated <- torch_squeeze(g_accumulated, dim = 1)
      a <- torch_cat(list(torch_zeros(1, dtype = torch_float32()), g_accumulated),dim=1)+predic_g_X
      times_i_tensor <- torch_tensor(times[[i]], dtype = torch_float32())
      times_concat_1 <- torch_cat(list(times_i_tensor, t + Delta_t), dim = 1)  # c(times[[i]], t + Delta_t)
      times_concat_2 <- torch_cat(list(t, times_i_tensor), dim = 1)  # c(t, times[[i]])
      diff_times <- times_concat_1 - times_concat_2
      v<-a*diff_times
      # Calcular M_1 y M_2
      M_1 <- torch_sum(diff_times * a^2)
      M_2 <- torch_sum(v)^2
      phi[i, ] <- common_factor * exp_X_shift*torch_exp(one_i*u*gamma*torch_sum(simula_z)- 0.5 *u_squared * (M_1 + aux * M_2))
    }
  }
  sample <- torch_mean(phi, dim = 1)
  density <- density_function(X_t_1, sample, M, h)
  return(torch_abs(torch_real(density)))
}




###############################################################
#                                                             #
#          ------ Simulation trajectories ------              #
#                                                             #
###############################################################


#--------Function to simulate trajectories of X_{t + Δt} given 𝔽(X_t)--------
simulate_sample <- function(m, Delta_t, t, X_t, g, d_g, f_X_std, gamma,lambda) {
  # Simulación del número de eventos Poisson para cada muestra
  sim_pois_sample <- rpois(m, lambda*Delta_t)
  # Create a list structure to store the jump times
  times_sample <- vector("list", m)
  for (i in 1:m) {
    if (sim_pois_sample[i] > 0) {
      times_sample[[i]] <- sort(runif(sim_pois_sample[i], t, t + Delta_t))
    } else {
      times_sample[[i]] <- numeric(0)
    }
  }
  # Precompute constant values
  g_X_t <- g(X_t)
  d_g_X_t <- d_g(X_t)
  constant_term <- 0.5 * g_X_t * d_g_X_t
  adjusted_X_t <- X_t + (f_X_std - constant_term) * Delta_t
  # Simulate sample paths
  sample_sim <- numeric(m)
  for (i in 1:m) {
    if (sim_pois_sample[i] == 0) {
      aux <- rnorm(1, 0, sqrt(Delta_t))
      sample_sim[i] <- aux * g_X_t + 
                       constant_term * aux^2 + 
                       adjusted_X_t
    } else {
      # Time differences
      diff_times <- diff(c(t, times_sample[[i]], t + Delta_t))
      Sigma <- diag(diff_times)
      # Process simulation
      simula_W_diff <- mvrnorm(1, mu = rep(0, sim_pois_sample[i] + 1), Sigma = Sigma)
      simula_z <- runif(sim_pois_sample[i], -0.5, 0.5)     # Simulation of the jumps
      # Compute accumulated values for g
      g_evaluated <- g(X_t + gamma * simula_z)
      g_accumulated <- c(0, cumsum(g_evaluated - g_X_t))
      aux_vec <- g_X_t + g_accumulated
      #---Simulation--
      aux_1 <- sum(simula_W_diff * aux_vec)
      aux_2 <- constant_term * sum(simula_W_diff)^2
      aux_3 <- gamma * sum(simula_z) + adjusted_X_t
      sample_sim[i] <- aux_1 + aux_2 + aux_3
    }
  }
  return(sample_sim)
}




#-------True parameters of the jump component
       
gamma<-2.4
lambda<-1.7






# ====== Simulated data ====== #

Delta<-0.5
t<-0
x<-1.5
mu_2<-(2/3)*(0.5^3)
m<-250
X<-rep(0,m)
f_X_std<-f(x)/(1+(Delta)*f(x)^2)
set.seed(25)

for(i in 1:m){
	X[i]<-simulate_sample(1,Delta, t, x, g, d_g, f_X_std, gamma,lambda)
}


#---- Tensors to evaluate the approximated likelihood function------


x_t_a <- torch_tensor(matrix(x, ncol = 1))
delta_t_a <- torch_tensor(matrix(Delta, ncol = 1))
t_a <- torch_tensor(matrix(t, ncol = 1))


#----parametros aproximacion likelihood
M<-200
h<-0.05
n<-200



# ====== Histogram of simulated data ====== #

df <- data.frame(X = X)
ggplot(df, aes(x = X)) +
  geom_histogram(bins = 30, 
                 fill = "#0073C2FF", 
                 color = "white", 
                 alpha = 0.8) +
  theme_minimal(base_size = 14) +
  labs(
    title = "",
    x = "Value",
    y = "Frequency"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold")
  )







#######################################################################
#                                                                     #
#     Inference of the parameters γ and λ: Algorithm 2                #
#                                                                     #
#######################################################################



# Number of simulations to estimate the second moment
n_1<-4000


# ====== h(λ, γ | (x_{t+Δt}^{(1)}, ..., x_{t+Δt}^{(N)})) ====== #

second_moment_estimator<-function(gamma,lambda){
  g_X_t <- g(x)
  d_g_X_t <- d_g(x)
  constant_term <- Delta*(g_X_t^2)+0.5*(g_X_t * d_g_X_t*Delta)^2+(gamma^2)*mu_2*lambda*Delta
  adjusted_X_t <- x + f_X_std * Delta
  val_1<-(X-adjusted_X_t)^2
  sim_pois_sample <- rpois(n_1, lambda*Delta)
  times_sample <- vector("list", n_1)
  for (i in 1:n_1) {
    if (sim_pois_sample[i] > 0) {
      times_sample[[i]] <- sort(runif(sim_pois_sample[i], t, t + Delta))
    } else {
      times_sample[[i]] <- numeric(0)
    }
  }
  sample_sim <- numeric(n_1)
  for (i in 1:n_1) {
    if (sim_pois_sample[i] == 0) {
      sample_sim[i] <- constant_term 
    } else {
      diff_times <- t + Delta-times_sample[[i]]

      simula_z <- runif(sim_pois_sample[i], -0.5, 0.5)
      g_evaluated <- g(x + gamma * simula_z)-g_X_t
      aux_2<-2*g_X_t*sum(g_evaluated*diff_times)
      A <- tcrossprod(g_evaluated)  # Matriz A_ij = a_i * a_j  (Producto exterior)
      B <- outer(1:sim_pois_sample[i], 1:sim_pois_sample[i], pmax)  # Matriz de índices max(i, j)
      aux_3<- sum(A * diff_times[B])
      sample_sim[i] <- constant_term + aux_2 + aux_3
    }
  }
 mean_sim<-mean(sample_sim)
 return(mean((val_1-mean_sim)^2))
}



#------Initial value for the MH algorithm------
# Optimize h(λ, γ | (x_{t+Δt}^{(1)}, ..., x_{t+Δt}^{(N)})) via "Nelder-Mead"

second_moment_estimator_optim <- function(params) {
  gamma <- params[1]
  lambda <- params[2]
  return(exp(second_moment_estimator(gamma, lambda)))
}
set.seed(23)
start_vals <- c(gamma = rexp(1), lambda = rexp(1))#
opt_result <- optim(
  par = start_vals,
  fn = second_moment_estimator_optim,
  method = "Nelder-Mead",
  control = list(maxit = 10000, reltol = 1e-12)
)
# Optimal results
MLE <- opt_result$par     # Optimal values of gamma and lambda
opt_result$value          # Minimum value found of the negative log-likelihood
max_MLE<-as.numeric(opt_result$value)
gamma_MLE<-as.numeric(MLE[1])
lambda_MLE<-as.numeric(MLE[2])
#----Initial value
gamma_MLE
lambda_MLE



# ====== MCMC with h(λ, γ | (x_{t+Δt}^{(1)}, ..., x_{t+Δt}^{(N)})) function ====== #


mcmc_metropolis_lognorm <- function(second_moment_estimator, gamma0, lambda0, n_iter , sigma_gamma = 0.01, sigma_lambda = 0.05) {
  # Initialize vectors to store samples
  gamma_samples <- numeric(n_iter)
  lambda_samples <- numeric(n_iter)
  # Initial values
  gamma_current <- gamma0
  lambda_current <- lambda0
  loss_likelihood_current <- exp(second_moment_estimator(gamma_current, lambda_current))
  for (i in 1:n_iter) {
    gamma_proposed <- rlnorm(1, mean = log(gamma_current)-(sigma_gamma^2)/2, sd = sigma_gamma)
    lambda_proposed <- rlnorm(1, mean = log(lambda_current)-(sigma_lambda^2)/2, sd = sigma_lambda)
    # Evaluation of the new likelihood
    loss_likelihood_proposed <- exp(second_moment_estimator(gamma_proposed, lambda_proposed))
    # Compute the acceptance ratio with density correction
    acceptance_ratio <-exp(5*(loss_likelihood_current-loss_likelihood_proposed))
    # Accept or reject the proposal
    print(acceptance_ratio)
    if (runif(1) < min(1, acceptance_ratio)) {
      gamma_current <- gamma_proposed
      lambda_current <- lambda_proposed
      loss_likelihood_current <- loss_likelihood_proposed
    }
    # Save samples
    gamma_samples[i] <- gamma_current#
    lambda_samples[i] <- lambda_current#
    cat(sprintf("Iteración %d: gamma = %.6f, lambda = %.6f\n", i, gamma_samples[i], lambda_samples[i]))#
   }
  return(data.frame(gamma = gamma_samples, lambda = lambda_samples))#
}


###############################################################
#                       MH - Algorithm 2                      #
###############################################################

n_iter<-10000
gamma0<-gamma_MLE
lambda0<-lambda_MLE
mcmc_samples_test <- mcmc_metropolis_lognorm (second_moment_estimator, gamma0, lambda0, n_iter)#

# --- Data set
gamma_vals <- mcmc_samples_test$gamma[1:10000]
lambda_vals <- mcmc_samples_test$lambda[1:10000]
#---Quantile
quan_gamma <- quantile(gamma_vals, probs = c(0.025, 0.975))
quan_lambda <- quantile(lambda_vals, probs = c(0.025, 0.975))

###############################################################
#                     Visualization of Results                #
###############################################################
   
# Traza de gamma
trace_gamma <- ggplot(data.frame(iter = 1:10000, value = gamma_vals), aes(x = iter, y = value)) +
  geom_line(color = "#0072B2", size = 0.7) +
  labs(title = expression("Simulation of " * gamma),
       x = "Iteration", y = expression(gamma)) +
  theme_minimal(base_size = 14)
# Traza de lambda
trace_lambda <- ggplot(data.frame(iter = 1:10000, value = lambda_vals), aes(x = iter, y = value)) +
  geom_line(color = "#009E73", size = 0.7) +
  labs(title = expression("Simulation of " * lambda),
       x = "Iteration", y = expression(lambda)) +
  theme_minimal(base_size = 14)
# Histograma de gamma
hist_gamma <- ggplot(data.frame(value = gamma_vals), aes(x = value)) +
  geom_histogram(fill = "#0072B2", alpha = 0.4, bins = 50, color = "white") +
  geom_vline(xintercept = 2.4, color = "blue", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = gamma_MLE, color = "red", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = quan_gamma, color = "gold", linetype = "dotted", size = 1.2) +
  geom_vline(xintercept = mean_gamma, color = "brown", linetype = "dashed", size = 1.2) +
  labs(title = expression("Histogram of " * gamma), x = expression(gamma), y = "Frequency") +
  theme_minimal(base_size = 14)
# Histograma de lambda
hist_lambda <- ggplot(data.frame(value = lambda_vals), aes(x = value)) +
  geom_histogram(fill = "#009E73", alpha = 0.4, bins = 50, color = "white") +
  geom_vline(xintercept = 1.7, color = "blue", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = lambda_MLE, color = "red", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = quan_lambda, color = "gold", linetype = "dotted", size = 1.2) +
  geom_vline(xintercept = mean_lambda, color = "brown", linetype = "dashed", size = 1.2) +
  labs(title = expression("Histogram of " * lambda), x = expression(lambda), y = "Frequency") +
  theme_minimal(base_size = 14)
# Leyenda personalizada como gráfico horizontal y centrado#
legend_plot <- ggplot() +
  # Línea azul (True value)
  annotate("segment", x = 1, xend = 1.2, y = 1, yend = 1, colour = "blue", linetype = "dashed", size = 1) +
  annotate("text", x = 1.25, y = 1, label = "True value", color = "blue", hjust = 0, size = 5) +
  # Línea roja (Maximum E2)
  annotate("segment", x = 2, xend = 2.2, y = 1, yend = 1, colour = "red", linetype = "dashed", size = 1) +
  annotate("text", x = 2.25, y = 1, label = expression("Maximum of " * E[2](lambda,gamma)), 
           color = "red", hjust = 0, size = 5) +
  # Línea dorada (95% CI)
  annotate("segment", x = 3.4, xend = 3.6, y = 1, yend = 1, colour = "gold", linetype = "dotted", size = 1.2) +
  annotate("text", x = 3.65, y = 1, label = "95% CI", color = "goldenrod", hjust = 0, size = 5) +
  # Línea café (Mean of Simulated Data)
  annotate("segment", x = 4.8, xend = 5.0, y = 1, yend = 1, colour = "brown", linetype = "dashed", size = 1.2) +
  annotate("text", x = 5.05, y = 1, label = "Mean of Simulated Data", color = "brown", hjust = 0, size = 5) +
  xlim(0.8, 5.8) + ylim(0.9, 1.1) +
  theme_void()
#----Graph-----
combo_plot <- wrap_plots(
  trace_gamma + trace_lambda,
  hist_gamma + hist_lambda,
  legend_plot,
  ncol = 1,
  heights = c(2, 2, 0.7)
) +
  plot_annotation(
    title = "Simulation with MH",
    theme = theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5))
  )
#--Plot Graph-----
print(combo_plot)





#######################################################################
#                                                                     #
#     Inference of the parameters γ and λ: Algorithm 1                #
#                                                                     #
#######################################################################


# ====== -log of approximated likelihood function ====== #

           
neg_log_likelihood<-function(gamma,lambda){
with_no_grad({
log_lk<-sum(log(as.numeric(Aproximation_density_red_2(
       torch_tensor(X, dtype = torch_float32()),  # Estado siguiente
      torch_tensor(seq(-M * h, M * h, by = h), dtype = torch_float32()),  # u
      n,  # Número de muestras
       t_a[1, , drop = FALSE],  # Tiempo actual
      delta_t_a[1, , drop = FALSE],  # Paso temporal
      x_t_a[1, , drop = FALSE],  # Estado actual
      M,  
      h, 
      f,  
     g,   
     gamma,
     lambda
    ))))
  })      
  return(-log_lk)     
}


# ====== MCMC with approximated likelihood function ====== #


mcmc_metropolis_lognorm_like <- function(neg_log_likelihood, gamma0, lambda0, n_iter , sigma_gamma = 0.01, sigma_lambda = 0.05) {
  # Inicializar vectores para almacenar muestras
  gamma_samples <- numeric(n_iter)
  lambda_samples <- numeric(n_iter)
  # Valores iniciales
  gamma_current <- gamma0
  lambda_current <- lambda0
  # Evaluación inicial de la verosimilitud
  loss_likelihood_current <- -neg_log_likelihood(gamma_current, lambda_current)
  for (i in 1:n_iter) {
    # Propuestas usando distribución Log-Normal
    gamma_proposed <- rlnorm(1, mean = log(gamma_current)-(sigma_gamma^2)/2, sd = sigma_gamma)
    lambda_proposed <- rlnorm(1, mean = log(lambda_current)-(sigma_lambda^2)/2, sd = sigma_lambda)
    q_gamma_forward <- dlnorm(gamma_proposed, meanlog = log(gamma_current)-(sigma_gamma^2)/2, sdlog = sigma_gamma)
    q_gamma_reverse <- dlnorm(gamma_current, meanlog = log(gamma_proposed)-(sigma_gamma^2)/2, sdlog = sigma_gamma)
    q_lambda_forward <- dlnorm(lambda_proposed, meanlog = log(lambda_current)-(sigma_lambda^2)/2, sdlog = sigma_lambda)
    q_lambda_reverse <- dlnorm(lambda_current, meanlog = log(lambda_proposed)-(sigma_lambda^2)/2, sdlog = sigma_lambda)
    # Evaluación de la nueva verosimilitud
    loss_likelihood_proposed <- -neg_log_likelihood(gamma_proposed, lambda_proposed)
    # Calcular la razón de aceptación con corrección de densidad
   acceptance_ratio <-exp(loss_likelihood_proposed-loss_likelihood_current)*((q_gamma_reverse*q_lambda_reverse)/(q_gamma_forward*q_lambda_forward))
   # Aceptar o rechazar la propuesta
    print(acceptance_ratio)
    if (runif(1) < min(1, acceptance_ratio)) {
      gamma_current <- gamma_proposed
      lambda_current <- lambda_proposed
      loss_likelihood_current <- loss_likelihood_proposed
    }
    # Guardar muestras
    gamma_samples[i] <- gamma_current
    lambda_samples[i] <- lambda_current
  cat(sprintf("Iteración %d: gamma = %.6f, lambda = %.6f\n", i, gamma_samples[i], lambda_samples[i]))
  }
  return(data.frame(gamma = gamma_samples, lambda = lambda_samples))
}

###############################################################
#                       MH - Algorithm 1                      #
###############################################################

#------Initial value for the MH algorithm------

gamma0<-2.56
lambda0<-1.8



n_iter<-1000#
set.seed(30)
mcmc_samples_test_like <- mcmc_metropolis_lognorm_like (neg_log_likelihood, gamma0, lambda0, n_iter)


# --- Data set-----
gamma_vals_like_2 <- mcmc_samples_test_like$gamma[1:1000]
lambda_vals_like_2 <- mcmc_samples_test_like$lambda[1:1000]
#---Quantile
quan_gamma_like <- quantile(gamma_vals_like_2, probs = c(0.025, 0.975))
quan_lambda_like <- quantile(lambda_vals_like_2, probs = c(0.025, 0.975))

###############################################################
#                     Visualization of Results                #
###############################################################
# Traza de gamma
trace_gamma_like <- ggplot(data.frame(iter = 1:1000, value = gamma_vals_like_2), aes(x = iter, y = value)) +
  geom_line(color = "#0072B2", size = 0.7) +
  labs(title = expression("Simulation of " * gamma),
       x = "Iteration", y = expression(gamma)) +
  theme_minimal(base_size = 14)
# Traza de lambda
trace_lambda_like <- ggplot(data.frame(iter = 1:1000, value = lambda_vals_like_2), aes(x = iter, y = value)) +
  geom_line(color = "#009E73", size = 0.7) +
  labs(title = expression("Simulation of " * lambda),
       x = "Iteration", y = expression(lambda)) +
  theme_minimal(base_size = 14)
# Histograma de gamma
hist_gamma_like <- ggplot(data.frame(value = gamma_vals_like_2), aes(x = value)) +
  geom_histogram(fill = "#0072B2", alpha = 0.4, bins = 50, color = "white") +
  geom_vline(xintercept = gamma, color = "blue", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = quan_gamma_like, color = "gold", linetype = "dotted", size = 1.2) +
  geom_vline(xintercept = gamma_mean_like, color = "brown", linetype = "dashed", size = 1.2) +
  labs(title = expression("Histogram of " * gamma), x = expression(gamma), y = "Frequency") +
  theme_minimal(base_size = 14)
# Histograma de lambda
hist_lambda_like <- ggplot(data.frame(value = lambda_vals_like_2), aes(x = value)) +
  geom_histogram(fill = "#009E73", alpha = 0.4, bins = 50, color = "white") +
  geom_vline(xintercept = lambda, color = "blue", linetype = "dashed", size = 1.1) +
  geom_vline(xintercept = quan_lambda_like, color = "gold", linetype = "dotted", size = 1.2) +
  geom_vline(xintercept = lambda_mean_like, color = "brown", linetype = "dashed", size = 1.2) +
  labs(title = expression("Histogram of " * lambda), x = expression(lambda), y = "Frequency") +
  theme_minimal(base_size = 14)
# Leyenda como gráfico
legend_plot_like <- ggplot() +
  annotate("segment", x = 1, xend = 1.2, y = 1, yend = 1, colour = "blue", linetype = "dashed", size = 1) +
  annotate("text", x = 1.25, y = 1, label = "True value", color = "blue", hjust = 0, size = 5) +
  annotate("segment", x = 2.4, xend = 2.6, y = 1, yend = 1, colour = "gold", linetype = "dotted", size = 1.2) +
  annotate("text", x = 2.65, y = 1, label = "95% CI", color = "goldenrod", hjust = 0, size = 5) +
  annotate("segment", x = 3.8, xend = 4.0, y = 1, yend = 1, colour = "brown", linetype = "dashed", size = 1.2) +
  annotate("text", x = 4.05, y = 1, label = "Mean of Simulated Data", color = "brown", hjust = 0, size = 5) +
  xlim(0.8, 5.2) + ylim(0.9, 1.1) +
  theme_void()
# Combinar en un solo gráfico
combo_plot_like <- wrap_plots(
  trace_gamma_like + trace_lambda_like,
  hist_gamma_like + hist_lambda_like,
  legend_plot_like,
  ncol = 1,
  heights = c(2, 2, 0.7)
) +
  plot_annotation(
    title = "Simulation with Likelihood",
    theme = theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5))
  )
#-----Plot Graph
print(combo_plot_like)

