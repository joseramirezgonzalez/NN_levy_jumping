###############################################################
#                                                             #
#  ------ Code used for inference via neural networks ------  #
#  ------       when  additive jumps are present     ---------#
#                                                             #
###############################################################

#----Torch library
library(torch)  # Load the Torch library for tensor computations and neural networks
library(MASS) 
###############################################################
#                                                             #
#       ------ Drift and Diffusion coefficients ------        #
#                                                             #
###############################################################


f <- function(x) { return(0.28*(x-x^3)) }  # Drift coefficient function
g <- function(x) { return(1) }     # Diffusion coefficient function
d_g <- function(x) { return(0) }     # Derivative of the diffusion coefficient with respect to 



#----Laplace----

# Simulación de n variables Laplace(μ, b)
rLaplace <- function(n, mu, b) {
  u <- runif(n, -0.5, 0.5)
  mu - b * sign(u) * log(1 - 2 * abs(u))
}




###############################################################
#                                                             #
#          ------ Simulation trajectories ------              #
#                                                             #
###############################################################


T <- 5   #----Final time T
tam <- 200*T - 1  #---Partition the interval [0, T] into N = tam subintervals
Delta <- T / tam  #----Each subinterval has size Delta

set.seed(9)  #---Set a random seed for reproducibility

tiempos <- seq(0, T, by = Delta)  #--Time vector (t_1, ..., t_N)

K <- 10    #----Number of simulated trajectories

x <- c()   #-----Container to store all simulated trajectories




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
      simula_z <- rnorm(sim_pois_sample[i], 0, sqrt(0.12))     # Simulation of the jumps z_i U(-0.1,0.1)
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



#-----Parameters
gamma<-0.31
lambda<-1.7


#-----Simulations-----
for(j in 1:K){
 x[1+length(tiempos)*(j-1)] <- 1.5    #---Initial point
 for (i in 2:length(tiempos)) {
 	f_X_std<-f(x[(i-1)+length(tiempos)*(j-1)])/(1+(Delta)*f(x[(i-1)+length(tiempos)*(j-1)])^2)
   x[i+length(tiempos)*(j-1)]<-simulate_sample(1,Delta, t, x[(i-1)+length(tiempos)*(j-1)], g, d_g, f_X_std, gamma,lambda)#
  }
}


times <- rep(tiempos, K)  #---Store the time points associated with each simulation---

# Convert to tensors
x_real_red_tensor <- torch_tensor(matrix(x, ncol = 1), dtype = torch_float())     # Simulated trajectories as a float tensor
tiempo_tensor <- torch_tensor(matrix(times, ncol = 1), dtype = torch_float())     # Corresponding time points as a float tensor


#------Load required packages for graphs of simulations---
library(ggplot2)
library(latex2exp)  # Allows LaTeX expressions in plot labels
library(gridExtra)
# Check that the data exists and is not empty
if (!exists("times") || !exists("x") || length(times) == 0 || length(x) == 0) {
  stop("Error: 'times' or 'x' are not properly defined or are empty.")
}

# Create a data frame with the original data
df <- data.frame(
  t = times,
  X_t = x,
  simulation = factor(rep(1:10, each = 1000), labels = paste("Simulation", 1:10))  # Properly labeled factor variable
)

#----Create the plot with colored lines for each simulation and include LaTeX-formatted equations
ggplot(df, aes(x = t, y = X_t, color = simulation)) +
  geom_line(size = 0.3) +  
  geom_point(shape = 4, size = 1) +  
  scale_color_manual(values = rainbow(10)) +  
  annotate("text", x = max(df$t) - 1, y = max(df$X_t), 
           label = expression(
             X[t+Delta*t] == X[t] + f^m*(X[t]) * Delta*t + g(X[t]) * Delta*W[t] +
             frac(1,2) * g(X[t]) * dot(g)(X[t]) * ((Delta*W[t])^2 - Delta*t)
           ), 
           hjust = 1, vjust = 1.2, size = 5) +  
  annotate("text", x = max(df$t) - 1, y = max(df$X_t) - 0.5, 
           label = expression(f(x) == -0.25 * x^3 ~ "," ~ g(x) == 0.57*x), 
           hjust = 1, vjust = 0, size = 5) +
  labs(
    x = expression(t), 
    y = expression(X[t]),
    title = "Stochastic Process Simulation",
    color = "Simulations"
  ) +
  theme_minimal()



###############################################################
#                                                             #
#         ------ Definition of neural networks ------         #
#                                                             #
###############################################################


# Custom dataset
MyDataset <- dataset(
  initialize = function(x_real_red, tiempos) {
    self$x_real_red <- x_real_red
    self$tiempos <- tiempos
  },
  .getitem = function(index) {
    list(
      self$x_real_red[index],
      self$tiempos[index]
    )
  },
  .length = function() {
    self$x_real_red$size(1)
  }
)

# Create dataset and dataloader
batch_size <- 100
train_dataset <- MyDataset(x_real_red_tensor, tiempo_tensor)
dataloader <- dataloader(train_dataset, batch_size = batch_size, shuffle = FALSE)




#----Define neural network associated with the drift coefficient f

CombinedModel <- nn_module(
  initialize = function() {
    self$shared <- nn_sequential(
      nn_linear(1, 32),
      nn_elu()
    )
    self$f_specific <- nn_sequential(
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu()
    )
    self$f_head <- nn_linear(32, 1)
  },
  forward = function(x) {
    shared_out <- self$shared(x)
    f_out <- self$f_specific(shared_out)
    predicted_f <- self$f_head(f_out)
    return(list(predicted_f = predicted_f))
  }
)

#----Define neural network associated with the diffusion coefficient g
CombinedModel2 <- nn_module(
  initialize = function() {
    self$shared <- nn_sequential(
      nn_linear(1, 32),
      nn_elu()
    )
    self$g_specific <- nn_sequential(
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu()
    )
    self$g_head <- nn_sequential(
      nn_linear(32, 1),  
      nn_softplus()    
    )
    },
  forward = function(x) {
    shared_out <- self$shared(x)
    g_out <- self$g_specific(shared_out)
    predicted_g <- self$g_head(g_out)
    return(list(predicted_g = predicted_g))
  }
)



# Create the models and apply initialization
model <- CombinedModel()
model2 <- CombinedModel2()


#---Define the optimization method
optimizer <- optim_adam(model$parameters, lr = 0.001)
optimizer2 <- optim_adam(model2$parameters, lr = 0.001)



#------Coefficient approximations using the neural network models-----
predicted_f<-function(x) {model(x)$predicted_f}
predicted_g<-function(x) {model2(x)$predicted_g}


###############################################################
#                                                             #
# ------ Functions to construct the approximate density ----- #
#  ------   of X_{t+Δt} given \mathcal{F}(X_t)          ------#
#                                                             #
###############################################################


#------Construction of the density function via Yang and the characteristic function of X_{t+Δt} given \mathcal{F}(X_t)

density_function <- function(u, sample, M, h) {
  density <- torch_zeros(u$size(1), dtype = torch_cfloat())
  pi_torch <- torch_tensor(pi, dtype = torch_float())  
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float()), torch_tensor(1, dtype = torch_float()))  
  for (i in seq_len(u$size(1))) {
    aux <- torch_zeros(1, dtype = torch_cfloat())  
    for (j in seq(-M, M)) {
      idx <- j + M + 1  
      term_exp <- torch_exp(-j * h * u[i] * one_i)
      aux <- aux + term_exp * sample[idx]*h
    }
    density[i] <- aux / (2 * pi_torch)
  }
  return(density)  
}

# Get epsilon for maximum precision (float64 in PyTorch)
epsilon <- torch_finfo(torch_float64())$eps






#----Approximation parameters for the density
M<-200
h<-0.05
n<-200
#------------------------



###############################################################
#                                                             #
#        ------ Definition of loss functions ------           #      
#                                                             #
###############################################################






#------Loss function D_2(\hat{f}, \hat{g}, B_k,j)---------
custom_combined_loss <- function(predicted_f, predicted_g, n,t,X_t, M, h,lambda, gamma) {
  density_loss <- torch_zeros(X_t$size(1)-1, dtype = torch_float()) 
  for (i in 1:(X_t$size(1)-1)) {
  	density_loss[i]<-1
    X_t_2<-X_t[i, ,drop = FALSE]$clone()
    f_X_t<-predicted_f(X_t_2)
    g_X_t <- predicted_g(X_t_2)
    Delta<-t[i+1,  ,drop = FALSE]-t[i, ,drop = FALSE]
    deriva<-X_t[i+1,  ,drop = FALSE]- X_t[i,  ,drop = FALSE]-f_X_t*Delta
    if(g_X_t$item()!=0){
    density_loss[i]<- Aproximation_density_red_2(
      X_t[i+1,  ,drop = FALSE],
      torch_tensor(seq(-M * h, M * h, by = h), dtype = torch_float()),  # u
     n,
      t[i,  ,drop = FALSE],
      t[i+1,  ,drop = FALSE]-t[i, ,drop = FALSE],
      X_t[i,  ,drop = FALSE],
       M,
       h,
      predicted_f,
      predicted_g,
      gamma,
     lambda
    )
    }
    else{
    	if(deriva$item()!=0){
    	  	density_loss[i]<-0  	
        	}
    	}
    }
  mse_loss <- -torch_sum(torch_log(density_loss)) # Error medio
  return(mse_loss)
}



 



#-----------Loss function L_1(\hat{f}, \hat{g}, B_j, j)---------
loss_function_f_2 <- function(predicted_f,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, ,drop = FALSE] - tiempos[i - 1, ,drop = FALSE]
      X_next<-x_real[i, ,drop = FALSE]$clone()
      X_t<-x_real[i-1, ,drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      predicted_X[i-1] <- X_next-X_t-(f_X_t / (1 + Delta * torch_square(f_X_t)))  * Delta
   }  
 	 mse_loss <- torch_abs(torch_mean(predicted_X)) 
  return(mse_loss)
}
#-----------------------------




#-----------Loss function L_2(\hat{f}, \hat{g}, B_j, j)---------
loss_function_g_second_2<- function(predicted_f,predicted_g,tiempos, x_real,gamma, lambda,n_1 ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-6,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, ,drop = FALSE] - tiempos[i - 1, ,drop = FALSE]
      X_t<-x_real[i-1, ,drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- torch_square(x_real[i, ,drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
       #predicted_X_2[i-1] <- 0.5*torch_square(Delta*d_g_X_t*g_X_t)+torch_square(g_X_t)*Delta
       predicted_X_2[i-1] <- second_moment_estimator(
       x = X_t,
       t = tiempos[i - 1, ,drop = FALSE],
       Delta = Delta,
       f = predicted_f,
       g = predicted_g,
       gamma = gamma,
       lambda = lambda,
       n_1 = n_1)
      }
  	 mse_loss <- torch_mean(torch_square(predicted_X-predicted_X_2))  
  return(mse_loss)
}
#-----------------------












#-----------------------H(\hat{f}, \hat{g}, B_j, j)---------
loss_function_g <- function(predicted_f,predicted_g,tiempos, x_real,gamma,lambda,n_1 ) {
	index<-c()
	aux <- torch_zeros(1, dtype = torch_float()) 
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
# predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-4,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, ,drop = FALSE] - tiempos[i - 1, ,drop = FALSE]
      X_t<-x_real[i-1, ,drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      aux <- (x_real[i, ,drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
       predicted_X[i-1]<-aux
      if(g_X_t$item()!=0){
      index<-c(index,i-1)	
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- aux/torch_sqrt(second_moment_estimator(
       x = X_t,
       t = tiempos[i - 1, ,drop = FALSE],
       Delta = Delta,
       f = predicted_f,
       g = predicted_g,
       gamma = gamma,
       lambda = lambda,
       n_1 = n_1))
    }
 }  
 if((length(index)>0)&&(length(index)<(tiempos$size(1)-1))){
 mse_loss <- torch_mean(torch_square(predicted_X[index]))+torch_mean(torch_square(predicted_X[-index]))  # Error medio
  return(mse_loss)
  }
 else{
 	mse_loss <- torch_mean(torch_square(predicted_X)) # Error medio
  return(mse_loss)
 } 
}

#-----------------------




#-----------Loss function L_3(\hat{f}, \hat{g}, B_j, j)---------
loss_function_f <- function(predicted_f,predicted_g,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, ,drop = FALSE] - tiempos[i - 1, ,drop = FALSE]
      X_next<-x_real[i, ,drop = FALSE]$clone()
      X_t<-x_real[i-1, ,drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      predicted_X[i-1] <- X_next-X_t-(f_X_t / (1 + Delta * torch_square(f_X_t)))  * Delta
     if(g_X_t$item()!=0){
        index<-c(index,i-1)     	
      }
    }
    if((length(index)>0)&&(length(index)<(tiempos$size(1)-1))){
 mse_loss <- torch_mean(torch_square(predicted_X[index]))+torch_mean(torch_square(predicted_X[-index])) 
  return(mse_loss)
  }
 else{
 	 mse_loss <- torch_mean(torch_square(predicted_X))  # Error medio
     return(mse_loss)
  } 
 }
#-----------------------





#-----------------------L_4(\hat{f}, \hat{g}, B_j, j)---------
loss_function_g_2 <- function(predicted_f,predicted_g,tiempos, x_real,gamma,lambda,n_1 ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-4,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, ,drop = FALSE] - tiempos[i - 1, ,drop = FALSE]
      X_t<-x_real[i-1, ,drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- torch_square(x_real[i, ,drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
      if(g_X_t$item()!=0){ 
         predicted_X_2[i-1] <- second_moment_estimator(
       x = X_t,
       t = tiempos[i - 1, ,drop = FALSE],
       Delta = Delta,
       f = predicted_f,
       g = predicted_g,
       gamma = gamma,
       lambda = lambda,
       n_1 = n_1)
         index<-c(index,i-1)	
       }
    }
  if((length(index)>0)&&(length(index)<(tiempos$size(1)-1))){
  mse_loss <- torch_mean(torch_square(predicted_X[index]-predicted_X_2[index]))+torch_mean(torch_square(predicted_X[-index]))  # Error medio
  return(mse_loss)
   }
  else{
 	 mse_loss <- torch_mean(torch_square(predicted_X-predicted_X_2))  # Error medio
    return(mse_loss)
   } 
 }
#-----------------------










#-------second_moment_estimator-----------

second_moment_estimator <- function(x, t, Delta, f, g, gamma, lambda, n_1) {
  # Convertir escalares a tensores Torch
  x <- x$clone()
  gamma <- torch_tensor(gamma, dtype = torch_float32())

  mu_2 <- 0.12

  # Evaluaciones de funciones Torch
  g_X_t <- g(x)

  # Aproximación centrada de la derivada de g usando h_next
  h_next <- torch_tensor(1e-6, dtype = torch_float32())
  d_g_X_t <- (g(x + h_next) - g(x - h_next)) / (2 * h_next)

  f_X <- f(x)
  f_X_std <- f_X / (1 + Delta * f_X^2)

  constant_term <- Delta * (g_X_t^2) + 0.5 * (g_X_t * d_g_X_t * Delta)^2 +
                   (gamma^2) * mu_2 * lambda * Delta
  Delta<-torch_squeeze(Delta, dim = 1)
  t<-torch_squeeze(t, dim = 1)
  sim_pois_sample <- rpois(n_1, lambda * as.numeric(Delta))
  times_sample <- vector("list", n_1)
  for (i in seq_len(n_1)) {
    if (sim_pois_sample[i] > 0) {
      times_sample[[i]] <- sort(runif(sim_pois_sample[i], as.numeric(t), as.numeric(t) + as.numeric(Delta)))
    } else {
      times_sample[[i]] <- numeric(0)
    }
  }

  sample_sim <- torch_zeros(n_1, dtype = torch_float32())

  for (i in seq_len(n_1)) {
    if (sim_pois_sample[i] == 0) {
      sample_sim[i] <- constant_term
    } else {
     # Simulación de saltos
    simula_z <- rnorm(sim_pois_sample[i], 0, sqrt(0.12))
    z_tensor <- torch_tensor(simula_z, dtype = torch_float32())

    # Inicializar g_diff
    g_diff <- torch_zeros(sim_pois_sample[i], dtype = torch_float32())

    for (j in 1:sim_pois_sample[i]) {
      shifted_x <- x + gamma * z_tensor[j]
      g_diff[j] <- g(shifted_x) - g_X_t
     }
      
      times_i_tensor <- torch_tensor(times_sample[[i]], dtype = torch_float32())
     # times_concat_1 <- torch_cat(list(times_i_tensor, t + Delta), dim = 1)  # c(times[[i]], t + Delta_t)
      
      #times_concat_2 <- torch_cat(list(t, times_i_tensor), dim = 1)  # c(t, times[[i]])
     
      diff_times <- t+Delta-times_i_tensor      

      aux_2 <- 2 * g_X_t * torch_sum(g_diff * diff_times)

      
 
      
      if(sim_pois_sample[i]>1){
      A <- torch_outer(torch_squeeze(g_diff, dim = 1), torch_squeeze(g_diff, dim = 1))
 
      
      
      idx <- matrix(pmax(rep(1:sim_pois_sample[i], each = sim_pois_sample[i]), rep(1:sim_pois_sample[i], sim_pois_sample[i])), nrow = sim_pois_sample[i])
     
      
      # Inicializar matriz B vacía
      B <- torch_zeros(c(sim_pois_sample[i], sim_pois_sample[i]), dtype = torch_float32())

     for (i_1 in 1:sim_pois_sample[i]) {
       for (i_2 in 1:sim_pois_sample[i]) {
         idx_val <- max(i_1, i_2)  # pmax manual
         B[i_1, i_2] <- diff_times[idx_val]
      }
    }
      
      aux_3 <- torch_sum(A * B)
      sample_sim[i] <- constant_term + aux_2 + aux_3
      }
      
      else{
      	A <- g_diff*g_diff
      aux_3 <- torch_sum(A * diff_times)
      sample_sim[i] <- constant_term + aux_2 + aux_3
      	
      	
      }

    }
  }

  mean_sim <- torch_mean(sample_sim)
  return(mean_sim)
}






#------Coefficient approximations using the neural network models-----
predicted_f<-function(x) {model(x)$predicted_f}
predicted_g<-function(x) {model2(x)$predicted_g}







#-------Characteristic function of X_{t+Δt} given \mathcal{F}(X_t) with jumps-----
Aproximation_density_red_2 <- function(X_t_1, u, n, t, Delta_t, X_t, M, h, predicted_f, predicted_g,gamma,lambda) {
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float32()), torch_tensor(1, dtype = torch_float32())) 
  h_aprox<-torch_tensor(10^-6,dtype = torch_float())
   X_t_2<-X_t$clone()
  two_i <- 2 * one_i  # 2i como tensor complejo
  gamma <- torch_tensor(gamma, dtype = torch_float32())
  predic_g_X <- predicted_g(X_t_2)
  predic_f_X <- predicted_f(X_t_2)
   predic_f_X_s<-predic_f_X/(1+Delta_t*predic_f_X^2)
  d_g_X_t<-(predicted_g(X_t_2+h_aprox)-predicted_g(X_t_2-h_aprox))/(2*h_aprox)
  c <- 0.5 * predic_g_X * d_g_X_t
  X_shift <- X_t + (predic_f_X_s - c) * Delta_t
  Delta_t<-torch_squeeze(Delta_t, dim = 1)
  t<-torch_squeeze(t, dim = 1)
  sim_pois <- rpois(n, lambda * as.numeric(Delta_t))
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
    	  simula_z<-torch_tensor(rnorm(sim_pois[i],0,sqrt(0.12)), dtype = torch_float32()) # Simulation of the jumps
    	  
    	  
    values_subtracted <- torch_zeros(sim_pois[i], dtype = torch_float32())

    for (j in 1:sim_pois[i]) {
      shifted_x <- X_t_2 + gamma * simula_z[j]
      values_subtracted[j] <- g(shifted_x) - predic_g_X
     }	  
    	  
    	  
    	  
    	  
    	  
    #  values <-torch_tensor(predicted_g(X_t_2 + gamma * simula_z), dtype = torch_float32())
     # values_subtracted <- values - predic_g_X
      g_accumulated <- torch_cumsum(values_subtracted, dim = 1)
      a <- torch_cat(list(torch_zeros(1, dtype = torch_float32()), g_accumulated))+predic_g_X
      if(sim_pois[i]>1){print("El valor de a es:")
      	print(a)}
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
#    ------ Training methodology Ramirez–Sun ------           #      
#                                                             #
###############################################################


#----Set the networks to training mode
model$train()
model2$train()

#---Define the number of epochs for the method---
epochs_1 <- 100   
epochs_2 <- 400    
epochs_f_g<-10 
n_1<-400

#-------Phase 1:

for (epoch in 1:epochs_f_g) {
	i<-0
	coro::loop(for (batch in dataloader) {
		i<-i+1
    optimizer$zero_grad()
    optimizer2$zero_grad()
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    order_indices <- order(as_array(batch_tiempos)) 
    batch_tiempos <- batch_tiempos[order_indices] 
    batch_x_real <- batch_x_real[order_indices]
    loss <- custom_combined_loss(function(x) predicted_f(x),function(x) predicted_g(x),n, batch_tiempos,batch_x_real, M, h,lambda,gamma)
    cat(sprintf("Epoch %d/%d,%d, Batch Loss for f: %f\n", epoch, epochs_f_g, i,loss$item()))
    loss$backward()
    optimizer$step()
    optimizer2$step()
     })
}




#----Reset the neural network associated with the diffusion coefficient g for phase 2----

# Define the model
CombinedModel2 <- nn_module(
  initialize = function() {
    self$shared <- nn_sequential(
      nn_linear(1, 32),
      nn_elu()
    )
    self$g_specific <- nn_sequential(
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu(),
      nn_linear(32, 32),
      nn_elu()
    )
    self$g_head <- nn_sequential(
      nn_linear(32, 1),  
      nn_softplus()      
    )
    },
  forward = function(x) {
    shared_out <- self$shared(x)
    g_out <- self$g_specific(shared_out)
    predicted_g <- self$g_head(g_out)
    return(list(predicted_g = predicted_g))
  }
)

# Create the model 
model2 <- CombinedModel2()
optimizer2 <- optim_adam(model2$parameters, lr = 0.001)
predicted_g<-function(x) {model2(x)$predicted_g}
model2$train()


#-------Phase 2 and Phase 3:
 
batch_list <- list()
coro::loop(for (batch in dataloader) {
       batch_list <- append(batch_list, list(batch)) 
 })
num_batches <- length(batch_list)
batches_por_trayectoria <- num_batches / K  

train_f<-4 #Number train_f for Phase 3


for (epoch in 1:(epochs_1+epochs_2)) {
if(epoch<=epochs_1) {    #---Phase 2
 	selected_indices_f <- c()
 	batch_losses_f <- c()
 	selected_indices_f_2 <- c()
 	batch_losses_f_2 <- c()
    index_K<- sample(1:K,batches_por_trayectoria, replace = TRUE)      #---Random variables U_1
 	for(l in 1:batches_por_trayectoria){
 	  batch<-batch_list[[(index_K[l] - 1) * batches_por_trayectoria + l]]
      batch_x_real<-batch[[1]]
      batch_tiempos<-batch[[2]]
      with_no_grad({
      loss <- loss_function_f_2(  #----L_1 function
      function(x) predicted_f(x),
      batch_tiempos, batch_x_real)
      })
     batch_losses_f <- c(batch_losses_f, loss$item())
    }
   inicio <-  1
   fin <-  batches_por_trayectoria
   cont<-0
   aux<-1
  # Extract losses from the current trajectory
   batch_losses_f_traj <- batch_losses_f[inicio:fin]
   for(r in 1:batches_por_trayectoria){
   	if(batch_losses_f_traj[r]!=0){
   		cont<-cont+1
   	}
   	if(batch_losses_f_traj[r]==Inf){
   		aux<-0
   	}
   }
   top_k <- max(1, round(0.8 * batches_por_trayectoria))     #-----R_1
   if((aux==1)&&(cont>=8)){
   # Normalize within the trajectory
   batch_losses_f_traj <- batch_losses_f_traj / sum(batch_losses_f_traj)
   # Select 80% of the batches from this trajectory
   selected_indices_f_traj <- sample(inicio:fin, top_k, replace = FALSE, prob = batch_losses_f_traj) #---Random variables w_1
   # Save the selected indices
   selected_indices_f <- c(selected_indices_f, selected_indices_f_traj)
   }
    else{
   	selected_indices_f <- sample(inicio:fin, top_k, replace = FALSE)
   	}
   for (selected_idx in selected_indices_f) {
   batch <- batch_list[[(index_K[l]-1)*batches_por_trayectoria+selected_idx]]
   #-----Optimize with respect to the coefficient f-----
    optimizer$zero_grad() 
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
   loss <- loss_function_f_2(    #----L_1 function
       function(x) predicted_f(x),
       batch_tiempos, batch_x_real)
      cat(sprintf("Epoch %d/%d, Batch Loss for f: %f\n", epoch, epochs_1+epochs_2, loss$item()))
    loss$backward()
    optimizer$step() 
    }
  
   index_K<- sample(1:K,batches_por_trayectoria, replace = TRUE)    #---Random variables U_2
   for(l in 1:batches_por_trayectoria){
 	batch<-batch_list[[(index_K[l] - 1) * batches_por_trayectoria + l]]
    batch_x_real<-batch[[1]]
    batch_tiempos<-batch[[2]]
    with_no_grad({
   loss <-loss_function_g_second_2(     #----L_2 function
       function(x) predicted_f(x),function(x) predicted_g(x),
       batch_tiempos, batch_x_real,gamma, lambda,n_1) 
    })
    batch_losses_f_2 <- c(batch_losses_f_2, loss$item()) 
    }
    cont<-0
    aux<-1
   # Extract losses from the current trajectory
   batch_losses_f_traj <- batch_losses_f_2[inicio:fin]
   for(r in 1:batches_por_trayectoria){
   	if(batch_losses_f_traj[r]!=0){
   		cont<-cont+1
   	}
   	if(batch_losses_f_traj[r]==Inf){
   		aux<-0
   		}
   }
   top_k_2 <- max(1, round(0.2 * batches_por_trayectoria))    #-----R_2
  if((aux==1)&&(cont>=2)){
    # Select 20% of the batches from this trajectory
    # Normalize within the trajectory
    batch_losses_f_traj <- batch_losses_f_traj / sum(batch_losses_f_traj)
    selected_indices_f_traj <- sample(inicio:fin, top_k_2, replace = FALSE, prob = batch_losses_f_traj) #---Random variables w_2
    # Save the selected indices
    selected_indices_f_2 <- c(selected_indices_f_2, selected_indices_f_traj)
   }
  else{
   	selected_indices_f_2 <- sample(inicio:fin, top_k_2, replace = FALSE)
   	}
  for (selected_idx in selected_indices_f_2) {
    batch <- batch_list[[(index_K[l]-1)*batches_por_trayectoria+selected_idx]]
    #-----Optimize with respect to the coefficient g-----
    optimizer2$zero_grad() 
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    loss <-loss_function_g_second_2(      #----L_2 function
       function(x) predicted_f(x),function(x) predicted_g(x),
       batch_tiempos, batch_x_real,gamma, lambda,n_1)
    cat(sprintf("Epoch %d/%d, Batch Loss for g: %f\n", epoch, epochs_1+epochs_2, loss$item()))
    loss$backward()
    optimizer2$step()
    }
}   #---End Phase 2
else{       #------Phase 3
	for(i in 1: train_f){
	selected_indices_f <- c()
 	batch_losses_f <- c()
 	selected_indices_f_2 <- c()
 	batch_losses_f_2 <- c()
    index_K<- sample(1:K,batches_por_trayectoria, replace = TRUE)     #---Random variables U_3
 	for(l in 1:batches_por_trayectoria){
 	batch<-batch_list[[(index_K[l] - 1) * batches_por_trayectoria + l]]
    batch_x_real<-batch[[1]]
    batch_tiempos<-batch[[2]]
    with_no_grad({
    loss <- loss_function_g(    #---H function
      function(x) predicted_f(x),
      function(x) predicted_g(x),
      batch_tiempos, batch_x_real,gamma, lambda,n_1)
    })
   batch_losses_f <- c(batch_losses_f, loss$item())
   }
   inicio <- 1
   fin <-  batches_por_trayectoria
   cont<-0
   aux<-1
   # Extract losses from the current trajectory
   batch_losses_f_traj <- batch_losses_f[inicio:fin]
   print(batch_losses_f_traj)   
   for(r in 1:batches_por_trayectoria){
   	if(batch_losses_f_traj[r]!=0){
   		cont<-cont+1
   	}
    if(batch_losses_f_traj[r]==Inf){
   	   aux<-0
   	}
   }
  top_k <- max(1, round(0.4 * batches_por_trayectoria))      #----R_3
  if((aux==1)&&(cont>=4)){
   # Normalize within the trajectory
   batch_losses_f_traj <- batch_losses_f_traj / sum(batch_losses_f_traj)
   # Select 40% of the batches from this trajectory
   selected_indices_f_traj <- sample(inicio:fin, top_k, replace = FALSE, prob = batch_losses_f_traj) #---Random variables w_3
   # Save the selected indices
   selected_indices_f <- c(selected_indices_f, selected_indices_f_traj)
   }
   else{
   	selected_indices_f <- sample(inicio:fin, top_k, replace = FALSE)
   	
   }
   for (selected_idx in selected_indices_f) {
    batch <- batch_list[[(index_K[l]-1)*batches_por_trayectoria+selected_idx]]
    #-----Optimize with respect to the coefficient f-----
    optimizer$zero_grad()
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
   loss<-loss_function_f(   #---L_3 function
       function(x) predicted_f(x),
       function(x) predicted_g(x),
       batch_tiempos, batch_x_real)
      +loss_function_g_2(      #----L_4 function
        function(x) predicted_f(x),
        function(x) predicted_g(x),
        batch_tiempos, batch_x_real,gamma, lambda,n_1)
    cat(sprintf("Epoch %d/%d, Batch Loss for f: %f\n", epoch, epochs_1+epochs_2, loss$item()))
    loss$backward()
    optimizer$step()
    
    }
 }
   index_K<- sample(1:K,batches_por_trayectoria, replace = TRUE)     #---Random variables U_4
   for(l in 1:batches_por_trayectoria){
 	batch<-batch_list[[(index_K[l] - 1) * batches_por_trayectoria + l]]
    batch_x_real<-batch[[1]]
    batch_tiempos<-batch[[2]]
   with_no_grad({
   loss <- 1/loss_function_g(       #-----1/H function
      function(x) predicted_f(x),
      function(x) predicted_g(x),
      batch_tiempos, batch_x_real,gamma, lambda,n_1)
    })
    batch_losses_f_2 <- c(batch_losses_f_2, loss$item()) 
    }
   cont<-0
   aux<-1
   # Extraer pérdidas de la trayectoria actual
   batch_losses_f_traj <- batch_losses_f_2[inicio:fin]
   print(batch_losses_f_traj) 
   for(r in 1:batches_por_trayectoria){
   	  if(batch_losses_f_traj[r]==Inf){
   		aux<-0
   		}
   	  if(batch_losses_f_traj[r]!=0){
   		cont<-cont+1
   	    }
   }
  top_k_2 <- max(1, round(0.4 * batches_por_trayectoria))      #----R_4
  if((aux==1)&&(cont>=4)){
   # Select 40% of the batches from this trajectory
   # Normalize within the trajectory
   batch_losses_f_traj <- batch_losses_f_traj / sum(batch_losses_f_traj)
   selected_indices_f_traj <- sample(inicio:fin, top_k_2, replace = FALSE, prob = batch_losses_f_traj)  #---Random variables w_4 
   # Save the selected indices
   selected_indices_f_2 <- c(selected_indices_f_2, selected_indices_f_traj)
    }
   else{
   	selected_indices_f_2 <- sample(inicio:fin, top_k_2, replace = FALSE)
   	}
   for (selected_idx in selected_indices_f_2) {
    batch <- batch_list[[(index_K[l]-1)*batches_por_trayectoria+selected_idx]]
    #-----Optimize with respect to the coefficient g-----
    optimizer2$zero_grad() 
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    loss <-loss_function_g_second_2(     #---L_2 function
       function(x) predicted_f(x),function(x) predicted_g(x),
       batch_tiempos, batch_x_real,gamma, lambda,n_1)
    cat(sprintf("Epoch %d/%d, Batch Loss for g: %f\n", epoch, epochs_1+epochs_2, loss$item()))
    loss$backward()
    optimizer2$step()
    }
   } #----End Phase 3
 }


#-----------------




#-----Set the model to evaluation mode-----
model$eval()
model2$eval()

###############################################################
#                                                             #
#  ------ Results: Comparison between the true coefficients   #
#  ------ and those estimated by the neural networks          #
#  ------ using Ramirez-Sun methodology ------                #
#                                                             #
###############################################################


#-----Graphs------
vec_1<-rep(0,tiempo_tensor$size(1))
vec_2<-rep(0,tiempo_tensor$size(1))
vec_4<-rep(0,tiempo_tensor$size(1))
vec_5<-rep(0,tiempo_tensor$size(1))

for(i in 1:tiempo_tensor$size(1)){
	vec_4[i]<-as.numeric(predicted_f(x_real_red_tensor[i,drop=FALSE]))
	vec_5[i]<-as.numeric(f(x_real_red_tensor[i,drop=FALSE]))
	vec_1[i]<-as.numeric(g(x_real_red_tensor[i,drop=FALSE]))
	vec_2[i]<-as.numeric(predicted_g(x_real_red_tensor[i,drop=FALSE]))
}

# Create the data frames with the data

df_f <- data.frame(
  X_t = as.numeric(x_real_red_tensor),
  f_real = as.numeric(vec_5),
  f_approx = as.numeric(vec_4)
)

df_g <- data.frame(
  X_t = as.numeric(x_real_red_tensor),
  g_real = as.numeric(vec_1),
  g_approx = as.numeric(vec_2)
)

# Define the Y-axis limits with some margin

y_limits_f <- range(c(df_f$f_real, df_f$f_approx)) + c(-6, 6) * 0.2 * diff(range(c(df_f$f_real, df_f$f_approx)))
y_limits_g <- range(c(df_g$g_real, df_g$g_approx)) + c(-6, 6) * 0.2 * diff(range(c(df_g$g_real, df_g$g_approx)))

# Compute the MSE

mse_f <- mean((df_f$f_approx - df_f$f_real)^2)
mse_g <- mean((df_g$g_approx - df_g$g_real)^2)

# Create plot for f(X_t)

p1 <- ggplot(df_f, aes(x = X_t)) +
  geom_point(aes(y = f_real, color = "Real function"), shape = 4, size = 2) +
  geom_point(aes(y = f_approx, color = "Approximation"), shape = 4, size = 2) +
  labs(
    title = expression(paste("Comparison of ", f(X[t]), " and ", hat(f)(X[t]))),
    x = expression(X[t]),
    y = expression(f(X[t]))
  ) +
  scale_color_manual(values = c("blue", "red"), labels = c(expression(hat(f)(X[t])), expression(f(X[t])))) +
  annotate("text", x = min(df_f$X_t), y = min(y_limits_f), 
           label = sprintf("MSE(f, %s) = %.4f", "\u0302f", mse_f), 
           hjust = 0, vjust = -1.5, size = 3.5, color = "black") +
  ylim(y_limits_f) +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")

# Create plot for g(X_t)

p2 <- ggplot(df_g, aes(x = X_t)) +
  geom_point(aes(y = g_real, color = "Real function"), shape = 4, size = 2) +
  geom_point(aes(y = g_approx, color = "Approximation"), shape = 4, size = 2) +
  labs(
    title = expression(paste("Comparison of ", g(X[t]), " and ", hat(g)(X[t]))),
    x = expression(X[t]),
    y = expression(g(X[t]))
  ) +
  scale_color_manual(values = c("blue", "red"), labels = c(expression(hat(g)(X[t])), expression(g(X[t])))) +
  annotate("text", x = min(df_g$X_t), y = max(y_limits_g), 
           label = sprintf("MSE(g, %s) = %.4f", "\u0302g", mse_g), 
           hjust = 0, vjust = 1.5, size = 3.5, color = "black") +
  ylim(y_limits_g) +
  theme_minimal() +
  theme(legend.title = element_blank(), legend.position = "bottom")

# Display both plots in a single window
grid.arrange(p1, p2, nrow = 1)



