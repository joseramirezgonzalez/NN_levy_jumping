###############################################################
#                                                             #
#  ------ Code used for inference via neural networks ------  #
#  ------       when no additive jumps are present      ------#
#                                                             #
###############################################################

#----Torch library
library(torch)  # Load the Torch library for tensor computations and neural networks

###############################################################
#                                                             #
#       ------ Drift and Diffusion coefficients ------        #
#                                                             #
###############################################################


f <- function(x) { return(-0.25*x^3) }  # Drift coefficient function
g <- function(x) { return(0.57*x) }     # Diffusion coefficient function
d_g <- function(x) { return(0.57) }     # Derivative of the diffusion coefficient with respect to x



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



#-----Simulations-----
for(j in 1:K){
 x[1+length(tiempos)*(j-1)] <- 1.5    #---Initial point
 W_diff <-rnorm(length(tiempos)-1) * sqrt(diff(tiempos))
 for (i in 2:length(tiempos)) {
   X_t<-x[i - 1+length(tiempos)*(j-1)]
   g_X_t <- g(X_t)
   f_X_t <- f(X_t)
   d_g_X_t<-d_g(X_t)
   constant_term <- 0.5 * g_X_t * d_g_X_t
   adjusted_X_t <- X_t + (f_X_t / (1 + Delta * f_X_t^2)- constant_term )*Delta
    x[i+length(tiempos)*(j-1)] <- W_diff[i - 1] * g_X_t  + adjusted_X_t+ constant_term * W_diff[i - 1]^2
  }
}


times <- rep(tiempos, K)  #---Store the time points associated with each simulation---

# Convert to tensors
x_real_red_tensor <- torch_tensor(x, dtype = torch_float())     # Simulated trajectories as a float tensor
tiempo_tensor <- torch_tensor(times, dtype = torch_float())     # Corresponding time points as a float tensor


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


#-------Characteristic function of X_{t+Δt} given \mathcal{F}(X_t) without jumps-----

Aproximation_density_red_W <- function(X_t_next, u, Delta_t, X_t, M, h, predicted_f, predicted_g) {
  one_i <- torch_complex(torch_tensor(0, dtype = torch_float()), torch_tensor(1, dtype = torch_float()))  
  two_i <- 2 * one_i  
  h_aprox<-torch_tensor(10^-6,dtype = torch_float())
  X_t_2<-X_t$clone()
  predic_g_X <- predicted_g(X_t_2)
  predic_f_X <- predicted_f(X_t_2)
   predic_f_X_2<-predic_f_X/(1+Delta*predic_f_X^2)
  deriva<- X_t+ predic_f_X_2*Delta_t
  x <- torch_tensor(X_t, requires_grad = TRUE)
  d_g_X_t<-(predicted_g(X_t+h_aprox)-predicted_g(X_t-h_aprox))/(2*h_aprox)
  c <- 0.5 * d_g_X_t *predic_g_X
  X_shift <- deriva - c * Delta_t
  phi <- torch_zeros(u$size(1), dtype = torch_cfloat())
  u_squared <- torch_square(u)
  common_factor <- 1 / torch_sqrt(1 - two_i * u * c * Delta_t)
  common_exp <- torch_exp(-0.5 * u_squared * Delta_t*torch_square(predic_g_X)*1/(1 - two_i * u * c * Delta_t) )
  exp_X_shift <- torch_exp(one_i * u * X_shift)
  phi <- exp_X_shift * common_factor * common_exp
  density <- torch_abs(torch_real(density_function(X_t_next, phi, M, h)))
  return(density)
 }






###############################################################
#                                                             #
#        ------ Definition of loss functions ------           #      
#                                                             #
###############################################################





#---Loss function D_1(\hat{f}, \hat{g}, B_k,j)----
loss_function_f_Fang <- function(predicted_f,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_next<-x_real[i, drop = FALSE]$clone()
      X_t<-x_real[i-1, drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      predicted_X[i-1] <- X_next-X_t-(f_X_t / (1 + Delta * torch_square(f_X_t)))  * Delta
   }  
 	 mse_loss <- torch_mean(torch_square(predicted_X)) 
  return(mse_loss)
}
#--------------------




#------Loss function D_2(\hat{f}, \hat{g}, B_k,j)---------
custom_combined_loss <- function(predicted_f, predicted_g, t,X_t, M, h) {
  density_loss <- torch_zeros(X_t$size(1)-1, dtype = torch_float()) 
  for (i in 1:(X_t$size(1)-1)) {
  	density_loss[i]<-1
    X_t_2<-X_t[i, drop = FALSE]$clone()
    f_X_t<-predicted_f(X_t_2)
    g_X_t <- predicted_g(X_t_2)
    Delta<-t[i+1,  drop = FALSE]-t[i, drop = FALSE]
    deriva<-X_t[i+1,  drop = FALSE]- X_t[i,  drop = FALSE]-f_X_t*Delta
    if(g_X_t$item()!=0){
    density_loss[i]<- Aproximation_density_red_W(
      X_t[i+1,  drop = FALSE],
      torch_tensor(seq(-M * h, M * h, by = h), dtype = torch_float()),  # u
     Delta,
      X_t[i,  drop = FALSE],
       M,
       h,
      predicted_f,
      predicted_g
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
#----Approximation parameters for the density
M<-200
h<-0.05
#------------------------



#-----------Loss function L_1(\hat{f}, \hat{g}, B_j, j)---------
loss_function_f_2 <- function(predicted_f,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_next<-x_real[i, drop = FALSE]$clone()
      X_t<-x_real[i-1, drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      predicted_X[i-1] <- X_next-X_t-(f_X_t / (1 + Delta * torch_square(f_X_t)))  * Delta
   }  
 	 mse_loss <- torch_abs(torch_mean(predicted_X)) 
  return(mse_loss)
}
#-----------------------------






#-----------Loss function L_2(\hat{f}, \hat{g}, B_j, j)---------
loss_function_g_second_2<- function(predicted_f,predicted_g,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-6,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_t<-x_real[i-1, drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- torch_square(x_real[i, drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
       predicted_X_2[i-1] <- 0.5*torch_square(Delta*d_g_X_t*g_X_t)+torch_square(g_X_t)*Delta
      }
  	 mse_loss <- torch_mean(torch_square(predicted_X-predicted_X_2))  
  return(mse_loss)
}
#-----------------------




#-----------------------H(\hat{f}, \hat{g}, B_j, j)---------
loss_function_g <- function(predicted_f,predicted_g,tiempos, x_real ) {
	index<-c()
	aux <- torch_zeros(1, dtype = torch_float()) 
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
# predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-4,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_t<-x_real[i-1, drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      aux <- (x_real[i, drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
       predicted_X[i-1]<-aux
      if(g_X_t$item()!=0){
      index<-c(index,i-1)	
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- aux/torch_sqrt(0.5*torch_square(Delta*d_g_X_t*g_X_t)+torch_square(g_X_t)*Delta)
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
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_next<-x_real[i, drop = FALSE]$clone()
      X_t<-x_real[i-1, drop = FALSE]$clone()
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
loss_function_g_2 <- function(predicted_f,predicted_g,tiempos, x_real ) {
 index<-c()
 predicted_X <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 predicted_X_2 <- torch_zeros(tiempos$size(1)-1, dtype = torch_float()) 
 h_next<-torch_tensor(10^-4,dtype = torch_float())
   for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_t<-x_real[i-1, drop = FALSE]$clone()
      f_X_t <- predicted_f(X_t)
      g_X_t <- predicted_g(X_t)
      d_g_X_t<-(predicted_g(X_t+h_next)-predicted_g(X_t-h_next))/(2*h_next)
      predicted_X[i-1] <- torch_square(x_real[i, drop = FALSE]-X_t - (f_X_t / (1 + Delta * torch_square(f_X_t))) * Delta)
      if(g_X_t$item()!=0){ 
         predicted_X_2[i-1] <- 0.5*torch_square(Delta*d_g_X_t*g_X_t)+torch_square(g_X_t)*Delta
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



#-----------------------Aditional Loss Function --------
loss_function_f_g <- function(predicted_f, predicted_g, tiempos, x_real, k_index,batch) {
    torch_manual_seed(batch)
  mse_values <- torch_zeros(c(k_index), dtype = torch_float())  
  for (k in 1:k_index) {
      with_no_grad({ 
  	W_diff <- torch_randn(tiempos$size(1) - 1) * torch_sqrt(torch_diff(tiempos)) 
  	})
    simulated_x <- torch_zeros(tiempos$size(1))
    simulated_x[1] <- x_real[1, drop = FALSE]
    for (i in 2:tiempos$size(1)) {
      Delta <- tiempos[i, drop = FALSE] - tiempos[i - 1, drop = FALSE]
      X_t<-simulated_x[i - 1, drop = FALSE]$clone()
      f_X_t <- predicted_f(simulated_x[i - 1, drop = FALSE]$clone())
      g_X_t <- predicted_g(simulated_x[i - 1, drop = FALSE]$clone())
      x_aux <- torch_tensor(X_t, requires_grad = TRUE)
      predicted_g(x_aux)$backward()
      d_g_X_t<-x_aux$grad
       constant_term <- 0.5 * g_X_t * d_g_X_t
      adjusted_X_t <- X_t + (f_X_t / (1 + Delta * f_X_t^2) - constant_term) * Delta
      simulated_x[i] <-  W_diff[i - 1] * g_X_t + constant_term * W_diff[i - 1]^2 + adjusted_X_t
     mse_values[k]<-mse_values[k]+torch_abs(simulated_x[i]-x_real[i,drop=FALSE])^2
    }
  mse_values[k]<-mse_values[k]/(tiempos$size(1)-1)
  }
  return(torch_mean(mse_values))
}
#-----------------------



#------Coefficient approximations using the neural network models-----
predicted_f<-function(x) {model(x)$predicted_f}
predicted_g<-function(x) {model2(x)$predicted_g}




###############################################################
#                                                             #
#    ------ Training methodology Fang et al. 2022 ------      #      
#                                                             #
###############################################################



#----Set the networks to training mode
model$train()
model2$train()

#---Define the number of epochs for the two-step method
epochs_f <- 400   
epochs_g <- 30


#--------Step 1-------
for (epoch in 1:epochs_f) {
	coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    order_indices <- order(as_array(batch_tiempos)) 
    batch_tiempos <- batch_tiempos[order_indices] 
    batch_x_real <- batch_x_real[order_indices]
    loss <- loss_function_f_Fang(
        function(x) predicted_f(x),
        batch_tiempos, batch_x_real)
       cat(sprintf("Epoch %d/%d, Batch Loss for f: %f\n", epoch, epochs_f, loss$item()))
    loss$backward()
    optimizer$step()
     })
}
#---------------------------------

#--------Step 2-------
for (epoch in 1:epochs_g) {
 coro::loop(for (batch in dataloader) {
    optimizer2$zero_grad()
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    order_indices <- order(as_array(batch_tiempos)) 
    batch_tiempos <- batch_tiempos[order_indices] 
    batch_x_real <- batch_x_real[order_indices]
    loss <- custom_combined_loss(function(x) predicted_f(x),function(x) predicted_g(x), batch_tiempos,batch_x_real, M, h)
    cat(sprintf("Epoch %d/%d, Batch Loss for g: %f\n", epoch, epochs_g, loss$item()))
    loss$backward()
    optimizer2$step()
     })
}
#---------------------------

###############################################################
#                                                             #
#  ------ Results: Comparison between the true coefficients   #
#  ------ and those estimated by the neural networks          #
#  ------ using Fang et al. 2022 methodology ------           #
#                                                             #
###############################################################


#-----Graphs------
vec_4 <- rep(0, tiempo_tensor$size(1))
vec_5 <- rep(0, tiempo_tensor$size(1))

vec_1 <- rep(0, tiempo_tensor$size(1))
vec_2 <- rep(0, tiempo_tensor$size(1))

for (i in 1:tiempo_tensor$size(1)) {
  vec_1[i] <- as.numeric(predicted_g(x_real_red_tensor[i, drop = FALSE]))
  vec_2[i] <- as.numeric(g(x_real_red_tensor[i, drop = FALSE]))
  vec_4[i] <- as.numeric(predicted_f(x_real_red_tensor[i, drop = FALSE]))
  vec_5[i] <- as.numeric(f(x_real_red_tensor[i, drop = FALSE]))
}

max_y <- max(c(vec_4, vec_5))
min_y <- min(c(vec_4, vec_5))

max_y_2 <- max(c(vec_1, vec_2))
min_y_2 <- min(c(vec_1, vec_2))

library(ggplot2)
library(gridExtra)

# Compute MSE with 4 decimal places
mse_f <- round(mean((vec_4 - vec_5)^2), 4)
mse_g <- round(mean((vec_1 - vec_2)^2), 4)

# Data frames
df_f <- data.frame(
  X_t = as.numeric(x_real_red_tensor),
  Real_f = vec_5,   # f(x) - red
  Est_f = vec_4     # \hat{f}(x) - blue
)

df_g <- data.frame(
  X_t = as.numeric(x_real_red_tensor),
  Real_g = vec_2,   # g(x) - red
  Est_g = vec_1     # \hat{g}(x) - blue
)

# Plot for f
plot_f <- ggplot(df_f, aes(x = X_t)) +
  geom_point(aes(y = Real_f), color = "red", shape = 1, size = 2, alpha = 0.7) +
  geom_point(aes(y = Est_f), color = "blue", shape = 4, size = 2, alpha = 0.7) +
  labs(
    title = expression("Comparison of " * f(X[t]) * " and " * hat(f)(X[t])),
    x = expression(X[t]),
    y = expression(f(X[t])),
    subtitle = paste("MSE:", mse_f)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.1))) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title = element_text(size = 14)
  )

# Plot for g
plot_g <- ggplot(df_g, aes(x = X_t)) +
  geom_point(aes(y = Real_g), color = "red", shape = 1, size = 2, alpha = 0.7) +
  geom_point(aes(y = Est_g), color = "blue", shape = 4, size = 2, alpha = 0.7) +
  labs(
    title = expression("Comparison of " * g(X[t]) * " and " * hat(g)(X[t])),
    x = expression(X[t]),
    y = expression(g(X[t])),
    subtitle = paste("MSE:", mse_g)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.1))) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title = element_text(size = 14)
  )

# Legend for f
legend_f <- ggplot() +
  geom_point(aes(x = 1, y = 1), color = "red", shape = 1, size = 3) +
  geom_text(aes(x = 1.2, y = 1, label = "f(x)"), parse = TRUE, hjust = 0, size = 4) +
  geom_point(aes(x = 2.5, y = 1), color = "blue", shape = 4, size = 3) +
  geom_text(aes(x = 2.7, y = 1, label = "hat(f)(x)"), parse = TRUE, hjust = 0, size = 4) +
  xlim(0.5, 4) + ylim(0.8, 1.2) +
  theme_void()

# Legend for g
legend_g <- ggplot() +
  geom_point(aes(x = 1, y = 1), color = "red", shape = 1, size = 3) +
  geom_text(aes(x = 1.2, y = 1, label = "g(x)"), parse = TRUE, hjust = 0, size = 4) +
  geom_point(aes(x = 2.5, y = 1), color = "blue", shape = 4, size = 3) +
  geom_text(aes(x = 2.7, y = 1, label = "hat(g)(x)"), parse = TRUE, hjust = 0, size = 4) +
  xlim(0.5, 4) + ylim(0.8, 1.2) +
  theme_void()

# Combine everything: plots on top, legends below, in parallel columns
top_row <- arrangeGrob(plot_f, plot_g, ncol = 2)
bottom_row <- arrangeGrob(legend_f, legend_g, ncol = 2)

# Display everything together
grid.arrange(top_row, bottom_row, heights = c(4, 1))



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
epochs_f_g<-30 


#-------Phase 1:

for (epoch in 1:epochs_f_g) {
	coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    optimizer2$zero_grad()
    batch_x_real <- batch[[1]]
    batch_tiempos <- batch[[2]]
    order_indices <- order(as_array(batch_tiempos)) 
    batch_tiempos <- batch_tiempos[order_indices] 
    batch_x_real <- batch_x_real[order_indices]
    loss <- custom_combined_loss(function(x) predicted_f(x),function(x) predicted_g(x), batch_tiempos,batch_x_real, M, h)
    cat(sprintf("Epoch %d/%d, Batch Loss for f: %f\n", epoch, epochs_f_g, loss$item()))
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
       batch_tiempos, batch_x_real) 
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
       batch_tiempos, batch_x_real)
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
      batch_tiempos, batch_x_real)
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
        batch_tiempos, batch_x_real)
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
      batch_tiempos, batch_x_real)
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
       batch_tiempos, batch_x_real)
    cat(sprintf("Epoch %d/%d, Batch Loss for g: %f\n", epoch, epochs_1+epochs_2, loss$item()))
    loss$backward()
    optimizer2$step()
    }
   } #----End Phase 3
 }

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




###############################################################
#                                                             #
#                    ------ Diagram ------                    #
#                                                             #
###############################################################



#---Packages and library

install.packages("DiagrammeR", type = "source")
install.packages("DiagrammeRsvg", type = "source")
install.packages("rsvg", type = "source")

library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)


#---Diagram
diagram <-grViz("
digraph G {

  graph [layout = dot, rankdir = LR, nodesep=1.6, ranksep=1.6]

  node [shape = circle, style=filled, fixedsize=true, fontname=\"Helvetica\", fontsize=24]

  # Nodo externo estilizado y grande
  data1 [label='Data', shape=circle, style=\"filled,setlinewidth(4)\", 
         fillcolor=\"white:gray90\", gradientangle=90, 
         fontsize=32, fontcolor=black, width=4, height=4, color=gray30]

  # Red f
  subgraph cluster_f {
    label = <<B><FONT POINT-SIZE='70'>Neural Network f</FONT></B><BR/><FONT POINT-SIZE='44'>Linear → ELU → ELU → ELU → ELU → Linear</FONT>>;
    style = filled;
    color = lightsteelblue1;
    fillcolor = ghostwhite;

    node [fillcolor = lightsteelblue2, width=0.85, height=0.85];
    f_input [label='In', fillcolor=palegreen, width=1.3, height=1.3];
    f_h1_1 [label=''];
    f_h1_2 [label=''];
    f_h1_3 [label=''];
    {rank = same; f_h1_1; f_h1_2; f_h1_3;}

    f_h2_1 [label=''];
    f_h2_2 [label=''];
    f_h2_3 [label=''];

    f_h3_1 [label=''];
    f_h3_2 [label=''];
    f_h3_3 [label=''];

    f_h4_1 [label=''];
    f_h4_2 [label=''];
    f_h4_3 [label=''];

    node [fillcolor = palegreen, fontcolor=black, width=1.3, height=1.3];
    f_out [label='f(x)'];

    dummy_f [label=\"\", shape=point, style=invis, width=0.01];
  }

  # Red g
  subgraph cluster_g {
    label = <<B><FONT POINT-SIZE='70'>Neural Network g</FONT></B><BR/><FONT POINT-SIZE='44'>Linear → ELU → ELU → ELU → Linear</FONT>>;
    style = filled;
    color = thistle3;
    fillcolor = snow;

    node [fillcolor = thistle2, width=0.85, height=0.85];
    g_input [label='In', fillcolor=palegreen, width=1.3, height=1.3];
    g_h1_1 [label=''];
    g_h1_2 [label=''];
    g_h1_3 [label=''];
    {rank = same; g_h1_1; g_h1_2; g_h1_3;}

    g_h2_1 [label=''];
    g_h2_2 [label=''];
    g_h2_3 [label=''];

    g_h3_1 [label=''];
    g_h3_2 [label=''];
    g_h3_3 [label=''];

    node [fillcolor = palegreen, fontcolor=black, width=1.3, height=1.3];
    g_out [label='g(x)'];

    dummy_g [label=\"\", shape=point, style=invis, width=0.01];
  }

  # Conexiones red f
  edge [penwidth=5, color=gray40];
  f_input -> f_h1_1; f_input -> f_h1_2; f_input -> f_h1_3;
  f_h1_1 -> f_h2_1; f_h1_1 -> f_h2_2; f_h1_1 -> f_h2_3;
  f_h1_2 -> f_h2_1; f_h1_2 -> f_h2_2; f_h1_2 -> f_h2_3;
  f_h1_3 -> f_h2_1; f_h1_3 -> f_h2_2; f_h1_3 -> f_h2_3;

  f_h2_1 -> f_h3_1; f_h2_1 -> f_h3_2; f_h2_1 -> f_h3_3;
  f_h2_2 -> f_h3_1; f_h2_2 -> f_h3_2; f_h2_2 -> f_h3_3;
  f_h2_3 -> f_h3_1; f_h2_3 -> f_h3_2; f_h2_3 -> f_h3_3;

  f_h3_1 -> f_h4_1; f_h3_1 -> f_h4_2; f_h3_1 -> f_h4_3;
  f_h3_2 -> f_h4_1; f_h3_2 -> f_h4_2; f_h3_2 -> f_h4_3;
  f_h3_3 -> f_h4_1; f_h3_3 -> f_h4_2; f_h3_3 -> f_h4_3;

  f_h4_1 -> f_out; f_h4_2 -> f_out; f_h4_3 -> f_out;

  # Conexiones red g
  g_input -> g_h1_1; g_input -> g_h1_2; g_input -> g_h1_3;
  g_h1_1 -> g_h2_1; g_h1_1 -> g_h2_2; g_h1_1 -> g_h2_3;
  g_h1_2 -> g_h2_1; g_h1_2 -> g_h2_2; g_h1_2 -> g_h2_3;
  g_h1_3 -> g_h2_1; g_h1_3 -> g_h2_2; g_h1_3 -> g_h2_3;

  g_h2_1 -> g_h3_1; g_h2_1 -> g_h3_2; g_h2_1 -> g_h3_3;
  g_h2_2 -> g_h3_1; g_h2_2 -> g_h3_2; g_h2_2 -> g_h3_3;
  g_h2_3 -> g_h3_1; g_h2_3 -> g_h3_2; g_h2_3 -> g_h3_3;

  g_h3_1 -> g_out; g_h3_2 -> g_out; g_h3_3 -> g_out;

  # Flechas externas con numeración y flechas más estéticas
  edge [penwidth=10, color=deepskyblue];
  data1 -> f_input [label='input', xlabel='1', fontcolor=navy, fontsize=50, labeldistance=8, labelloc=above];

  edge [penwidth=10, color=indianred];
  f_out -> g_input [label='output', xlabel='2', fontcolor=navy, fontsize=50, labeldistance=8, labelloc=above];

  edge [penwidth=10, color=deepskyblue];
  data1 -> g_input [label='input', xlabel='3', fontcolor=navy, fontsize=50, labeldistance=12, labelloc=above];

  edge [penwidth=10, color=indianred];
  g_out -> f_input  [label='output', xlabel='4', fontcolor=navy, fontsize=50, labeldistance=12, labelloc=above];
}")

# Convert to SVG
svg_code <- export_svg(diagram)

# Save as PDF (use rsvg_pdf from the {rsvg} package)
rsvg_pdf(charToRaw(svg_code), file = "network_diagram.pdf")
getwd()
