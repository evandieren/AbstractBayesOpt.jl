"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using BayesOpt
using LinearAlgebra


import Random
Random.seed!(1234)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
∇f(x) = ForwardDiff.gradient(f, x)

f_val_grad(x) = [f(x); ∇f(x)]

d = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

RBF_kernel(x, y) = exp(-norm(x .- y)^2/2)
∂ₓRBF_kernel(x, y, i) = RBF_kernel(x, y)*(y[i] - x[i]) # partial deriv ∂/∂xᵢ k(x,y)

grad_kernel = gradKernel(RBF_kernel,∂ₓRBF_kernel) #gradKernel(matern52_kernel)

model = GradientGP(grad_kernel,d+1)

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]


println(x_train)

σ² = 1e-3
val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] + sqrt(σ²)*randn(d+1) for i = eachindex(val_grad)]

println(y_train)
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train, σ²)

# Init of the acquisition function
ξ = 0.5
acqf = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))

# This maximises the function
problem = BOProblem(
                    f_val_grad,
                    domain,
                    model,
                    x_train,
                    y_train,
                    acqf,
                    10,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = reduce(vcat,result.xs)
ys = hcat(result.ys...)[1,:]

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

plot_domain = collect(lower[1]:0.01:upper[1])

plot_x = map(x -> [x], plot_domain)
plot_x = prep_input(model,plot_x)
post_mean, post_var = mean_and_var(result.gp.gpx(plot_x))

if isa(model, GradientGP)
    post_mean = reshape(post_mean, :, d+1)[:,1] # This returns f(x) to match the StandardGP
    post_var = reshape(post_var, :, d+1)[:,1]
    post_var[post_var .< 0] .= 0
end

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, EI ξ=$(ξ), σ²=$(σ²)",
        legend=:outertopright)
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(post_var),ribbon_scale=2,color="green")
scatter!(
    xs[1:n_train],
    ys[1:n_train];
    label="Train Data"
)
scatter!(
    xs[n_train+1:end],
    ys[n_train+1:end];
    label="Candidates"
)
scatter!(
    [xs[argmin(ys)]],
    [minimum(ys)];
    label="Best candidate"
)
savefig("gradgp_RBF.pdf")