"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using BayesOpt
using LinearAlgebra
using JLD2
import Random
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
∇f(x) = ForwardDiff.gradient(f, x)
min_f = −1.988699758534924
f_val_grad(x) = [f(x); ∇f(x)]

d = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

grad_kernel = gradKernel(ApproxMatern52Kernel())

σ² = 1e-6
model = GradientGP(grad_kernel,d+1,σ²)

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
println(x_train)
val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] for i = eachindex(val_grad)]
#y_train = [val_grad[i] + sqrt(σ²)*randn(d+1) for i = eachindex(val_grad)]

println(y_train)


using Optim
# Negative log marginal likelihood (no noise term)
function nlml_grad(params,kernel,X_train,y_train,σ²)
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(ℓ))
    mod = GradientGP(gradKernel(k),d+1, σ²)

    #println(mean(gp.gpx(x_train)))

    x̃, ỹ = KernelFunctions.MOInputIsotopicByOutputs(X_train, size(y_train[1])[1]), vec(permutedims(reduce(hcat, y_train)))

    -AbstractGPs.logpdf(mod.gp(x̃,σ²), ỹ)  # Negative log marginal likelihood
end


# Initial log-parameters: log(lengthscale), log(magnitude)
initial_params = [log(1.0), log(1.0)]

# Optimize with BFGS
res = Optim.optimize(x -> nlml_grad(x,ApproxMatern52Kernel(),x_train,y_train,σ²), initial_params, Optim.Newton())

res

# Extract optimized values
opt_params = Optim.minimizer(res)
ell_opt = exp(opt_params[1])
scale_opt = exp(opt_params[2])

println("Optimized lengthscale: ", ell_opt)
println("Optimized magnitude: ", scale_opt)

grad_kernel = gradKernel(scale_opt *(ApproxMatern52Kernel() ∘ ScaleTransform(ell_opt)))
model = GradientGP(grad_kernel,d+1,σ²)
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train)


# plot_domain = collect(lower[1]:0.01:upper[1])

# plot_x = map(x -> [x], plot_domain)
# plot_x = prep_input(model,plot_x)
# post_mean, post_var = mean_and_var(model.gpx(plot_x))
# post_mean = reshape(post_mean, :, d+1)[:,1] # This returns f(x) to match the StandardGP
# post_var = reshape(post_var, :, d+1)[:,1]
# post_var[post_var .< 0] .= 0


# plot(plot_domain,post_mean,ribbon=sqrt.(post_var),ribbon_scale=2)
# plot!(plot_domain,f.(plot_domain))
# Init of the acquisition function
ξ = 1e-3
acqf1 = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))
acqf2 = ProbabilityDescent()
acqf3 = EnsembleAcquisition([0.9, 0.1], [acqf1, acqf2])

# This maximises the function
problem = BOProblem(
                    f_val_grad,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    40,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = BayesOpt.optimize(problem)
xs = reduce(vcat,result.xs)
ys = hcat(result.ys...)[1,:]

running_min = accumulate(min, f.(xs)) #[n_train+1:end]

feval_grad = 2:2:(2*length(running_min))
error_grad = norm.(running_min .- min_f)

xs_ = prep_input(result.gp, result.xs) 

K̃ = kernelmatrix(result.gp.gp.kernel,xs_,xs_) + σ²*I(length(xs_))
κ_K = cond(K̃)

@save "grad_bo_1d.jld2" feval_grad error_grad

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

plot_domain = collect(lower[1]:0.01:upper[1])

plot_x = map(x -> [x], plot_domain)
plot_x = prep_input(model,plot_x)
post_mean, post_var = mean_and_var(result.gp.gpx(plot_x))

post_mean = reshape(post_mean, :, d+1)[:,1] # This returns f(x) to match the StandardGP
post_var = reshape(post_var, :, d+1)[:,1]
post_var[post_var .< 0] .= 0


plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, σ²=$(σ²)",
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
savefig("gradgp_matern_1D_ensemble_prob_descent.pdf")