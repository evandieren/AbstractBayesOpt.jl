"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using JLD2
using BayesOpt
using LaTeXStrings
import Random
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))

problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-12 # 1e-10

kernel = 2*Matern52Kernel()
model = StandardGP(kernel, σ²) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 5
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

println(x_train)
y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)
println(y_train)
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))
#acqf = KnowledgeGradient(domain, [optimize_mean!(model, domain)[2]])

#plot_domain = collect(lower[1]:0.1:upper[1])
#acqf_dom = [acqf(model,x) for x in plot_domain]
#plot(plot_domain,acqf_dom)

#plot(plot_domain,f.(plot_domain))
#line!(optimize_mean!(model, domain))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    85,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = reduce(vcat,result.xs)
ys = reduce(vcat,result.ys)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))


@load "grad_bo_1d.jld2" feval_grad error_grad

running_min = accumulate(min, f.(xs))

Plots.plot(1:length(running_min),norm.(running_min .+ 1.9887),yaxis=:log, title="Error w.r.t true minimum (1D BO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="BO",xlims=(1,length(running_min)))
Plots.vspan!([1,n_train]; color=:blue,alpha=0.2, label="")
Plots.vspan!([n_train,2*n_train]; color=:purple,alpha=0.2, label="")
Plots.plot(feval_grad,error_grad,label="gradBO",yaxis=:log)
savefig("1D_error_BO.pdf")

plot_domain = collect(lower[1]:0.01:upper[1])

plot_domain = prep_input(model,plot_domain)

post_mean, post_var = mean_and_var(result.gp.gpx(plot_domain))

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, EI ξ=$(ξ), σ²=$(σ²)",
        legend=:outertopright)
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(abs.(post_var)),ribbon_scale=2,color="green")
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
savefig("gp_Matern_1D.pdf")