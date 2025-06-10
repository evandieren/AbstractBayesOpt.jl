"""
This short example shows a 2D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using LaTeXStrings
using BayesOpt
using LinearAlgebra

import Random
Random.seed!(123456)

# Objective Function 
# Branin
function branin(x::AbstractVector)
    x1 = x[1]
    x2 = x[2]
    b = 5.1 / (4*pi^2);
    c = 5/pi;
    r = 6;
    a = 1;
    s = 10;
    t = 1 / (8*pi);
    term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
    term2 = s*(1-t)*cos(x1);
    y = term1 + term2 + s;
end

# Rosenbrock
rosenbrock(x::AbstractVector) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Himmelblau
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2

f(x) = himmelblau(x)

problem_dim = 2
lower = [-6,-6.0] #[-5.0, 0.0]
upper = [6.0,6.0] #[10.0, 15.0]
domain = ContinuousDomain(lower, upper)
σ² = 0.0

kernel = Matern52Kernel()
model = StandardGP(kernel,σ²) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
y_train = f.(x_train) + sqrt(σ²).* randn(n_train)
y_train = map(x -> [x], y_train)

# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed)
model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(y_train)[1])

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    x_train,
                    y_train,
                    acqf,
                    80,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = result.xs
ys = result.ys

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

running_min = accumulate(min, f.(xs))

Plots.plot(n_train:length(running_min),norm.(running_min)[n_train:end],yaxis=:log, title="Error w.r.t true minimum (2D BO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="BO",xlims=(1,length(running_min)))
Plots.vspan!([1,n_train]; color=:blue,alpha=0.2, label="")

# Uncomment for GLMakie plotting 
# using GLMakie
# @info "Starting plotting procedure..."
# plot_grid_size = 250 # Grid for surface plot.
# x_grid = range(domain.lower[1], domain.upper[1], length=plot_grid_size)
# y_grid = range(domain.lower[2], domain.upper[2], length=plot_grid_size)

# grid_values = [f([x,y]) for x in x_grid, y in y_grid]
# grid_mean = [mean(result.gp.gpx([[x,y]]))[1] for x in x_grid, y in y_grid]
# grid_std = sqrt.([abs(var(result.gp.gpx([[x,y]]))[1]) for x in x_grid, y in y_grid])
# grid_acqf = [result.acqf(result.gp,[x,y]) for x in x_grid, y in y_grid]


# fig = Figure(;size=(1000, 600))
# ax1 = Axis(fig[1, 1], title="True function", xlabel="X-axis", ylabel="Y-axis")
# ax2 = Axis(fig[1, 2], title="Posterior mean", xlabel="X-axis", ylabel="Y-axis")
# ax3 = Axis(fig[2, 1], title="Posterior standard deviation", xlabel="X-axis", ylabel="Y-axis")
# ax4 = Axis(fig[2, 2], title="Acquisition function (EI)", xlabel="X-axis", ylabel="Y-axis")

# #GLMakie.contour!(ax1, x_grid, y_grid, grid_values, colormap=:viridis, levels=200)
# x1_coords = hcat(xs...)[1,:]
# x1_train = x1_coords[1:n_train]
# x1_candidates = x1_coords[n_train+1:end]

# x2_coords = hcat(xs...)[2,:]
# x2_train = x2_coords[1:n_train]
# x2_candidates = x2_coords[n_train+1:end]


# # True function evaluation
# GLMakie.surface!(ax1,x_grid, y_grid,fill(0f0, size(grid_values));
#                  color=grid_values, shading = NoShading, colormap = :coolwarm)
# GLMakie.scatter!(ax1, x1_train, x2_train,color="white", marker=:cross)
# GLMakie.scatter!(ax1, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
# GLMakie.scatter!(ax1, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
# Colorbar(fig[1, 1][1, 2],
#          limits = (minimum(grid_values),maximum(grid_values)),
#          colormap=:coolwarm)

# # Posterior mean
# GLMakie.surface!(ax2,x_grid, y_grid,fill(0f0, size(grid_mean));
#                  color=grid_mean, shading = NoShading,
#                  colorrange = (minimum(grid_values),maximum(grid_values)),
#                  colormap = :coolwarm)
# GLMakie.scatter!(ax2, x1_train, x2_train,color="white", marker=:cross)
# GLMakie.scatter!(ax2, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
# GLMakie.scatter!(ax2, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
# Colorbar(fig[1, 2][1, 2],
#          limits = (minimum(grid_values),maximum(grid_values)),
#          colormap=:coolwarm)

# # Posterior variance
# GLMakie.surface!(ax3,x_grid, y_grid,fill(0f0, size(grid_std));
#                  color=grid_std, shading = NoShading,
#                  colormap = :coolwarm)
# GLMakie.scatter!(ax3, x1_train, x2_train,color="white", marker=:cross)
# GLMakie.scatter!(ax3, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
# GLMakie.scatter!(ax3, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
# Colorbar(fig[2, 1][1, 2],limits = (minimum(grid_std),maximum(grid_std)),
#             colormap=:coolwarm)

# # Acquisition function
# GLMakie.surface!(ax4,x_grid, y_grid,fill(0f0, size(grid_acqf)); 
#                  color=grid_acqf, shading = NoShading,
#                  colormap = :coolwarm)
# GLMakie.scatter!(ax4, x1_train, x2_train,color="white", marker=:cross)
# GLMakie.scatter!(ax4, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
# GLMakie.scatter!(ax4, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
# Colorbar(fig[2, 2][1, 2],limits = (minimum(grid_acqf),maximum(grid_acqf)),
# colormap=:coolwarm)

# #savefig(fig,"output_example_2D.png")
# GLMakie.activate!(inline=true)
# display(fig)
# save("gp_matern_2D_10_60.png",fig)