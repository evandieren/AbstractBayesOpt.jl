# using GLMakie
# 
# function plot_state(p::BOStruct,n_train,x_cand,fn)
#     d = length(p.domain.lower)
#     plot_domain = nothing
# 
#     if d == 1
#         plot_domain = collect(p.domain.lower[1]:0.01:p.domain.upper[1])
#         grid_mean, grid_var = mean_and_var(p.gp.gpx(plot_domain))
#         grid_values = p.f.(plot_domain)
#         grid_acqf = [p.acqf(p.gp,x) for x in plot_domain]
# 
#         p1 = Plots.plot(plot_domain, grid_values,
#         label="target function",
#         xlim=(p.domain.lower[1], p.domain.upper[1]),
#         ylabel="y",
#         title="AbstractBayesOpt, EI ξ=$(p.acqf.ξ), σ²=$(p.noise), iter=$(p.iter)",
#         legend=:outertopright, size=(800,600))
#         Plots.plot!(plot_domain, grid_mean; label="GP", ribbon=sqrt.(abs.(grid_var)),ribbon_scale=2,color="green")
#         Plots.scatter!(
#             reduce(vcat,p.xs)[1:n_train],
#             reduce(vcat,p.ys)[1:n_train];
#             label="Train Data"
#         )
#         Plots.scatter!(
#             reduce(vcat,p.xs)[n_train+1:end],
#             reduce(vcat,p.ys)[n_train+1:end];
#             label="Candidates"
#         )
#         Plots.scatter!(
#             [x_cand],
#             [p.f(x_cand)];
#             label="Candidate chosen",
#             color=:orange
#         )
# 
#         p2 = Plots.plot(plot_domain, grid_acqf,
#         label="ACQ Function",
#         xlim=(p.domain.lower[1], p.domain.upper[1]),
#         xlabel="x",
#         ylabel="y",
#         legend=:outertopright, 
#         title="EI: $(round(p.acqf(p.gp,x_cand), sigdigits=2))")
#         Plots.scatter!(
#             [x_cand],
#             [p.acqf(p.gp,x_cand)];
#             label="Candidate chosen",
#             color=:orange)
# 
#        p3 = Plots.plot(p1, p2, layout=(2,1), link=:x)
#         Plots.savefig(p3, fn)
#     else
#         plot_grid_size = 250 # Grid for surface plot.
#         x_grid = range(p.domain.lower[1], p.domain.upper[1], length=plot_grid_size)
#         y_grid = range(p.domain.lower[2], p.domain.upper[2], length=plot_grid_size)
#         grid_values = [p.f([x,y]) for x in x_grid, y in y_grid]
#         grid_mean = [mean(p.gp.gpx([[x,y]]))[1] for x in x_grid, y in y_grid]
#         grid_std = sqrt.([abs(var(p.gp.gpx([[x,y]]))[1]) for x in x_grid, y in y_grid])
#         grid_acqf = [p.acqf(p.gp,[x,y]) for x in x_grid, y in y_grid]
# 
#         x1_coords = hcat(p.xs...)[1,:]
#         x1_train = x1_coords[1:n_train]
#         x1_candidates = x1_coords[n_train+1:end]
# 
#         x2_coords = hcat(p.xs...)[2,:]
#         x2_train = x2_coords[1:n_train]
#         x2_candidates = x2_coords[n_train+1:end]
# 
#         fig = GLMakie.Figure(;size=(1000, 600))
#         ax1 = Axis(fig[1, 1], title="True function, iter $(p.iter)", xlabel="X", ylabel="Y")
#         ax2 = Axis(fig[1, 2], title="ACQ func", xlabel="X", ylabel="Y")
# 
#         # True function evaluation
#         GLMakie.surface!(ax1,x_grid, y_grid,fill(0f0, size(grid_values));
#                         color=grid_values, shading = NoShading, colormap = :coolwarm)
#         GLMakie.scatter!(ax1, x1_train, x2_train,color="white", marker=:cross)
#         GLMakie.scatter!(ax1, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
#         GLMakie.scatter!(ax1, x_cand[1],x_cand[2],color=:orange,marker=:star6,markersize=20)
#         # GLMakie.scatter!(ax1, [p.xs[argmin(p.ys)][1]], [p.xs[argmin(p.ys)][2]],marker=:star5,markersize=15,color="green")
#         Colorbar(fig[1, 1][1, 2],
#                 limits = (minimum(grid_values),maximum(grid_values)),
#                 colormap=:coolwarm)

        # # Posterior mean
        # GLMakie.surface!(ax2,x_grid, y_grid,fill(0f0, size(grid_mean));
        #                 color=grid_mean, shading = NoShading,
        #                 colorrange = (minimum(grid_values),maximum(grid_values)),
        #                 colormap = :coolwarm)
        # GLMakie.scatter!(ax2, x1_train, x2_train,color="white", marker=:cross)
        # GLMakie.scatter!(ax2, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
        # # GLMakie.scatter!(ax2, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
        # Colorbar(fig[1, 2][1, 2],
        #         limits = (minimum(grid_values),maximum(grid_values)),
        #         colormap=:coolwarm)

        # # Posterior variance
        # GLMakie.surface!(ax3,x_grid, y_grid,fill(0f0, size(grid_std));
        #                 color=grid_std, shading = NoShading,
        #                 colormap = :coolwarm)
        # GLMakie.scatter!(ax3, x1_train, x2_train,color="white", marker=:cross)
        # GLMakie.scatter!(ax3, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
        # # GLMakie.scatter!(ax3, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
        # Colorbar(fig[2, 1][1, 2],limits = (minimum(grid_std),maximum(grid_std)),
        #             colormap=:coolwarm)

        # Acquisition function
#         GLMakie.surface!(ax2,x_grid, y_grid,fill(0f0, size(grid_acqf)); 
#                         color=grid_acqf, shading = NoShading,
#                         colormap = :coolwarm)
#         GLMakie.scatter!(ax2, x1_train, x2_train,color="white", marker=:cross)
#         GLMakie.scatter!(ax2, x1_candidates, x2_candidates,color= 1:length(x1_candidates),colormap=:viridis)
#         GLMakie.scatter!(ax2, x_cand[1],x_cand[2],color=:orange,marker=:star6,markersize=20)
#         # GLMakie.scatter!(ax4, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],marker=:star5,markersize=15,color="green")
#         Colorbar(fig[1, 2][1, 2],limits = (minimum(grid_acqf),maximum(grid_acqf)),
#         colormap=:coolwarm)
# 
#         save(fn,fig)
#     end
# end
