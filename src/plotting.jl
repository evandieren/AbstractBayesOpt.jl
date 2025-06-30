function plot_state(p::BOProblem,n_train,x_cand,fn)
    d = length(p.domain.lower)
    plot_domain = nothing
    if d == 1
        plot_domain = collect(p.domain.lower[1]:0.01:p.domain.upper[1])
        grid_mean, grid_var = mean_and_var(p.gp.gpx(plot_domain))
        grid_values = p.f.(plot_domain)
        grid_acqf = [p.acqf(p.gp,x) for x in plot_domain]

        p1 = plot(plot_domain, grid_values,
        label="target function",
        xlim=(p.domain.lower[1], p.domain.upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, EI ξ=$(p.acqf.ξ), σ²=$(p.noise), iter=$(p.iter)",
        legend=:outertopright, size=(800,600))
        plot!(plot_domain, grid_mean; label="GP", ribbon=sqrt.(abs.(grid_var)),ribbon_scale=2,color="green")
        scatter!(
            reduce(vcat,p.xs)[1:n_train],
            reduce(vcat,p.ys)[1:n_train];
            label="Train Data"
        )
        scatter!(
            reduce(vcat,p.xs)[n_train+1:end],
            reduce(vcat,p.ys)[n_train+1:end];
            label="Candidates"
        )
        scatter!(
            [x_cand],
            [p.f(x_cand)];
            label="Candidate chosen",
            color=:orange
        )

        p2 = plot(plot_domain, grid_acqf,
        label="ACQ Function",
        xlim=(p.domain.lower[1], p.domain.upper[1]),
        xlabel="x",
        ylabel="y",
        legend=:outertopright)
        scatter!(
            [x_cand],
            [p.acqf(p.gp,x_cand)];
            label="Candidate chosen",
            color=:orange)

        p3 = plot(p1, p2, layout=(2,1), link=:x)
        savefig(p3, fn)
    else
        plot_grid_size = 250 # Grid for surface plot.
        x_grid = range(domain.lower[1], domain.upper[1], length=plot_grid_size)
        y_grid = range(domain.lower[2], domain.upper[2], length=plot_grid_size)
        grid_values = [p.f([x,y]) for x in x_grid, y in y_grid]
        grid_mean = [mean(p.gp.gpx([[x,y]]))[1] for x in x_grid, y in y_grid]
        grid_std = sqrt.([abs(var(p.gp.gpx([[x,y]]))[1]) for x in x_grid, y in y_grid])
        grid_acqf = [p.acqf(p.gp,[x,y]) for x in x_grid, y in y_grid]
    end
end