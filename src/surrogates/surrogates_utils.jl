Base.copy(gp::AbstractGPs.PosteriorGP) = AbstractGPs.PosteriorGP(gp.prior,NamedTuple{keys(gp.data)}(map(copy, values(gp.data))))
Base.copy(s::Nothing) = nothing