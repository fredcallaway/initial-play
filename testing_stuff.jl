
function p_norm(vec::Vector{Float64}, p::Float64)
    val = sum(vec.^p)
    return val^(1/p)
end


rows = [[1., 2., 3.5], [1.5, 1.5, 1.5], [3., 3., 0.7]]

map(vec -> p_norm(vec, 1.), rows)
