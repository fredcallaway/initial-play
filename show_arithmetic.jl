module Tmp

push!(LOAD_PATH, pwd())
using Arithmetic

g = Grammar()
add!(g, Op(
    "plus",
    (a, b) -> :($a + $b),
    (Int, Int),
    Int,
))
add!(g, Op(
    "minus",
    (a, b) -> :($a - $b),
    (Int, Int),
    Int
))
for i = 0:9
    add!(g, i)
end

program = gen(g, Int, 0.4; max_depth=3)
print(compile(program))

end #module
