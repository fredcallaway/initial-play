module Arithmetic

import Base
using DataStructures: DefaultDict

export Grammar, Op, Node, compile, gen, add!

"An operation (i.e. non-terminal) in a program grammar."
struct Op
    name::String
    f::Function
    child_types::Tuple{Vararg{Type}}
    return_type::Type
end
Base.show(io::IO, op::Op) = print(io, op.name)

"A node in a syntax tree."
struct Node
    root
    children
end
Node(x) = Node(x, [])
Base.show(io::IO, node::Node) = show(io, compile(node))

"Returns an expression or base type that computes the Node's program."
function compile(node::Node)
    if length(node.children) > 0
        node.root.f(map(compile, node.children)...)
    else
        :($(node.root))
    end
end

"A grammar over programs."
struct Grammar
    operations
    terminals
end
Grammar() = begin
    Grammar(
        DefaultDict(() -> Op[]),
        DefaultDict( (T) -> T[]; passkey=true),
    )
end

"Adds an Op or terminal to a grammar."
function add!(g::Grammar, x)
    if x isa Op
        push!(g.operations[x.return_type], x)
    else
        push!(g.terminals[typeof(x)], x)
    end
end

"Generate a program from a grammar."
function gen(g::Grammar, t::Type, p_term::Float64; max_depth=10)
    if rand() < p_term || max_depth == 0
        Node(rand(g.terminals[t]))
    else
        op = rand(g.operations[t])
        children = [gen(g, ct, p_term; max_depth=max_depth-1)
                    for ct in op.child_types]
        Node(op, children)
    end
end

Base.show(io::IO, g::Grammar) = begin
    println(io, "Grammar")
    println(io, "  Operations")
    for (t, ops) in pairs(g.operations)
        println(io, string("    ", t, ": ", join(ops, ", ")))
    end
    println("  Terminals")
    for (t, xs) in pairs(g.terminals)
        print(io, string("    ", t, ": ", join(xs, ", ")))
    end
end

end # module
