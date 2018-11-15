using Flux
using Flux.Tracker
using Flux.Tracker: update!


b = param(rand(2))
loss() = sum(b.^2)

params = Params([b])
grads = Tracker.gradient(() -> loss(), params)
grads[b]
opt = SGD([b], 0.1)
opt()
b


η = 0.1 # Learning Rate
for p in (w, b)
  update!(p, -η * grads[p])
end

p = b
update!(p, -.1 * grads[p])

println('-'^70)
for i in 1:10
  sgd()
  println(loss(x))
end
