"""test tasks."""
from prediction.tasks import tasks


t = tasks.get('TB/death_pvals', T=0, n_top_pvals=100)

X = t.X
y = t.y


print(X)
print(y)

print(y.value_counts())


print(X.index.equals(y.index))
