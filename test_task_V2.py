from prediction.tasks.task_v2 import Task
from prediction.tasks.MIMIC_v2 import task_metas


meta = task_metas[0]
t = Task(meta)


X = t.X
y = t.y


print(X)
print(y)

print(y.value_counts())


print(X.index.equals(y.index))
