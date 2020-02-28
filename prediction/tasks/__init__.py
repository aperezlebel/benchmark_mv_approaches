from .task import Task

from .TB import tasks_meta as TB_tasks_meta
from .UKBB import tasks_meta as UKBB_tasks_meta

# Merge tasks metas
tasks_meta = TB_tasks_meta + UKBB_tasks_meta

# Create tasks
tasks = {meta.tag: Task(meta) for meta in tasks_meta}
