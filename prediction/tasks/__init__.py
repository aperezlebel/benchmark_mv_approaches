from .task_v2 import Task

from .TB_v2 import task_metas as TB_task_metas
from .UKBB_v2 import task_metas as UKBB_task_metas
from .MIMIC_v2 import task_metas as MIMIC_task_metas
from .NHIS_v2 import task_metas as NHIS_task_metas

# Merge tasks metas
tasks_meta = (
    TB_task_metas
    + UKBB_task_metas
    + MIMIC_task_metas
    + NHIS_task_metas
)

# Create tasks
tasks = {meta.tag: Task(meta) for meta in tasks_meta}
