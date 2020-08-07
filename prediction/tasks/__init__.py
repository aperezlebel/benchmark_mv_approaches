"""Gather tasks metas from all databases and create the task accessor."""
from .task import Task

from .TB import task_metas as TB_task_metas
from .UKBB import task_metas as UKBB_task_metas
from .MIMIC import task_metas as MIMIC_task_metas
from .NHIS import task_metas as NHIS_task_metas


class _TaskAccessor(object):
    """Gather tasks from all databases and give an accessor to them."""

    def __init__(self):
        """Init."""
        self.task_metas = {
            'TB': TB_task_metas,
            'UKBB': UKBB_task_metas,
            'MIMIC': MIMIC_task_metas,
            'NHIS': NHIS_task_metas,
        }

    def get(self, tag, n_top_pvals=100, RS=0, T=0):
        """Return asked task with given parameters."""
        db, name = tag.split('/')
        task_meta = self.task_metas[db]

        kwargs = {
            'n_top_pvals': n_top_pvals,
            'RS': RS,
            'T': T,
        }
        return Task(task_meta[name](**kwargs))

    def __getitem__(self, tag):
        """Access a task with default parameters."""
        return self.get(tag)

    def keys(self):
        """Iteratate over task names and metas."""
        for db, metas in self.task_metas.items():
            for task_name in metas.keys():
                yield f'{db}/{task_name}'


tasks = _TaskAccessor()
