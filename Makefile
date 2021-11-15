boxplot:
	python main.py figs mi

boxplot-a:
	python main.py figs mi -a

bagging:
	python main.py figs mi --bagging

bagging-a:
	python main.py figs mi --bagging -a

breakout:
	python main.py figs breakout

breakout-a:
	python main.py figs breakout -a

difficulty:
	python main.py figs difficulty

difficulty-a:
	python main.py figs difficulty -a

importance:
	python main.py figs imp --mode abs --task --root results_mia_importance

importance-a:
	python main.py figs imp --mode abs --task --root results_mia_importance -a

all_article:
