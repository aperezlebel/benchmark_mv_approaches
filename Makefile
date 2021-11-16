boxplot:
	python main.py figs mi

boxplot-a:
	python main.py figs mi -a

bagging:
	python main.py figs mi --bagging

bagging-a:
	python main.py figs mi --bagging -a

linear:
	python main.py figs mi --linear

linear-a:
	python main.py figs mi --linear -a

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

friedman-trees:
	python main.py figs friedman --ref MIA

friedman-trees-a:
	python main.py figs friedman --ref MIA -a

friedman-linear:
	python main.py figs friedman --linear

friedman-linear-a:
	python main.py figs friedman --linear -a

friedman:
	make friedman-trees
	make friedman-linear

friedman-a:
	make friedman-trees-a
	make friedman-linear-a

all_article:
