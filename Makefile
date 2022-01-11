boxplot:
	python main.py figs mi

bagging:
	python main.py figs mi --bagging

linear:
	python main.py figs mi --linear

breakout:
	python main.py figs breakout

difficulty:
	python main.py figs difficulty

importance:
	python main.py figs imp --mode abs --task --root results_mia_importance

friedman-trees:
	python main.py figs friedman --ref MIA

friedman-linear:
	python main.py figs friedman --linear

friedman:
	make friedman-trees
	make friedman-linear

wilcoxon-trees:
	python main.py figs wilcoxon

wilcoxon-linear:
	python main.py figs wilcoxon --linear

wilcoxon:
	make wilcoxon-trees
	make wilcoxon-linear

scores-trees:
	python main.py figs scores

scores-linear:
	python main.py figs scores --linear

scores:
	make scores-trees
	make scores-linear

mv:
	python main.py datastats mv

desc:
	python main.py figs desc

ftypes:
	python main.py datastats ftypes

fcor:
	python main.py datastats fcor --abs

all:
	make boxplot
	make bagging
	make linear
	make breakout
	make difficulty
	make importance
	make friedman-trees
	make friedman-linear
	make friedman
	make wilcoxon-trees
	make wilcoxon-linear
	make wilcoxon
	make scores-trees
	make scores-linear
	make scores
	make desc
