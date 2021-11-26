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

wilcoxon-trees:
	python main.py figs wilcoxon

wilcoxon-trees-a:
	python main.py figs wilcoxon -a

wilcoxon-linear:
	python main.py figs wilcoxon --linear

wilcoxon-linear-a:
	python main.py figs wilcoxon --linear -a

wilcoxon:
	make wilcoxon-trees
	make wilcoxon-linear

wilcoxon-a:
	make wilcoxon-trees-a
	make wilcoxon-linear-a

scores-trees:
	python main.py figs scores

scores-trees-a:
	python main.py figs scores -a

scores-linear:
	python main.py figs scores --linear

scores-linear-a:
	python main.py figs scores --linear -a

scores:
	make scores-trees
	make scores-linear

scores-a:
	make scores-trees-a
	make scores-linear-a

mv:
	python main.py datastats mv

mv:
	python main.py datastats mv -a

desc:
	python main.py figs desc

desc-a:
	python main.py figs desc -a

all_article:
