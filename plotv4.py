"""Plot the results of --train4."""
from prediction.PlotHelperV4 import PlotHelperV4


ph = PlotHelperV4(root_folder='results_selected/FINAL_RESULTS/trial1/')

print(ph.databases())
print(ph.tasks('TB'))
print(ph.methods('TB', 'platelet'))
