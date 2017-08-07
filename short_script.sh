cd ~/chaste-build
cmake ~/workspace/Chaste

python projects/PyAP/python/general_cmaes.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --num-runs 102 --num-cores 3

python projects/PyAP/python/general_mcmc.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled --non-adaptive -i 1000000

python projects/PyAP/python/general_mcmc.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled -i 1000000

python projects/PyAP/python/general_plot_histograms.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled --non-adaptive

python projects/PyAP/python/general_plot_histograms.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled

python projects/PyAP/python/plot_HH.py  --unscaled

python projects/PyAP/python/plot_HH.py  --unscaled --non-adaptive

