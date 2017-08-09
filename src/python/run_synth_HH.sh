for i in {0,1,2,3,4,5}
    do for j in {1,2,3}
        do s=$((3*$i + $j))
        (python projects/PyAP/python/synth_hh_mcmc.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled --non-adaptive -i 2000000 --seed $s && python projects/PyAP/python/plot_synth_hh_histograms.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled --seed $s --non-adaptive --burn 2) &
        done
        wait
    done

for i in {0,1,2,3,4,5}
    do for j in {1,2,3}
        do s=$((3*$i + $j))
        (python projects/PyAP/python/synth_hh_mcmc.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled -i 2000000 --seed $s && python projects/PyAP/python/plot_synth_hh_histograms.py --data-file projects/PyAP/python/input/synthetic_HH/traces/synthetic_HH.csv --unscaled --seed $s --burn 2) &
        done
        wait
    done

