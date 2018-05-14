This was the primary [Chaste](https://github.com/Chaste/Chaste)-based project that I developed and used for my [DPhil](https://rhjohnstone.github.io/dphil/) for simulation of [cardiac action potentials](https://en.wikipedia.org/wiki/Cardiac_action_potential).

I tried to keep it tidy and reusable while I was using it regularly, but in the last few months of writing up, I probably let things slide for last-minute hacks for specific graphs and things. Sorry about that!

# PyAP

This is a Chaste project, originally based on James Grogran's [PyChaste](https://github.com/jmsgrogan/PyChaste) project. The idea is to wrap solving of action potential models (done by Chaste) in Python to make MCMC easier by passing vectors of parameter values all over the place.

Going to give instructions for using this on [arcus-b](http://www.arc.ox.ac.uk/content/arcus-phase-b) (Oxford University supercomputer).

Copy the included `.bash_profile` to `~/.bash_profile`. I've had weird things happen when I do

```bash
source ~/.bash_profile
```

Although this should work... Anyway, I just log out and log back in to arcus-b.

It's recommended to work in the `${DATA}` directory, so I would make `${DATA}/chaste-build`, and I also make `${DATA}/workspace`, probably not necessary, but just a habit from the old Chaste days.

I clone the Chaste repo as normal into `${DATA}/workspace/Chaste`, and this project into `${DATA}/workspace/PyAP`.

The PyAP project code needs to be included in the `projects` folder of the main Chaste source. This can be done with a symbolic link:

```bash
cd $CHASTE_SOURCE_DIR/projects
ln -s $PYAP_PROJECT_SOURCE_DIR
```

or just by copying the project in. To build, create a build directory outside the source tree and proceed as:

```bash
cd $BUILD_DIR
cmake ${DATA}/workspace/Chaste -DCMAKE_BUILD_TYPE=RELEASE -DBOOST_LIBRARYDIR=/system/software/linux-x86_64/lib/boost/1_58_0/lib -DBOOST_INCLUDEDIR=/system/software/linux-x86_64/lib/boost/1_58_0/include -DBoost_NO_SYSTEM_PATHS=BOOL:ON -DBoost_NO_BOOST_CMAKE=BOOL:ON -DXERCESC_LIBRARY=/system/software/linux-x86_64/xerces-c/3.3.1/lib/libxerces-c.so -DXERCESC_INCLUDE=/system/software/linux-x86_64/xerces-c/3.3.1/include/ -DXSD_EXECUTABLE=/system/software/linux-x86_64/lib/xsd/3.3.0-1/bin/xsd -DChaste_ERROR_ON_WARNING=OFF -DChaste_USE_CVODE:BOOL=ON -DSUNDIALS_INCLUDE_DIR=/system/software/arcus-b/lib/sundials/mvapich2-2.0.1/2.5.0/double/include/sundials -DXSD_INCLUDE_DIR=/system/software/linux-x86_64/lib/xsd/3.3.0-1/include -DSUNDIALS_sundials_nvecserial_LIBRARY=/system/software/linux-x86_64/lib/cvode/2.7.0/lib/libsundials_nvecserial.so -DSUNDIALS_sundials_cvode_LIBRARY=/system/software/linux-x86_64/lib/cvode/2.7.0/lib/libsundials_cvode.so
make [-jN] chaste_project_PyAP
make [-jN] project_PyAP_Python
```

where N is the number of processors to do the build with. I think you should omit the [-jN] on arcus-b, unless you're doing it as part of a SLURM job.

A suggestion: add
```bash
alias cchaste='cmake ${DATA}/workspace/Chaste -DCMAKE_BUILD_TYPE=RELEASE -DBOOST_LIBRARYDIR=/system/software/linux-x86_64/lib/boost/1_58_0/lib -DBOOST_INCLUDEDIR=/system/software/linux-x86_64/lib/boost/1_58_0/include -DBoost_NO_SYSTEM_PATHS=BOOL:ON -DBoost_NO_BOOST_CMAKE=BOOL:ON -DXERCESC_LIBRARY=/system/software/linux-x86_64/xerces-c/3.3.1/lib/libxerces-c.so -DXERCESC_INCLUDE=/system/software/linux-x86_64/xerces-c/3.3.1/include/ -DXSD_EXECUTABLE=/system/software/linux-x86_64/lib/xsd/3.3.0-1/bin/xsd -DChaste_ERROR_ON_WARNING=OFF -DChaste_USE_CVODE:BOOL=ON -DSUNDIALS_INCLUDE_DIR=/system/software/arcus-b/lib/sundials/mvapich2-2.0.1/2.5.0/double/include/sundials -DXSD_INCLUDE_DIR=/system/software/linux-x86_64/lib/xsd/3.3.0-1/include -DSUNDIALS_sundials_nvecserial_LIBRARY=/system/software/linux-x86_64/lib/cvode/2.7.0/lib/libsundials_nvecserial.so -DSUNDIALS_sundials_cvode_LIBRARY=/system/software/linux-x86_64/lib/cvode/2.7.0/lib/libsundials_cvode.so'
alias cdchaste='cd ${DATA}/chaste-build'
alias cdata='cd ${DATA}'
```

to ~/.bashrc so you can just type cchaste instead of copying and pasting that whole thing. Plus the others.

## Usage

Keep Python scripts in
```bash
$PYAP_PROJECT_SOURCE_DIR/src/python
```

and run them from inside
```bash
$BUILD_DIR
```

with, for example (this script is already included)
```bash
python projects/PyAP/python/pyap_example.py
```

## MCMC-like stuff

You have to be careful with setting cell membrane capacitances. Some of the CellML models scale all currents by the capacitance, others only scale the stimulus current (or all _except_ the stimulus, I don't remember), and then you have to specify these things in the config file.

There is an optional, but recommended, CMA-ES minimisation script. If you run this before the MCMC, the MCMC will start from the best point found by the CMA-ES. If you don't run the CMA-ES, the MCMC will just start from the original model parameters, and so will either take a while to find the mode and converge, or might get stuck in a local optimum that is much less good than the "real" one.

To run, the input data trace must be located at
```bash
$PYAP_PROJECT_SOURCE_DIR/src/python/input/$EXPT_NAME/traces/$TRACE_NAME.csv
```
where it is in csv format, with the first column being time and the second column being membrane voltage, both already in the correct units.
There must also be

```bash
$PYAP_PROJECT_SOURCE_DIR/src/python/input/$EXPT_NAME/PyAP_options.txt
```
See already-included examples for how this file must look. Currently all of those options must be there, but their values can be changed.

To run the CMA-ES, you also have to specify how many cores to use, and how many CMA-ES minimisations to perform. e.g.
```bash
python projects/PyAP/python/general_cmaes.py --data-file projects/PyAP/python/input/$EXPT_NAME/traces/$TRACE_NAME.csv --num-cores 3 --num-runs 9
```
Assuming this doesn't break for some weird reason, it should also plot the trace generated by the overall best-fitting parameters.
If you're not on arcus-b, the output will be saved in
```bash
$BUILD_DIR/projects/PyAP/python/output
```

To run the MCMC, you need to supply the data file again:
```bash
python projects/PyAP/python/general_mcmc.py --data-file projects/PyAP/python/input/$EXPT_NAME/traces/$TRACE_NAME.csv
```
This will save the whole (thinned) chain (minus burn-in, to save space) as a text file.

The marginal histograms can then be plotted with:
```bash
python projects/PyAP/python/general_plot_histograms.py --data-file projects/PyAP/python/input/$EXPT_NAME/traces/$TRACE_NAME.csv
```
