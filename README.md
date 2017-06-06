# PyAP

Going to give instructions for using this on arcus-b.

This is a Chaste project, originally based on James Grogran's [PyChaste](https://github.com/jmsgrogan/PyChaste) project. The idea is to wrap solving of action potential models (done by Chaste) in Python to make MCMC easier by passing vectors of parameter values all over the place.

Assuming Chaste and CMake and all that have been installed as normal...

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

I'll add a specific example (probably after Eights...), but in the above example we've got the AP trace out of Chaste, which is all we need to start defining likelihood functions in Python with which to do MCMC etc.
