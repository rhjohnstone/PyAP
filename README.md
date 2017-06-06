# PyAP

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
cmake $CHASTE_SOURCE_DIR
make [-jN] chaste_project_PyAP
make [-jN] project_PyAP_Python
```

where N is the number of processors to do the build with.

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
