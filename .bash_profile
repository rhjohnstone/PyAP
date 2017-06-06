# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/bin

export PATH

if [[ `hostname -f` = *arcus.osc.local ]]
then
    echo ARCUS
    # this section contains all the commands to be run on arcus
    module load scons/2.3.4
    module load PETSc/openmpi-1.6.5/3.5_icc-2013 
    module load python/2.7
    module load vtk/5.8.0
    ### These should match with python/hostconfig/machines/arcus.py
    # Xerces
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/system/software/linux-x86_64/xerces-c/3.3.1/lib
    # Szip
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/system/software/linux-x86_64/lib/szip/2.1/lib
    # Boost
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/system/software/linux-x86_64/lib/boost/1_56_0/lib
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/include
    export CPATH=${LD_LIBRARY_PATH}:/usr/include
else
    echo
    echo ARCUS-B
    echo
    # this section contains all the commands to be run on arcusb
    module load cmake/3.8.0
    module load python/2.7
    module load vtk/5.10.1
    module unload intel-compilers/2013 intel-mkl/2013
    module load PETSc/mvapich2-2.0.1/3.5_icc-2015
    module load intel-mkl/2015
    module load hdf5-parallel/1.8.14_mvapich2_intel
    # The following libraries must be in the path to run executables
    export LD_LIBRARY_PATH=/system/software/linux-x86_64/xerces-c/3.3.1/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/linux-x86_64/lib/boost/1_58_0/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/linux-x86_64/lib/xsd/3.3.0-1/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/linux-x86_64/lib/szip/2.1/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/linux-x86_64/lib/vtk/5.10.1/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/arcus-b/lib/parmetis/4.0.3/mvapich2-2.0.1__intel-2015/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/system/software/arcus-b/lib/sundials/mvapich2-2.0.1/2.5.0/double/lib:$LD_LIBRARY_PATH

    # Add chaste libraries - you may need to change this depending on where you installed (or plan to install) Chaste
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DATA}/chaste-build/lib
fi
