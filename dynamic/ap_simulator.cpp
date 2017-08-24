/*

Copyright (c) 2005-2016, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <string>
#include <iostream> // RJ, probably unnecessary
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include "APSimulator.hpp"
#include "PythonObjectConverters.hpp" // added by RJ

using namespace boost::python;

PyObject *myCPPExceptionType = NULL;

void translateMyCPPException(Exception const &e)
{
    assert(myCPPExceptionType != NULL);
    object pythonExceptionInstance(e);
    PyErr_SetObject(myCPPExceptionType, pythonExceptionInstance.ptr());
}

// Make the python module
BOOST_PYTHON_MODULE(_chaste_project_PyAP_ap_simulator)
{
    import_array(); // test
    numeric::array::set_module_and_type("numpy", "ndarray"); // test
    
    to_python_converter<std::vector<double>, StdVectorToNumpyArray<std::vector<double> > >();

    //class_<std::vector<double> > ("VecDouble")
    //     .def(vector_indexing_suite<std::vector<double> >())
    //;

    class_<APSimulator>("APSimulator")
        .def("DefineStimulus", &APSimulator::DefineStimulus)
        .def("DefineSolveTimes", &APSimulator::DefineSolveTimes)
        .def("DefineModel", &APSimulator::DefineModel)
        .def("SolveForVoltageTraceWithParams", &APSimulator::SolveForVoltageTraceWithParams)
        .def("SetTolerances", &APSimulator::SetTolerances)
        .def("SetExtracellularPotassiumConc", &APSimulator::SetExtracellularPotassiumConc)
        .def("SetIntracellularPotassiumConc", &APSimulator::SetIntracellularPotassiumConc)
        .def("SetExtracellularSodiumConc", &APSimulator::SetExtracellularSodiumConc)
        .def("SetIntracellularSodiumConc", &APSimulator::SetIntracellularSodiumConc)
        .def("SetNumberOfSolves", &APSimulator::SetNumberOfSolves)
        //.def("LoadStateVariables", &APSimulator::LoadStateVariables)
        .def("SetToModelInitialConditions", &APSimulator::SetToModelInitialConditions)
        .def("UseDataClamp", &APSimulator::UseDataClamp)
        .def("SetExperimentalTraceAndTimesForDataClamp", &APSimulator::SetExperimentalTraceAndTimesForDataClamp)
    ;

    PythonIterableToStl()
      .from_python<std::vector<double> >()
      ;
      
    class_<Exception> myCPPExceptionClass("CPPException", init<std::string, std::string, unsigned>());
    myCPPExceptionClass.add_property("GetMessage", &Exception::GetMessage);
    myCPPExceptionClass.add_property("GetShortMessage", &Exception::GetShortMessage);

    myCPPExceptionType = myCPPExceptionClass.ptr();
    register_exception_translator<Exception>(&translateMyCPPException);
      
}
