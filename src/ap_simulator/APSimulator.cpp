#include "APSimulator.hpp"

#include "RegularStimulus.hpp"
#include "SimpleStimulus.hpp"
#include "MultiStimulus.hpp"
#include "AbstractIvpOdeSolver.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include "BoostFilesystem.hpp"
#include "RandomNumberGenerator.hpp"

#include "CheckpointArchiveTypes.hpp"
#include "ArchiveLocationInfo.hpp"
#include "SteadyStateRunner.hpp"

#include "hodgkin_huxley_squid_axon_model_1952_modifiedCvode.hpp"
#include "beeler_reuter_model_1977Cvode.hpp"
#include "luo_rudy_1991Cvode.hpp"
#include "ten_tusscher_model_2004_epiCvode.hpp"
#include "ohara_rudy_2011_endoCvode.hpp"
#include "davies_isap_2012CvodeDataClamp.hpp"
//#include "paci_hyttinen_aaltosetala_severi_ventricularVersionCvodeDataClamp.hpp"
//#include "gokhale_ex293_2017Cvode.hpp"

APSimulator::APSimulator()
    : mModelNumber(1),
      mNumberOfFailedSolves(0),
      mHowManySolves(1),
      mSolveStart(0),
      mSolveEnd(500),
      mSolveTimestep(0.2),
      mNumTimePts(2501),
      mHaveRunToSteadyState(false),
      mUseDataClamp(false),
      mDataClampOn(0),
      mDataClampOff(0)
{
    std::cerr << "*** INSIDE CONSTRUCTOR (nothing should happen here) ***" << std::endl << std::flush;
}

APSimulator::~APSimulator()
{
}

void APSimulator::DefineStimulus(double stimulus_magnitude, double stimulus_duration, double stimulus_period, double stimulus_start_time)
{
    mpStimulus.reset(new RegularStimulus(stimulus_magnitude,stimulus_duration,stimulus_period,stimulus_start_time));
    mStimPeriod = stimulus_period;
}

void APSimulator::DefineSolveTimes(double solve_start, double solve_end, double solve_timestep)
{
    std::cerr << "About to define solve times" << std::endl << std::flush;
    mSolveStart = solve_start;
    mSolveEnd = solve_end;
    mSolveTimestep = solve_timestep;
    mNumTimePts = (solve_end - solve_start)/solve_timestep + 1;
    std::cerr << "Solve times defined" << std::endl << std::flush;
}

void APSimulator::DefineModel(unsigned model_number)
{
    mModelNumber = model_number;
    boost::shared_ptr<AbstractIvpOdeSolver> p_solver;
    if ( model_number == 1u )
    {
        mpModel.reset(new Cellhodgkin_huxley_squid_axon_model_1952_modifiedFromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance"); // 120
        mParameterMetanames.push_back("membrane_potassium_current_conductance");   // 36
        mParameterMetanames.push_back("membrane_leakage_current_conductance");     // 0.3
    }
    else if ( model_number == 2u )
    {
        mpModel.reset(new Cellbeeler_reuter_model_1977FromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                        // 4e-2
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");         // 0.0035
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");  // 0.008
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                     // 9e-4
    }
    else if ( model_number == 3u )
    {
        mpModel.reset(new Cellluo_rudy_1991FromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                  // 23
        mParameterMetanames.push_back("membrane_delayed_rectifier_potassium_current_conductance");  // 0.282
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");   // 0.6047
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");               // 0.09
        mParameterMetanames.push_back("membrane_leakage_current_conductance");                      // 0.03921
        mParameterMetanames.push_back("membrane_plateau_potassium_current_conductance");            // 0.0183
    }
    else if ( model_number == 4u )
    {
        mpModel.reset(new Cellten_tusscher_model_2004_epiFromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                        // 14.838
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                     // 0.000175
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");         // 5.405
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");  // 0.096
        mParameterMetanames.push_back("membrane_slow_delayed_rectifier_potassium_current_conductance");   // 0.245
        mParameterMetanames.push_back("membrane_sodium_calcium_exchanger_current_conductance");           // 1000
        mParameterMetanames.push_back("membrane_transient_outward_current_conductance");                  // 0.294
        mParameterMetanames.push_back("membrane_background_calcium_current_conductance");                 // 0.000592
        mParameterMetanames.push_back("membrane_background_sodium_current_conductance");                  // 0.00029
        mParameterMetanames.push_back("membrane_calcium_pump_current_conductance");                       // 0.825
        mParameterMetanames.push_back("membrane_potassium_pump_current_conductance");                     // 0.0146
        mParameterMetanames.push_back("membrane_sodium_potassium_pump_current_permeability");             // 1.362
    }
    else if ( model_number == 5u )
    {
        mpModel.reset(new Cellohara_rudy_2011_endoFromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                        // 75
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                     // 0.0001
        mParameterMetanames.push_back("membrane_background_potassium_current_conductance");               // 0.003
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");         // 0.1908
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");  // 0.046
        mParameterMetanames.push_back("membrane_slow_delayed_rectifier_potassium_current_conductance");   // 0.0034
        mParameterMetanames.push_back("membrane_sodium_calcium_exchanger_current_conductance");           // 0.0008
        mParameterMetanames.push_back("membrane_sodium_potassium_pump_current_permeability");             // 30
        mParameterMetanames.push_back("membrane_transient_outward_current_conductance");                  // 0.02
        mParameterMetanames.push_back("membrane_background_calcium_current_conductance");                 // 2.5e-8
        mParameterMetanames.push_back("membrane_background_sodium_current_conductance");                  // 3.75e-10
        mParameterMetanames.push_back("membrane_calcium_pump_current_conductance");                       // 0.0005
        mParameterMetanames.push_back("membrane_persistent_sodium_current_conductance");                  // 0.0075
    }
    else if ( model_number == 6u ) // Davies 2012 with data clamp
    {
        mpModel.reset(new Celldavies_isap_2012FromCellMLCvodeDataClamp(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                        // 8.25
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                     // 0.000243
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");         // 0.5
        mParameterMetanames.push_back("membrane_potassium_pump_current_conductance");                     // 0.00276
        mParameterMetanames.push_back("membrane_slow_delayed_rectifier_potassium_current_conductance");   // 0.00746925
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");  // 0.0138542
        mParameterMetanames.push_back("membrane_calcium_pump_current_conductance");                       // 0.0575
        mParameterMetanames.push_back("membrane_background_calcium_current_conductance");                 // 0.0000007980336
        mParameterMetanames.push_back("membrane_sodium_calcium_exchanger_current_conductance");           // 5.85
        mParameterMetanames.push_back("membrane_sodium_potassium_pump_current_permeability");             // 0.61875
        mParameterMetanames.push_back("membrane_transient_outward_current_conductance");                  // 0.1805
        mParameterMetanames.push_back("membrane_transient_outward_chloride_current_conductance");         // 4e-7
        mParameterMetanames.push_back("membrane_background_chloride_current_conductance");                // 0.000225
        mParameterMetanames.push_back("membrane_persistent_sodium_current_conductance");                  // 0.011
        std::cerr << "Defined mpModel as Davies with DC" << std::endl << std::flush;
    }
    /*else if ( model_number == 7u ) // Paci ventricular
    {
        mpModel.reset(new Cellpaci_hyttinen_aaltosetala_severi_ventricularVersionFromCellMLCvodeDataClamp(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                                            // 3671.2302
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                                         // 8.635702e-5
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");                             // 28.1492
        mParameterMetanames.push_back("membrane_slow_delayed_rectifier_potassium_current_conductance");                       // 2.041
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");                      // 29.8667
        mParameterMetanames.push_back("membrane_calcium_pump_current_conductance");                                           // 0.4125
        mParameterMetanames.push_back("membrane_sodium_calcium_exchanger_current_conductance");                               // 4900
        mParameterMetanames.push_back("membrane_background_calcium_current_conductance");                                     // 0.69264
        mParameterMetanames.push_back("membrane_sodium_potassium_pump_current_permeability");                                 // 1.841424
        mParameterMetanames.push_back("membrane_transient_outward_current_conductance");                                      // 29.9038
        mParameterMetanames.push_back("membrane_background_sodium_current_conductance");                                      // 0.9
        mParameterMetanames.push_back("membrane_hyperpolarisation_activated_funny_current_potassium_component_conductance");  // 30.10312
    }
    else if ( model_number == 8u ) // Gokhale 2017
    {
        mpModel.reset(new Cellgokhale_ex293_2017FromCellMLCvode(p_solver, mpStimulus));
        mParameterMetanames.push_back("transfected_sodium_current_conductance");  // 90.34
        mParameterMetanames.push_back("transfected_inward_rectifier_potassium_current_conductance");  // 6.609
        mParameterMetanames.push_back("endogenous_sodium_current_conductance");  // 0.6976
        mParameterMetanames.push_back("endogenous_potassium_current_conductance");  // 0.1332
        mParameterMetanames.push_back("transfected_sodium_fast_component_contribution");  // 0.9
        mParameterMetanames.push_back("transfected_potassium_fast_component_contribution");  // 0.75
    }*/
    /*else if ( model_number == 10u ) // Paci ventricular LINEARISED by RJ
    {
        mpModel.reset(new Cellpaci_hyttinen_aaltosetala_severi_ventricularVersion_linearisedFromCellMLCvodeDataClamp(p_solver, mpStimulus));
        mParameterMetanames.push_back("membrane_fast_sodium_current_conductance");                                            // 3671.2302
        mParameterMetanames.push_back("membrane_L_type_calcium_current_conductance");                                         // 8.635702e-5
        mParameterMetanames.push_back("membrane_inward_rectifier_potassium_current_conductance");                             // 28.1492
        mParameterMetanames.push_back("membrane_slow_delayed_rectifier_potassium_current_conductance");                       // 2.041
        mParameterMetanames.push_back("membrane_rapid_delayed_rectifier_potassium_current_conductance");                      // 29.8667
        mParameterMetanames.push_back("membrane_calcium_pump_current_conductance");                                           // 0.4125
        mParameterMetanames.push_back("membrane_sodium_calcium_exchanger_current_conductance");                               // 4900
        mParameterMetanames.push_back("membrane_background_calcium_current_conductance");                                     // 0.69264
        mParameterMetanames.push_back("membrane_sodium_potassium_pump_current_permeability");                                 // 1.841424
        mParameterMetanames.push_back("membrane_transient_outward_current_conductance");                                      // 29.9038
        mParameterMetanames.push_back("membrane_background_sodium_current_conductance");                                      // 0.9
        mParameterMetanames.push_back("membrane_hyperpolarisation_activated_funny_current_potassium_component_conductance");  // 30.10312
    }*/
    mpModel->SetMaxSteps(10000);
}

void APSimulator::UseDataClamp(double data_clamp_on, double data_clamp_off)
{
    mUseDataClamp = true;
    mDataClampOn = data_clamp_on;
    mDataClampOff = data_clamp_off;
}

void APSimulator::SetExperimentalTraceAndTimesForDataClamp(const std::vector<double>& expt_times, const std::vector<double>& expt_trace)
{
    boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->SetExperimentalData(expt_times, expt_trace);
}

std::vector<std::string> APSimulator::GetParameterMetanames()
{
    return mParameterMetanames;
}

void APSimulator::SetToModelInitialConditions()
{
    mpModel->SetStateVariables(mpModel->GetInitialConditions());
}

std::vector<double> APSimulator::SolveForVoltageTraceWithParamsNoDataClamp(const std::vector<double>& rParams)
{
    //std::cerr << "About to try and solve" << std::endl << std::flush;
    //mpModel->SetStateVariables(mpModel->GetInitialConditions());
    std::vector<double> voltage_trace;
    for (unsigned j=0; j<rParams.size(); j++)
    {
        mpModel->SetParameter(mParameterMetanames[j], rParams[j]);
    }
    //std::cerr << "Just set(ted) parameters" << std::endl << std::flush;
    try
    {
        if (mHowManySolves > 1)
        {
            for (unsigned i=0; i<mHowManySolves-1; i++)
            {
                mpModel->Compute(mSolveStart, mSolveStart+mStimPeriod, mSolveTimestep);
            }
        }
        //std::cerr << "About to actually try and solve" << std::endl << std::flush;
        OdeSolution sol1 = mpModel->Compute(mSolveStart, mSolveEnd, mSolveTimestep);
        //std::cerr << "Just solved" << std::endl << std::flush;
        voltage_trace = sol1.GetAnyVariable("membrane_voltage");
    }
    catch (Exception &e)
    {
        std::cerr << "WARNING: CVODE failed to solve with these parameters" << std::endl << std::flush;
        std::cerr << "error was " << e.GetShortMessage() << std::endl << std::flush;
        voltage_trace = std::vector<double>(mNumTimePts, 0.0);
        mNumberOfFailedSolves++;
    }
    return voltage_trace;
}

std::vector<double> APSimulator::SolveForVoltageTraceWithParamsWithDataClamp(const std::vector<double>& rParams)
{
    std::vector<double> voltage_trace;
    
    for (unsigned j=0; j<rParams.size(); j++)
    {
        mpModel->SetParameter(mParameterMetanames[j], rParams[j]);
    }
    //mpModel->SetStateVariable("membrane_voltage",mExptTrace[0]);

    boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->TurnOffDataClamp();
    
    
    if (mHowManySolves > 1)
    {
        for (unsigned i=0; i<mHowManySolves-1; i++)
        {
            mpModel->ResetSolver();
            mpModel->Compute(mSolveStart, mDataClampOn, mSolveTimestep);
            mpModel->ResetSolver();
            boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->TurnOnDataClamp(200);
            mpModel->Compute(mDataClampOn, mDataClampOff, mSolveTimestep);
            mpModel->ResetSolver();
            boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->TurnOffDataClamp();
            mpModel->Compute(mDataClampOff, mSolveStart+mStimPeriod, mSolveTimestep);
        }
    }
    
    mpModel->ResetSolver();
    OdeSolution sol1 = mpModel->Compute(mSolveStart, mDataClampOn, mSolveTimestep);
    
    mpModel->ResetSolver();
    boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->TurnOnDataClamp(200);
    OdeSolution sol2 = mpModel->Compute(mDataClampOn, mDataClampOff, mSolveTimestep);
    mpModel->ResetSolver();
    boost::static_pointer_cast<AbstractCvodeCellWithDataClamp>(mpModel)->TurnOffDataClamp();
    OdeSolution sol3 = mpModel->Compute(mDataClampOff, mSolveEnd, mSolveTimestep);

    voltage_trace = sol1.GetAnyVariable("membrane_voltage");
    std::vector<double> voltage_trace_part_2 = sol2.GetAnyVariable("membrane_voltage");
    std::vector<double> voltage_trace_part_3 = sol3.GetAnyVariable("membrane_voltage");

    voltage_trace.erase(voltage_trace.end()-1);
    voltage_trace.insert( voltage_trace.end(), voltage_trace_part_2.begin(), voltage_trace_part_2.end() );
    voltage_trace.erase(voltage_trace.end()-1);
    voltage_trace.insert( voltage_trace.end(), voltage_trace_part_3.begin(), voltage_trace_part_3.end() );
    return voltage_trace;
}

std::vector<double> APSimulator::SolveForVoltageTraceWithParams(const std::vector<double>& rParams)
{
    std::vector<double> voltage_trace;
    if (mUseDataClamp == false)
    {
        voltage_trace = SolveForVoltageTraceWithParamsNoDataClamp(rParams);
    }
    else
    {
        voltage_trace = SolveForVoltageTraceWithParamsWithDataClamp(rParams);
    }
    return voltage_trace;
}  

void APSimulator::SetTolerances(double rel_tol, double abs_tol)
{
    mpModel->SetTolerances(rel_tol, abs_tol);
}

void APSimulator::SetExtracellularPotassiumConc( double extra_K_conc )
{
    try
    {
        mpModel->SetParameter("extracellular_potassium_concentration", extra_K_conc);
    }
    catch (Exception &e)
    {
        std::cerr << "No extracellular_potassium_concentration parameter in model" << std::endl << std::flush;
        std::cerr << "error was " << e.GetShortMessage() << std::endl << std::flush;
    }
}

void APSimulator::SetIntracellularPotassiumConc( double intra_K_conc )
{
    try
    {
        mpModel->SetParameter("cytosolic_potassium_concentration", intra_K_conc);
    }
    catch (Exception &e)
    {
        std::cerr << "No cytosolic_potassium_concentration parameter in model" << std::endl << std::flush;
        std::cerr << "error was " << e.GetShortMessage() << std::endl << std::flush;
    }
}

void APSimulator::SetExtracellularSodiumConc( double extra_Na_conc )
{
    try
    {
        mpModel->SetParameter("extracellular_sodium_concentration", extra_Na_conc);
    }
    catch (Exception &e)
    {
        std::cerr << "No extracellular_sodium_concentration parameter in model" << std::endl << std::flush;
        std::cerr << "error was " << e.GetShortMessage() << std::endl << std::flush;
    }
}

void APSimulator::SetIntracellularSodiumConc( double intra_Na_conc )
{
    try
    {
        mpModel->SetParameter("cytosolic_sodium_concentration", intra_Na_conc);
    }
    catch (Exception &e)
    {
        std::cerr << "No cytosolic_sodium_concentration parameter in model" << std::endl << std::flush;
        std::cerr << "error was " << e.GetShortMessage() << std::endl << std::flush;
    }
}

void APSimulator::SetNumberOfSolves( unsigned num_solves )
{
    mHowManySolves = num_solves;
}

bool APSimulator::RunToSteadyState()
{
    SteadyStateRunner steady_runner(mpModel);
    unsigned max_paces = 50000u;
    steady_runner.SetMaxNumPaces(max_paces);
    bool result = steady_runner.RunToSteadyState();
    mHaveRunToSteadyState = result;
    return result;
}

/*void APSimulator::ArchiveStateVariables()
{
    boost::filesystem::path arch_dir;
    if (mHaveRunToSteadyState)
    {
        arch_dir = "projects/RossJ/archived_variables/steady_state/";
    }
    else
    {
        arch_dir = "projects/RossJ/archived_variables/non_steady_state/";
    }
    boost::filesystem::create_directories(arch_dir);
    //OutputFileHandler handler(arch_dir.string(), false);
    //handler.SetArchiveDirectory();
    //std::string arch_name = handler.GetOutputDirectoryFullPath() + "m_"+boost::lexical_cast<std::string>(mModelNumber)+".arch";
    std::ofstream ofs((arch_dir.string() + "m_"+boost::lexical_cast<std::string>(mModelNumber)+".arch").c_str());
    boost::archive::text_oarchive output_arch(ofs);
    output_arch <<  *mpModel;
}

void APSimulator::LoadStateVariables()
{
    std::string arch_file = "~/workspace/RossJ/archived_variables/steady_state/m_"+boost::lexical_cast<std::string>(mModelNumber)+".arch";
    std::cout << arch_file << std::endl << std::flush;
    std::ifstream ifs(arch_file.c_str(), std::ios::binary);
    boost::archive::text_iarchive input_arch(ifs);
    input_arch >> *mpModel;
}*/

