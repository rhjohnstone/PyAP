#ifndef APSIMULATOR_HPP_
#define APSIMULATOR_HPP_

#include "AbstractCvodeCellWithDataClamp.hpp"  // currently not using data clamp anywhere
#include "OutputFileHandler.hpp"

class APSimulator
{
private:
    unsigned mModelNumber;
    boost::shared_ptr<AbstractCvodeCell> mpModel;
    boost::shared_ptr<AbstractStimulusFunction> mpStimulus;
    std::vector<std::string> mParameterMetanames;
    std::vector<double> mExptTimes;
    std::vector<double> mExptTrace;
    unsigned mNumberOfFailedSolves;
    unsigned mHowManySolves;
    double mSolveStart;
    double mSolveEnd;
    double mSolveTimestep;
    unsigned mNumTimePts;
    bool mHaveRunToSteadyState;
    double mStimPeriod;
public:
    APSimulator();
    ~APSimulator();
    //void RedirectStdErr();
    void DefineStimulus(double stimulus_magnitude, double stimulus_duration, double stimulus_period, double stimulus_start_time);
    void DefineSolveTimes(double solve_start, double solve_end, double solve_timestep);
    void DefineModel(unsigned model_number);
    std::vector<std::string> GetParameterMetanames();
    std::vector<double> SolveForVoltageTraceWithParams(const std::vector<double>& rParams);
    void SetTolerances(double rel_tol, double abs_tol);
    void SetExtracellularPotassiumConc( double extra_K_conc );
    void SetNumberOfSolves( unsigned num_solves );
    boost::shared_ptr<AbstractCvodeCell> GetModel();
    bool RunToSteadyState();
    //void ArchiveStateVariables();
    //void LoadStateVariables();
    void SetToModelInitialConditions();
};



#endif /*APSIMULATOR_HPP_*/
