#pragma once

#include "xrtTypes.hpp"
#include "Simulation.hpp"

#include <debug/VerboseLog.hpp>
#include <pluginSystem/IPlugin.hpp>
#include <boost/program_options/options_description.hpp>

namespace xrt{

    namespace po = boost::program_options;

    enum class ArgsErrorCode {SUCCESS, SUCCESS_EXIT, ERROR};

    class SimulationStarter: public PMacc::IPlugin
    {
        typedef std::list<po::options_description> BoostOptionsList;

        Simulation simulationClass;
    public:

        virtual ~SimulationStarter()
        {}

        virtual std::string pluginGetName() const
        {
            return "PIConGPU simulation starter";
        }

        virtual void start()
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            pluginConnector.loadPlugins();
            PMacc::log< XRTLogLvl::SIM_STATE > ("Startup");
            simulationClass.startSimulation();
        }

        ArgsErrorCode parseConfigs(int argc, char **argv)
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            BoostOptionsList options;

            po::options_description simDesc(simulationClass.pluginGetName());
            simulationClass.pluginRegisterHelp(simDesc);
            options.push_back(simDesc);

            BoostOptionsList pluginOptions = pluginConnector.registerHelp();
            options.insert(options.end(), pluginOptions.begin(), pluginOptions.end());

            // parse environment variables, config files and command line
            return parse(argc, argv, options);
        }

        /* Some required overrides that are not needed here */
        virtual void pluginRegisterHelp(po::options_description&)
        {}
        void notify(uint32_t)
        {}
        virtual void restart(uint32_t, const std::string)
        {}
        virtual void checkpoint(uint32_t, const std::string)
        {}
    protected:

        void pluginLoad()
        {
            simulationClass.load();
        }

        void pluginUnload()
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            pluginConnector.unloadPlugins();
            simulationClass.unload();
        }
    private:

        void printStartParameters(int argc, char **argv)
        {
            std::cout << "Start Parameters: ";
            for (int i = 0; i < argc; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }

        ArgsErrorCode parse(int argc, char** argv, const BoostOptionsList& options)
        {
            try
            {
                po::options_description desc("X-Ray tracing");

                desc.add_options()
                    ( "help,h", "print help message and exit" )
                    ( "validate", "validate command line parameters and exit" );

                // add all options from plugins
                for (auto iter = options.begin(); iter != options.end(); ++iter)
                    desc.add(*iter);

                // parse command line options and config file and store values in vm
                po::variables_map vm;
                po::store(boost::program_options::parse_command_line( argc, argv, desc ), vm);
                po::notify(vm);

                // print help message and quit simulation
                if ( vm.count( "help" ) )
                {
                    std::cerr << desc << "\n";
                    return ArgsErrorCode::SUCCESS_EXIT;
                }
                if ( vm.count( "validate" ) )
                    return ArgsErrorCode::SUCCESS_EXIT;
            }
            catch ( boost::program_options::error& e )
            {
                std::cerr << e.what() << std::endl;
                return ArgsErrorCode::ERROR;
            }

            return ArgsErrorCode::SUCCESS;
        }
    };
} // namespace xrt
