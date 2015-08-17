#pragma once

#include "xrtTypes.hpp"
#include "Simulation.hpp"
#include "plugins/plugins.hpp"
#include "plugins/ISimulationPlugin.hpp"

#include <debug/VerboseLog.hpp>
#include <algorithms/ForEach.hpp>
#include <compileTime/conversion/MakeSeq.hpp>
#include <forward.hpp>
#include <boost/program_options/options_description.hpp>

namespace xrt{

    namespace po = boost::program_options;

    enum class ArgsErrorCode {SUCCESS, SUCCESS_EXIT, ERROR};

    class SimulationStarter
    {
        typedef std::list<po::options_description> BoostOptionsList;
        BoostOptionsList options;

        Simulation simulationClass;
        std::list<ISimulationPlugin*> plugins;

        template<typename T_Type>
        struct PushBack
        {

            template<typename T>
            void operator()(T& list)
            {
                list.push_back(new T_Type());
            }
        };
    public:

        SimulationStarter()
        {
            loadPlugins();
        }

        virtual ~SimulationStarter()
        {
            for(auto&& plugin: plugins)
                __delete(plugin);
            plugins.clear();
        }

        void start()
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            pluginConnector.loadPlugins();
            PMacc::log< XRTLogLvl::SIM_STATE >("Startup");
            simulationClass.startSimulation();
        }

        ArgsErrorCode parseConfigs(int argc, char **argv)
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();

            po::options_description simDesc(simulationClass.pluginGetName());
            simulationClass.pluginRegisterHelp(simDesc);
            options.push_back(simDesc);

            BoostOptionsList pluginOptions = pluginConnector.registerHelp();
            options.insert(options.end(), pluginOptions.begin(), pluginOptions.end());

            // parse environment variables, config files and command line
            return parse(argc, argv);
        }

        void load()
        {
            PMacc::log< XRTLogLvl::SIM_STATE >("Loading simulation");
            simulationClass.load();
            PMacc::log< XRTLogLvl::SIM_STATE >("Loading plugins");

            for(auto&& plugin: plugins)
                plugin->setMappingDesc(simulationClass.getMappingDesc());
            PMacc::log< XRTLogLvl::SIM_STATE >("Loading done");
        }

        void unload()
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            PMacc::log< XRTLogLvl::SIM_STATE >("Unloading plugins");
            pluginConnector.unloadPlugins();
            PMacc::log< XRTLogLvl::SIM_STATE >("Unloading simulation");
            simulationClass.unload();
            PMacc::log< XRTLogLvl::SIM_STATE >("Everything unloaded");
        }
    private:

        typedef bmpl::transform<
                SpeciesPlugins,
                bmpl::apply1<
                    bmpl::_1,
                    PIC_Photons
                >
            >::type SpecializedSpeciesPlugins;

        /* create sequence with all plugins*/
        typedef PMacc::MakeSeq<
            StandAlonePlugins,
            SpecializedSpeciesPlugins
        >::type AllPlugins;

        void loadPlugins()
        {
            PMacc::algorithms::forEach::ForEach< AllPlugins, PushBack<bmpl::_1> > pushBack;
            pushBack(PMacc::forward(plugins));
        }

        void printStartParameters(int argc, char **argv) const
        {
            std::cout << "Start Parameters: ";
            for (int i = 0; i < argc; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }

        ArgsErrorCode parse(int argc, char** argv) const
        {
            try
            {
                po::options_description desc("X-Ray tracing");

                desc.add_options()
                    ( "help,h", "print help message and exit" )
                    ( "validate", "validate command line parameters and exit" );

                // add all options from plugins
                for (auto&& option: options)
                    desc.add(option);

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
