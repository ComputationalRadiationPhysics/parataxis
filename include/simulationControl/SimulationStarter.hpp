/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include "parataxisTypes.hpp"
#include "plugins/plugins.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "fields/DensityField.hpp"

#include <debug/VerboseLog.hpp>
#include <algorithms/ForEach.hpp>
#include <compileTime/conversion/MakeSeq.hpp>
#include <forward.hpp>
#include <boost/program_options/options_description.hpp>
#include <list>

namespace parataxis{

    namespace po = boost::program_options;

    enum class ArgsErrorCode {SUCCESS, SUCCESS_EXIT, ERROR};

    template<class T_Simulation>
    class SimulationStarter
    {
        typedef std::list<po::options_description> BoostOptionsList;
        BoostOptionsList options;

        T_Simulation simulationClass;
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
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Startup");
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
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Loading simulation");
            simulationClass.load();
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Loading plugins");

            for(auto&& plugin: Environment::get().PluginConnector().getPluginsFromType<ISimulationPlugin>())
                plugin->setMappingDesc(simulationClass.getMappingDesc());
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Loading done");
        }

        void unload()
        {
            PMacc::PluginConnector& pluginConnector = Environment::get().PluginConnector();
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Unloading plugins");
            pluginConnector.unloadPlugins();
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Unloading simulation");
            simulationClass.unload();
            PMacc::log< PARATAXISLogLvl::SIM_STATE >("Everything unloaded");
        }
    private:

        typedef bmpl::transform<
                SpeciesPlugins,
                bmpl::apply1<
                    bmpl::_1,
                    PIC_Photons
                >
            >::type SpecializedSpeciesPlugins;

        typedef bmpl::transform<
                FieldPlugins,
                bmpl::apply1<
                    bmpl::_1,
                    fields::DensityField
                >
            >::type SpecializedFieldPlugins;

        typedef bmpl::transform<
                DetectorPlugins,
                bmpl::apply1<
                    bmpl::_1,
                    Resolve_t<detector::PhotonDetector>
                >
            >::type SpecializedDetectorPlugins;

        /* create sequence with all plugins*/
        typedef PMacc::MakeSeq<
            StandAlonePlugins,
            SpecializedSpeciesPlugins,
            SpecializedFieldPlugins,
            SpecializedDetectorPlugins
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
} // namespace parataxis
