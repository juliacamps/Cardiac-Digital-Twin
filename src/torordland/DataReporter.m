%     Cardiac model ToR-ORd
%     Copyright (C) 2019 Jakub Tomek. Contact: jakub.tomek.mff@gmail.com
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.

classdef DataReporter
    %DATAREPORTER A class for running the evaluation pipeline (plus some
    %general helper functions such as APD extraction)
    %

    methods (Static)
        
        function runEvaluationSimulations(switches, folderOut, fnameSave, reportParams )
            % This function runs a broad range of simulations that then
            % enable generating figures and reports describing calibration,
            % validation, or both. Only Matlab-based calibration and
            % validation criteria are included here. Do not run this in
            % parfor, as this function itself is using parfor.
            %
            % IN:
            % switches - a structure of switches that can modify the
            % behaviour of the reporter. At the moment, this can be: 1) structure.parameter -a
            % structure of parameters as would be passed to modelRunner (it
            % can contain any current multipliers etc.), 2) structure.X0,
            % changing the starting state.
            %
            % folderOut - folder into which are stored intermediate results
            % of the evaluation
            %
            % fnameSave - name of the file which contains the results of
            % the evaluation
            %
            % reportParams - a structure of parameters determining the
            % character of the report that will be produced. This includes
            % 1) reportParams.dynamicRestitution ('full' = default, or
            % 'small'), differing in how many bcls are simulated for
            % alternans/dynamic restitution. The 'small' may be useful
            % e.g., when exploring some parameters, checking what it does
            % to calibration criteria, and many reports need to be
            % generated.
            % Other fields of this structure are used in makeReport.
            % 
            % 
            %
            % Note - do not run this in parfor as it's running a parfor itself.
            %
            
            if ( isfield(reportParams, 'dynamicRestitution')) % if "small" dynRest, then  fewer bcls are tested and for less beats.
                typeDynRest = reportParams.dynamicRestitution;
            else
                typeDynRest = 'full';
            end

            APDaccommodation = 1; % do measure this
     
            
            mkdir(folderOut);
            
            %% Simulation of dynamic restitution/alternans
            if (strcmp(typeDynRest, 'full'))
                bclRange = [220:20:400 500 600 750 1000 2000];
            elseif (strcmp(typeDynRest, 'small'))
                bclRange = [260 400 750 1000 2000];
            end
            %  Initialisation
            X0 = getStartingState('Torord_endo');
            
            beatsDynamic = 1000;%number of beats in the simulation
            if (strcmp(typeDynRest, 'small'))
                beatsDynamic = 500;
            end
            
            bcl = 1000;
            options=[];%options for ode solver
            
            
            parameter.bcl = bcl;
            parameter.cellType = 0;
            parameter.model = @model12;
            
            
            if (isfield(switches, 'parameter'))
                parameter = switches.parameter;
            end
            
            if (isfield(switches, 'X0'))
                X0 = switches.X0;
            end
            
            
            % and if certain current multipliers are not defined, we set
            % them to 1 (control value) - otherwise the code crashes.
            if(~isfield(parameter, 'INa_Multiplier')), parameter.INa_Multiplier = 1;end
            if(~isfield(parameter, 'INaL_Multiplier')), parameter.INaL_Multiplier = 1;end
            if(~isfield(parameter, 'ICaL_Multiplier')), parameter.ICaL_Multiplier = 1;end
            if(~isfield(parameter, 'IK1_Multiplier')), parameter.IK1_Multiplier = 1;end
            if(~isfield(parameter, 'IKr_Multiplier')), parameter.IKr_Multiplier = 1;end
            if(~isfield(parameter, 'IKs_Multiplier')), parameter.IKs_Multiplier = 1;end
            if(~isfield(parameter, 'nao')), parameter.nao = 140;end
            if(~isfield(parameter, 'cao')), parameter.cao = 1.8;end
            
            
            
            
            % parameters are created - one set for each bcl. Added to the end is one
            % parameter combo where bcl is 1000 and sodium is 50% blocked.
            % ICa block and IK1 block also done
            
            parameters(1:length(bclRange)+3) = parameter;
            
            
            for i = 1:length(bclRange)
                parameters(i).bcl = bclRange(i);
                
            end
            parameters(length(bclRange)+1).bcl = 1000;
            parameters(length(bclRange)+2).bcl = 1000;
            parameters(length(bclRange)+3).bcl = 1000;
            % INa+INaL block
            parameters(length(bclRange)+1).INa_Multiplier = parameters(length(bclRange)+1).INa_Multiplier * 0.5;
            parameters(length(bclRange)+1).INaL_Multiplier = parameters(length(bclRange)+1).INaL_Multiplier * 0.5;
            
            % ICa block and IK1 block
            parameters(length(bclRange)+2).ICaL_Multiplier = parameters(length(bclRange)+2).ICaL_Multiplier * 0.5;
            parameters(length(bclRange)+3).IK1_Multiplier = parameters(length(bclRange)+3).IK1_Multiplier * 0.5;
            
            ignoreFirst = beatsDynamic-2; % last 1 AP stored
            parfor i = 1:length(parameters)
                disp(i);
                tic
                [timeDynamic{i}, XDynamic{i}] = modelRunner(X0, options, parameters(i), beatsDynamic, ignoreFirst);
                currentsDynamic{i}= getCurrentsStructure(timeDynamic{i}, XDynamic{i}, parameters(i), 0);
                toc
            end
            
            
            %% Drug block behaviour
            % first, 1st drug at 3 pacing rates, then 2nd, etc.
            %1 uM E-4031 (70% IKr block), 1 uM HMR-1556 (90% IKs block), 1 uM nisoldipine (90% ICaL block), 10 uM mexiletine (54% INaL, 9% IKr, and 20% ICaL block)
            paramsBlock(1:12) = parameter;
            for i = 1:3:12
                paramsBlock(i).bcl = 500;
                paramsBlock(i+1).bcl = 1000;
                paramsBlock(i+2).bcl = 2000;
            end
            for i = 1:3
                paramsBlock(i).IKr_Multiplier = paramsBlock(i).IKr_Multiplier * 0.3;
            end
            for i = 4:6
                paramsBlock(i).IKs_Multiplier = paramsBlock(i).IKs_Multiplier * 0.1;
            end
            for i = 7:9
                paramsBlock(i).ICaL_Multiplier = paramsBlock(i).ICaL_Multiplier * 0.1;
            end
            
            for i = 10:12
                paramsBlock(i).INaL_Multiplier = paramsBlock(i).INaL_Multiplier * 0.46;
                paramsBlock(i).IKr_Multiplier = paramsBlock(i).IKr_Multiplier * 0.91;
                paramsBlock(i).ICaL_Multiplier = paramsBlock(i).ICaL_Multiplier * 0.8;
            end
            
            ignoreFirst = beatsDynamic-2; % last 1 AP stored
            parfor i = 1:length(paramsBlock)
                disp(i);
                tic
                [timeDrugBlock{i}, XDrugBlock{i}] = modelRunner(X0, options, paramsBlock(i), beatsDynamic, ignoreFirst);
                currentsDrugBlock{i}= getCurrentsStructure(timeDrugBlock{i}, XDrugBlock{i}, paramsBlock(i), 0);
                toc
            end
            
            %% Simulation of EADs at 15% IKr availability
            paramsEADs = parameter;
            
            paramsEADs.bcl = 4000;
            paramsEADs.nao = 137;
            paramsEADs.cao = 2;
            paramsEADs.IKr_Multiplier = paramsEADs.IKr_Multiplier * 0.15;
            
            ignoreFirst = beatsDynamic-1; % last 1 AP stored
            
            tic
            [timeEADs, XEADs] = modelRunner(X0, options, paramsEADs, beatsDynamic, ignoreFirst);
            currentsEADs= getCurrentsStructure(timeEADs, XEADs, paramsEADs, 0);
            toc
            
            %% S1-S2 simulation (single cell)
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            parameterS1S2_1000 = parameters(i1000);
            startingStateS1S2_1000 = XDynamic{i1000}{2}(end,:); % -2 to have penultimate bcl and without INa block
            
            S2interval = [140:5:500 600:100:1500];
            
            for i = 1:length(S2interval)
                parameterS1S2_1000.bcl = S2interval(i);
                [timeS1S2_1000{i}, XS1S2_1000{i}] = modelRunner(startingStateS1S2_1000, options, parameterS1S2_1000, 2, 0); % 2 beats - one as S1, one as S2.
                currentsS1S2_1000{i}= getCurrentsStructure(timeS1S2_1000{i}, XS1S2_1000{i}, parameterS1S2_1000, 0);
            end
            
            %% Accommodation to pacing rate change
            %e segments: 10 beats at 750, then 3 minutes at 480 or 410, then 3 minutes back to 750.
            [~, i750] = find(bclRange == 750, 1, 'first');
            if (APDaccommodation == 1)
                parameter750 = parameters(i750);
                start750State = XDynamic{i750}{2}(end,:);
                beatsAdaptationSegment1 = 10;
                [timeAdaptationSegment1, XAdaptationSegment1] = modelRunner(start750State, options, parameter750, beatsAdaptationSegment1, 0);
                
                stateAdaptationStartSegment2 = XAdaptationSegment1{end}(end, :);
                parameter480 = parameter750; parameter480.bcl = 480;
                parameter410 = parameter750; parameter410.bcl = 410;
                nBeats480 = 375;
                nBeats410 = 439;
                
                % Simulating both versions of Segment 2:
                [timeAdaptationSegment2_480, XAdaptationSegment2_480] = modelRunner(stateAdaptationStartSegment2, options, parameter480, nBeats480, 0);
                [timeAdaptationSegment2_410, XAdaptationSegment2_410] = modelRunner(stateAdaptationStartSegment2, options, parameter410, nBeats410, 0);
                
                % And doing Segment 3
                stateAfter480 = XAdaptationSegment2_480{end}(end,:);
                stateAfter410 = XAdaptationSegment2_410{end}(end,:);
                
                nBeats750 = 180000/750;
                [timeAdaptationSegment2_480to750, XAdaptationSegment2_480to750] = modelRunner(stateAfter480, options, parameter750, nBeats750, 0);
                [timeAdaptationSegment2_410to750, XAdaptationSegment2_410to750] = modelRunner(stateAfter410, options, parameter750, nBeats750, 0);
                % Data are added together and trimmed to have only membrane potential to
                % save space.
                time480 = [timeAdaptationSegment1;timeAdaptationSegment2_480;timeAdaptationSegment2_480to750];
                time410 = [timeAdaptationSegment1;timeAdaptationSegment2_410;timeAdaptationSegment2_410to750];
                X480 = [XAdaptationSegment1;XAdaptationSegment2_480;XAdaptationSegment2_480to750];
                X410 = [XAdaptationSegment1;XAdaptationSegment2_410;XAdaptationSegment2_410to750];
                
                adaptationAPDs480 = zeros(length(time480),1);
                adaptationAPDs410 = zeros(length(time410),1);
                
                for i = 1:length(time480)
                    adaptationAPDs480(i) = DataReporter.getAPD(time480{i}, X480{i}(:,1), 0.9);
                end
                
                for i = 1:length(time410)
                    adaptationAPDs410(i) = DataReporter.getAPD(time410{i}, X410{i}(:,1), 0.9);
                end
                
                adaptationTimes480 = [0:750:(9*750),   10*750+(0:480:((nBeats480-1) * 480)), 10*750+(nBeats480 * 480) + (0:750:(nBeats750-1) * 750) ];
                adaptationTimes410 = [0:750:(9*750),   10*750+(0:410:((nBeats410-1) * 410)), 10*750+(nBeats410 * 410) + (0:750:(nBeats750-1) * 750) ];
            else
                adaptationTimes480 = -1;
                adaptationTimes410 = -1;
            end
            
            
            
            save([folderOut '/' fnameSave]);
        end
        
        function outputs = makeReport(fileSimulation, fileExperiment, folderOut, reportParams)
            % This function visualizes the range of simulations produced by
            % DataReporter.runEvaluationSimulations.
            %
            % IN:
            % fileSimulation - path to file (fnameSave) produced by
            % runEvaluationSimulations
            %
            % fileExperiment - a file with relevant experimental data - in
            % this case, it's usually oliTraces.mat, which contains data on
            % human AP morphology.
            %
            % folderOut - folder in which is the evaluation report
            % generated
            %
            % 
            % reportParams - a structure of parameters determining the
            % character of the report that will be produced. This includes
            % 1) reportParams.dynamicRestitution ('full' = default, or
            % 'small')
            % 2) reportParams.reportType, which may be 'calibration',
            % 'validation', or 'full' (corresponding to both calibration
            % and validation), depending on which criteria are to be shown
            % in the report.
            % 
            % OUT:
            % outputs - a structure containing key biomarkers and error
            % codes of various tests that are run (these are also shown in
            % the report, by icon alongside each feature). The codes are:
            % 1 = PASS, 2 = FAIL, 3 = N/A (is not automatically evaluated).
            %
            
            folderCalibration = [folderOut '/calibration/'];
            folderValidation = [folderOut '/validation/'];
            
            mkdir(folderCalibration);
            mkdir([folderCalibration '/traces/']);
            mkdir(folderValidation);
            
            if ( isfield(reportParams, 'reportType')) % 'calibration','validation', or 'full'
                reportType = reportParams.reportType;
            else
                reportType = 'full';
            end            
            
            
            if ( isfield(reportParams, 'dynamicRestitution'))
                typeDynRest = reportParams.dynamicRestitution;
            else
                typeDynRest = 'full';
            end
         
            APDaccommodation = 1; % do measure this by default
            
            % branching of whether many or just several pacing frequencies
            % are used in alternans and dynamic restitution assessment.
            if (strcmp(typeDynRest, 'full'))
                bclRange = [200:20:400 500 600 750 1000 2000];
            elseif (strcmp(typeDynRest, 'small'))
                bclRange = [260 400 600 750 1000 2000];
            end
            
            %% First, action potential morphology is assessed
            folderOutBackup = folderOut;
            reportParamsBackup = reportParams;
            fs=14;
            load(fileSimulation);
            folderOut = folderOutBackup; % This variable gets overwritten by loading at the previus line, so we restore it.
            reportParams = reportParamsBackup;
            
            dataSzeged = load(fileExperiment);
            
            figure(1); clf;
            
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            time = currentsDynamic{i1000}.time;
            V = currentsDynamic{i1000}.V;
            
            dataSzeged.referenceTime=dataSzeged.referenceTime - 8.6;
            h0 = gcf;
            hold on
            gDark = (1 + 1.5*[26,147,111]/255)/2.5;
            gLight = (1 + 1.5*[136,212,152]/255)/2.5;
            % hold on
            x = [dataSzeged.referenceTime, fliplr(dataSzeged.referenceTime)];        % repeat x values
            yy = [dataSzeged.q90Trace, fliplr(dataSzeged.q10Trace)];   % vector of upper & lower boundaries
            h2=fill(x,yy,gLight, 'EdgeColor','none');    % fill area defined by x & yy
            set(gca, 'ColorOrder', circshift(get(gca, 'ColorOrder'), numel(h0)))
            x = [dataSzeged.referenceTime, fliplr(dataSzeged.referenceTime)];        % repeat x values
            yy = [dataSzeged.q75Trace, fliplr(dataSzeged.q25Trace)];   % vector of upper & lower boundaries
            h=fill(x,yy,gDark,  'EdgeColor','none');    % fill area defined by x & yy
            set(gca, 'ColorOrder', circshift(get(gca, 'ColorOrder'), numel(h0)))
            plot(time, V, 'LineWidth',3);
            xlabel('time (ms)', 'FontSize',fs); ylabel('membrane potential (mV)', 'FontSize',fs);
            hold off
            xlim([0 500]);
            set(gca, 'FontSize',fs);
            legend({'data 10-90 percentile', 'data 25-75 percentile', 'Model'}, 'FontSize',fs);
            
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 20 20]);
            saveas(gcf, [folderCalibration '/apMorphology.png']);
            
            % Scoring the fit to data - PASS if within 10-90 quantile. This
            % is done from 10 ms (to ignore peak) to 500 ms (no data beyond that), and using linear
            % interpolation of data.
            V = V((time>10)&(time<500));
            time = time((time>10)&(time<500));
            isWithin10_90 = true;
            for i = 1:length(time)
                timeNow = time(i);
                VNow = V(i);
                
                iTimePre = DataReporter.binarySearch(dataSzeged.referenceTime, timeNow);
                iTimePost = iTimePre + 1;
                
                tPre = dataSzeged.referenceTime(iTimePre);
                tPost = dataSzeged.referenceTime(iTimePost);
                vPre10 = dataSzeged.q10Trace(iTimePre);
                vPost10 = dataSzeged.q10Trace(iTimePost);
                vPre90 = dataSzeged.q90Trace(iTimePre);
                vPost90 = dataSzeged.q90Trace(iTimePost);
                dt = tPost-tPre;
                tInterp = timeNow - tPre;
                dv10 = vPost10 - vPre10;
                dv90 = vPost90 - vPre90;
                v10 = vPre10 + tInterp/dt * dv10; % interpolated mempot of 10-percentile
                v90 = vPre90 + tInterp/dt * dv90; % interpolated mempot of 10-percentile
                if (VNow < v10) || (VNow > v90)
                    isWithin10_90=false;
                    break;
                end
            end
            
            if (isWithin10_90)
                score.apMorphology = 1;
            else
                score.apMorphology = 2;
            end
            saveas(gcf, [folderCalibration '/apMorphology.png']);
            %% Examples of traces of AP and CaT
            
            [~, i400] = find(bclRange == 400, 1, 'first');
            [~, i750] = find(bclRange == 750, 1, 'first');
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            indicesPlotted = [i400 i750 i1000 ];
            legendContents = {'400', '750', '1000'};
            
            figure(2); clf
            hold on;
            for i = indicesPlotted
                plot(currentsDynamic{i}.time, currentsDynamic{i}.V, 'LineWidth',1.5); xlim([0 400]);
            end
            hold off
            legend(legendContents);
            xlabel('time (ms)');
            ylabel('membrane potential');
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 20 20]);
            saveas(gcf, [folderCalibration '/examples_AP.png']);
            
            figure(11); clf
            hold on;
            for i = indicesPlotted
                plot(currentsDynamic{i}.time, currentsDynamic{i}.Cai, 'LineWidth',1.5); xlim([0 400]);
            end
            hold off
            legend(legendContents);
            xlabel('time (ms)');
            ylabel('Cai');
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 20 20]);
            saveas(gcf, [folderCalibration '/examples_Cai.png']);
            
            score.exampleTraces = 3;
            
            %% Calcium transient properties - mainly time to peak and duration at 90% recovery
            figure(3); clf
            hold on
            patch([50;60;60;50],[0.5, 0.5, 4, 4]*1e-4, [0.5,0.5,0.75] ); %text(65, 3.95e-4, 'time to peak', 'FontSize',12)
            patch([316+55;408+55;408+55;316+55],[0.5, 0.5, 4, 4]*1e-4, [0.5,0.75,0.5] ); % +55 because it's from peak to the given level of repolarisation. 316-408 is the interval of time from peak to 90% recovery in Coppini; this includes addition of 8 ms for which bottom 10% of CaT is rising at the start of the AP and this shouldn't included in CaTD90
            
            plot(currentsDynamic{i1000}.time, currentsDynamic{i1000}.Cai, 'LineWidth', 2);            
            catDur = DataReporter.getAPD(currentsDynamic{i1000}.time, currentsDynamic{i1000}.Cai, 0.9);
            
            [~, iMaxCaT] = max(currentsDynamic{i1000}.Cai(currentsDynamic{i1000}.time<1000)); % maximum of first beat
            catTTP = currentsDynamic{i1000}.time(iMaxCaT);
            
            thresh90New = min(currentsDynamic{14}.Cai) + 0.1 * peak2peak(currentsDynamic{14}.Cai);

            plot([catDur,catDur], [0.9 1.1]*thresh90New, 'k','LineWidth',3)
            set(gca, 'FontSize',fs);
            legend('time to peak range','CaTD90% range','Model calcium transient','CaTD90');
            hold off
            xlabel('time (ms)', 'FontSize',fs);
            ylabel('Ca^{2+} (mM)', 'FontSize',fs);
            xlim([0 1000]);
            saveas(gcf, [folderCalibration '/CaT.png']);
            
%             if 
            if (catTTP>50)&&(catTTP<60)
                score.catTTP = 1;
            else
                score.catTTP = 2;
            end
            if (catDur>371)&&(catDur<463)
                score.catDur90 = 1;
            else
                score.catDur90 = 2;
            end
            
            score.CaT = max(score.catTTP,score.catDur90); % summary score
            
            %% Sodium current block - for full rate-dependence, this is 15 vs 17
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            i1000INaB = i1000 + 2;
            controlTime = currentsDynamic{i1000}.time;
            controlCai = currentsDynamic{i1000}.Cai;
            controlV = currentsDynamic{i1000}.V;
            
            inabTime = currentsDynamic{i1000INaB}.time;
            inabCai = currentsDynamic{i1000INaB}.Cai;
            inabV = currentsDynamic{i1000INaB}.V;
            
            figure(4); clf
            subplot(1,2,1);
            plot(controlTime, controlV, inabTime, inabV, 'LineWidth',1.5); legend('Control','50% INa&INaL block'); xlabel('time (ms)'); ylabel('membrane potential'); xlim([0 500])
            subplot(1,2,2);
            plot(controlTime, controlCai, inabTime, inabCai, 'LineWidth',1.5); legend('Control','50% INa&INaL block'); xlabel('time (ms)'); ylabel('Cai'); xlim([0 500])
            
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 15]); %
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderCalibration '/inaBlock.png']);
            
            
            outputs.catControl1000 = max(controlCai) - min(controlCai);
            outputs.catINaB1000 = max(inabCai) - min(inabCai);
            
            % Scoring - 1 if 50% block does a reduction, 3 otherwise
            if (outputs.catINaB1000/outputs.catControl1000 < 1)
                score.INaBlock = 1;
            else
                score.INaBlock = 2;
            end
            %% ICaL block
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            i1000ICaLB = i1000+3;
            controlTime = currentsDynamic{i1000}.time;
            controlCai = currentsDynamic{i1000}.Cai;
            controlV = currentsDynamic{i1000}.V;
            
            icalbTime = currentsDynamic{i1000ICaLB}.time;
            icalbCai = currentsDynamic{i1000ICaLB}.Cai;
            icalbV = currentsDynamic{i1000ICaLB}.V;
            
            outputs.sim_apd90 = DataReporter.getAPD(controlTime, controlV, 0.9);
            outputs.sim_apd90_icalBlock = DataReporter.getAPD(icalbTime, icalbV, 0.9);
            
            % Scoring - 1 if 50% block does a reduction of APD, 2 otherwise
            if (outputs.sim_apd90_icalBlock < outputs.sim_apd90)
                score.ICaLBlock = 1;
            else
                score.ICaLBlock = 2;
            end
            
            
            figure(5); clf
            subplot(1,2,1);
            plot(controlTime, controlV, icalbTime, icalbV, 'LineWidth',1.5); legend('Control','50% ICaL block'); xlabel('time (ms)'); ylabel('membrane potential'); xlim([0 500])
            title(['Control APD: ' num2str(outputs.sim_apd90) ', block APD: ' num2str(outputs.sim_apd90_icalBlock)]);
            subplot(1,2,2);
            plot(controlTime, controlCai, icalbTime, icalbCai, 'LineWidth',1.5); legend('Control','50% ICaL block'); xlabel('time (ms)'); ylabel('Cai'); xlim([0 500])
            
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 15]); %
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderCalibration '/icalBlock.png']);
          
            %% IK1 block
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            i1000IK1B = i1000+4;
            
            controlTime = currentsDynamic{i1000}.time;
            controlV = currentsDynamic{i1000}.V;
            
            ik1bTime = currentsDynamic{i1000IK1B}.time;
            ik1bV = currentsDynamic{i1000IK1B}.V;
            
            outputs.sim_resting1000Control = controlV(end);
            outputs.sim_resting1000IK1Block = ik1bV(end);
            % Scoring - 1 if 50% block does a reduction, 2 otherwise
            if (outputs.sim_resting1000Control < outputs.sim_resting1000IK1Block)
                score.IK1Block = 1;
            else
                score.IK1Block = 2;
            end
            
            figure(6); clf
            plot(controlTime, controlV, ik1bTime, ik1bV, 'LineWidth',1.5); legend('Control','50% IK1 block'); xlabel('time (ms)'); ylabel('membrane potential'); xlim([0 500])
            title(['Control resting pot.: ' num2str(outputs.sim_resting1000Control) ', block resting pot.: ' num2str(outputs.sim_resting1000IK1Block)]);
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 20]); %
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderCalibration '/ik1Block.png']);
            
            %% EAD assessment - an EAD is detected if dV/dt > 0.05 following 50th ms.
       
            figure(7); clf
            hold on
            
            plot(currentsEADs.time, currentsEADs.V, 'LineWidth',2)
            
            hold off
            xlim([0 1500]);
            %             legend('0.1','0.15','0.3');
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 20]); %
            saveas(gcf, [folderCalibration '/eadFormation.png']);
            
            vPost50 = currentsEADs.V(currentsEADs.time > 50);
            tPost50 = currentsEADs.time(currentsEADs.time > 50);
            max_dvdt = max(diff(vPost50)./diff(tPost50));
            thresholdEAD = 0.05;
            
            
            if (max_dvdt > thresholdEAD)
                score.EADs = 1;
            else
                score.EADs = 2;
            end
            %% alternans~also dynamic restitution, Ca levels, 
            % showing min/max APD and min/max CaT versus rate
            minAPD90 = nan(size(bclRange));
            maxAPD90 = nan(size(bclRange));
            minCaT = nan(size(bclRange));
            maxCaT = nan(size(bclRange));
            minCa = nan(size(bclRange));
            maxCa = nan(size(bclRange));
            minJSRDepletion = nan(size(bclRange));
            maxJSRDepletion = nan(size(bclRange));
            
            for i = 1:length(bclRange)
                tDyn = currentsDynamic{i}.time; tDyn1 = tDyn(tDyn<bclRange(i));  tDyn2 = tDyn(tDyn>=bclRange(i));
                vDyn = currentsDynamic{i}.V; vDyn1 = vDyn(tDyn<bclRange(i));  vDyn2 = vDyn(tDyn>=bclRange(i));
                caiDyn = currentsDynamic{i}.Cai; caiDyn1 = caiDyn(tDyn<bclRange(i));  caiDyn2 = caiDyn(tDyn>=bclRange(i));
                cajsrDyn = currentsDynamic{i}.CaJSR; cajsrDyn1 = cajsrDyn(tDyn<bclRange(i));  cajsrDyn2 = cajsrDyn(tDyn>=bclRange(i));
                cajsrDyn1Depletion = 100*(1-min(cajsrDyn1)/max(cajsrDyn1));
                cajsrDyn2Depletion = 100*(1-min(cajsrDyn2)/max(cajsrDyn2));
                
                apd1 = DataReporter.getAPD(tDyn1, vDyn1, 0.9); apd2 = DataReporter.getAPD(tDyn2, vDyn2, 0.9);
                cat1 = max(caiDyn1) - min(caiDyn1); cat2 = max(caiDyn2) - min(caiDyn2);
                minAPD90(i) = min(apd1, apd2); maxAPD90(i) = max(apd1, apd2);
                
                apd1 = DataReporter.getAPD(tDyn1, vDyn1, 0.3); apd2 = DataReporter.getAPD(tDyn2, vDyn2, 0.3);
                minAPD30(i) = min(apd1, apd2); maxAPD30(i) = max(apd1, apd2);
                minCaT(i) = min(cat1, cat2); maxCaT(i) = max(cat1, cat2);
                
                % also computing min/max ionic concentration
                minCa(i) = min(caiDyn);
                maxCa(i) = max(caiDyn);
                
                minJSRDepletion(i) = min(cajsrDyn1Depletion,cajsrDyn2Depletion);
                maxJSRDepletion(i) = max(cajsrDyn1Depletion,cajsrDyn2Depletion);
                
                figure(222); clf;
                subplot(2,1,1);
                plot(currentsDynamic{i}.time, currentsDynamic{i}.V);
                subplot(2,1,2);
                plot(currentsDynamic{i}.time, currentsDynamic{i}.Cai);
                saveas(gcf, [folderCalibration '/traces/' num2str(i) '.png']);
            end
            [~, i260] = find(bclRange == 260, 1, 'first');
            [~, i400] = find(bclRange == 400, 1, 'first');
            [~, i1000] = find(bclRange == 1000, 1, 'first');
            outputs.maxCa400 = maxCa(i400);
            outputs.maxCa1000 = maxCa(i1000);
            
            % Getting sample traces of vm and cai for 260-ms-bcl paced cell
            t260_1 = timeDynamic{i260}{1};
            t260_2 = timeDynamic{i260}{2};
            v260_1 = XDynamic{i260}{1}(:,1);
            v260_2 = XDynamic{i260}{2}(:,1);
            cai260_1 = XDynamic{i260}{1}(:,6);
            cai260_2 = XDynamic{i260}{2}(:,6);
            
            
            % rate-dependence
            figure(8); clf
   
            subplot(1,2,1);
            plot(bclRange, minAPD90, 'k', bclRange, maxAPD90, 'k', 'LineWidth',1.5); xlabel('bcl (ms)'); ylabel('min/max APD90');xlim([bclRange(1)-20, bclRange(end)]);
            subplot(1,2,2);
            plot(bclRange, minCaT, 'k', bclRange, maxCaT, 'k', 'LineWidth',1.5); xlabel('bcl (ms)'); ylabel('min/max CaT (mmol/L)');    xlim([bclRange(1)-20, bclRange(end)]);
           
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 15]); %
            saveas(gcf, [folderCalibration '/rateDependenceAPD_Ca.png']);
            
            figure(9); clf
            subplot(1,2,1);
            plot(bclRange, minAPD90, 'k', bclRange, maxAPD90, 'k', 'LineWidth',1.5); xlabel('bcl (ms)'); ylabel('min/max APD90'); xlim([bclRange(1)-20, 500]);
            subplot(1,2,2);
            plot(bclRange, minCaT, 'k', bclRange, maxCaT, 'k', 'LineWidth',1.5); xlabel('bcl (ms)'); ylabel('min/max CaT (mmol/L)');   xlim([bclRange(1)-20, 500]);

            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 15]); %
            saveas(gcf, [folderCalibration '/rateDependenceAPD_Ca_inset.png']);

            % Alternans scoring
            % first we get indices of pacing rates (without drug block)
            % faster than 300 ms bcl. The cumbersome two-tier extraction is
            % ment to be safe against additional entries at the end of
            % bclRange added by the user, and also taking into account
            % 'small' and 'full' assessment of dynamic restitution.
            [~, i400] = find(bclRange == 400, 1, 'first');
            bclsUnder400 = bclRange(1:(i400-1));
            bclsUnder300 = find(bclsUnder400<300);
            
            diffAPD = maxAPD90-minAPD90;
            if (max(diffAPD(bclsUnder300)) > 5)
                score.alternans = 1;
            else
                score.alternans = 2;
            end
            
            %% Drug block behaviour
            % Data are extracted from
            % a) means and stdevs are from ORd paper
            % b) APD of original and new model are obtained from Dutta
            % paper - higher resolution
            for i = 1:12
                apdDrug(i) = DataReporter.getAPD(currentsDrugBlock{i}.time, currentsDrugBlock{i}.V, 0.9);
            end
            %{'drug_e4031.png'}, {'drug_hmr1556.png'},{'drug_mexiletine.png'},{'drug_nisoldipine'}
            
            figure(10);          
            dataMean = 140 + [358, 540, 682]*115/214; % in square brackets, how many pixels above baseline (140 ms apd) we are
            dataStd = [117,127, 149]*115/214; % 115/214 because there are 214 pixels per 115 ms
            apdOptimisedORd = [345, 441, 500];
            errorbar([500,1000,2000], dataMean, dataStd,'k', 'LineWidth',2);
            hold on
            plot([500,1000,2000], apdOptimisedORd,'--', 'LineWidth', 1.5);
            plot([500,1000,2000], apdDrug(1:3), 'LineWidth', 1.5);
            hold off
            legend({'data', 'optimised dynIKr ORd', 'new model'}, 'Location','southeast')
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            xlabel('bcl'); ylabel('apd90'); title('E-4031 (1 \muM)');
            saveas(gcf, [folderValidation '/drug_e4031.png']);
            % scoring E4031
            absDistanceFromMean = abs(apdDrug(1:3)-dataMean);
            if (min(absDistanceFromMean<dataStd)==1) % if this is true, all distances from mean are smaller than corresponding stdev, i.e., the data is within stdev.
                score.drug_e4031 = 1;
            else
                score.drug_e4031 = 2;
            end
            
            figure(11);
            dataMean = 140 + [152, 244, 282]*115/214; % in square brackets, how many pixels above baseline (140 ms apd) we are
            dataStd = [53, 60, 67]*115/214; % 115/214 because there are 214 pixels per 115 ms
            apdOptimisedORd = [245, 293, 316];
            errorbar([500,1000,2000], dataMean, dataStd,'k', 'LineWidth',2);
            hold on
            plot([500,1000,2000], apdOptimisedORd,'--', 'LineWidth', 1.5);
            plot([500,1000,2000], apdDrug(4:6), 'LineWidth', 1.5);
            hold off
            xlabel('bcl'); ylabel('apd90'); title('HMR-1556 (1 \muM)');
            legend({'data', 'optimised dynIKr ORd', 'new model'}, 'Location','southeast')
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderValidation '/drug_hmr1556.png']);
            % scoring HMR1556
            absDistanceFromMean = abs(apdDrug(4:6)-dataMean);
            if (min(absDistanceFromMean<dataStd)==1)
                score.drug_hmr1556 = 1;
            else
                score.drug_hmr1556 = 2;
            end
            
            figure(12); clf;
            dataMean = 140 + [91, 142, 166]*115/214; % in square brackets, how many pixels above baseline (140 ms apd) we are
            dataStd = [46, 49, 49]*115/214; % 115/214 because there are 214 pixels per 115 ms
            apdOptimisedORd = [178, 209, 219];
            errorbar([500,1000,2000], dataMean, dataStd,'k', 'LineWidth',2);
            hold on
            plot([500,1000,2000], apdOptimisedORd,'--', 'LineWidth', 1.5);
            plot([500,1000,2000], apdDrug(7:9), 'LineWidth', 1.5);
            hold off
            xlabel('bcl'); ylabel('apd90'); title('nisoldipine (1 \muM)');
            legend({'data', 'optimised dynIKr ORd', 'new model'}, 'Location','southeast')
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderValidation '/drug_nisoldipine.png']);
            % scoring nisoldipine
            absDistanceFromMean = abs(apdDrug(7:9)-dataMean);
            if (min(absDistanceFromMean<dataStd)==1)
                score.drug_nisoldipine = 1;
            else
                score.drug_nisoldipine = 2;
            end
            
            figure(14); clf
            dataMean = 140 + [113, 168, 189]*115/214; % in square brackets, how many pixels above baseline (140 ms apd) we are
            dataStd = [60, 69, 75]*115/214; % 115/214 because there are 214 pixels per 115 ms
            apdOptimisedORd = [210, 241, 263];
            errorbar([500,1000,2000], dataMean, dataStd,'k', 'LineWidth',2);
            hold on
            plot([500,1000,2000], apdOptimisedORd,'--', 'LineWidth', 1.5);
            plot([500,1000,2000], apdDrug(10:12), 'LineWidth', 1.5);
            hold off
            xlabel('bcl'); ylabel('apd90'); title('mexiletine (10 \muM)');
            legend({'data', 'optimised dynIKr ORd', 'new model'}, 'Location','southeast')
            set(findall(gcf,'-property','FontSize'),'FontSize',14)
            saveas(gcf, [folderValidation '/drug_mexiletine.png']);
            % scoring mexiletine
            absDistanceFromMean = abs(apdDrug(10:12)-dataMean);
            if (min(absDistanceFromMean<dataStd)==1)
                score.drug_mexiletine = 1;
            else
                score.drug_mexiletine = 2;
            end
            
            %% Rate accommodation
            if (APDaccommodation == 1)
                figure(16);
                plot(adaptationTimes480, adaptationAPDs480, adaptationTimes410, adaptationAPDs410, 'LineWidth', 1.5);
                legend({'750-480-750','750-410-750'}, 'Location', 'southeast'); ylabel('APD (ms)'); title('Rate adaptation');
                saveas(gcf, [folderValidation '/rateAdaptation.png']);
            end
            
            score.apdAccommodation = 3; % no grading here
            
            %% S1-S2
            S2interval = [140:5:500 600:100:1500];
            apdS1S2_1000 = nan(size(S2interval));
            diS1S2_1000 = nan(size(S2interval));
            maxV = nan(size(S2interval));
            for i = 1:length(S2interval)
                % For each coupling interval, we get diastolic interval
                % (exists if and only if there are three segments under
                % APD90 threshold and then it's the middle one's length) and
                % APD of the 2nd spike                
                diS1S2_1000(i) = DataReporter.getDI(currentsS1S2_1000{i}.time, currentsS1S2_1000{i}.V);
                timeAP2_1000 = currentsS1S2_1000{i}.time; 
                VAP2_1000 = currentsS1S2_1000{i}.V; 
                VAP2_1000(timeAP2_1000<S2interval(i)) = []; 
                timeAP2_1000(timeAP2_1000<S2interval(i)) = [];
                apdS1S2_1000(i) = DataReporter.getAPD(timeAP2_1000, VAP2_1000, 0.9);
                maxV(i) = max(VAP2_1000);
            end
            
            apdS1S2_1000(diS1S2_1000 == -1) = NaN; % where diast interval undefined, we don't consider APD.
            apdS1S2_1000(maxV < 0) = NaN; % under-propagating APs are discarded.
            
            figure(15);
            plot(diS1S2_1000, apdS1S2_1000, 'LineWidth', 1.25);
            legend('S1=1000 ms');
            xlabel('diastolic interval (ms)');
            ylim([160 300]);
            set(gcf, 'PaperUnits', 'centimeters'); set(gcf, 'PaperPosition', [0 0 30 30]); %
                        
            dvdt1000 = diff(apdS1S2_1000)./diff(diS1S2_1000);

            
            saveas(gcf, [folderValidation '/S1S2.png']);


            score.s1s2 = 3;

         
            

            %% List of features reported & how they score 1 - ok, 2 - fail, 3 - NA            
            listOfFeatures = {'AP morphology', 'Sample traces', 'Calcium transient',...
                'INa block', 'ICaL block', 'IK1 block', 'EAD formation',  'Dynamic restitution and alternans',...
                'drug block (E-4031)','drug block (HMR-1556)','drug block (mexiletine)','drug block (nisoldipine)' ,'Rate accommodation','S1-S2'}; % validation
            imgsPerFeature = {{'calibration/apMorphology.png'}, {'calibration/examples_AP.png','calibration/examples_Cai.png'},{'calibration/CaT.png'},...
                {'calibration/inaBlock.png'},{'calibration/icalBlock.png'},{'calibration/ik1Block.png'},{'calibration/eadFormation.png'},{'calibration/rateDependenceAPD_Ca.png', 'calibration/rateDependenceAPD_Ca_inset.png'},...
                {'validation/drug_e4031.png'}, {'validation/drug_hmr1556.png'},{'validation/drug_mexiletine.png'},{'validation/drug_nisoldipine.png'},{'validation/rateAdaptation.png'},{'validation/S1S2.png'}}; %validation
            
            
            
            scoreCodes = [score.apMorphology,score.exampleTraces, score.CaT,... % sample traces have no rating
                score.INaBlock, score.ICaLBlock, score.IK1Block, score.EADs, score.alternans,...
                score.drug_e4031, score.drug_hmr1556, score.drug_mexiletine, score.drug_nisoldipine, score.apdAccommodation, score.s1s2  ]; % also no rating here.          
            
            
            % The features, images, and score codes are filtered by whether
            % we're running calibration, validation, or full report
            indicesCalibration = 1:8;
            indicesValidation = 9:14;
            if (strcmp(reportType, 'calibration'))
                listOfFeatures = listOfFeatures(indicesCalibration);
                imgsPerFeature = imgsPerFeature(indicesCalibration);
                scoreCodes = scoreCodes(indicesCalibration);
            elseif (strcmp(reportType, 'validation'))
                listOfFeatures = listOfFeatures(indicesValidation);
                imgsPerFeature = imgsPerFeature(indicesValidation);
                scoreCodes = scoreCodes(indicesValidation);
            end
            
            outputs.scoreCodes = scoreCodes;
            %% Printing html
            fid = fopen([folderOut '/report.html'], 'w');
            %% HEADER
            fprintf(fid, ['<html><head><title>QC report in ORd, ' fileSimulation '</title>\n']);
            % css and JS
            fidCSSandJS=fopen('htmlStuff/htmlAndCSS.txt');
            while 1
                tline = fgets(fidCSSandJS);
                if ~ischar(tline), break, end
                fprintf(fid, '%s', tline);
            end
            fclose(fidCSSandJS);
            
            fprintf(fid, '</head><body><div class=\"header\"><div id=\"header_title\">\n');
            fprintf(fid, 'Evaluation report</div>\n');
            fprintf(fid, '<div class=\"header_slider\"><input type=\"range\" min=\"10\" max=\"80\"></div>\n');
            fprintf(fid, ['<div id="header_filename">' fileSimulation ' <br/>' datestr(now) '</div></div>']);
            
            %% LEFT MENU
            fprintf(fid, '<div class=\"summary\"><ul>\n');
            % the menu itself
            for iElement = 1:length(listOfFeatures)
                fprintf(fid, '<li>');
                fprintf(fid, DataReporter.getIconBase64(scoreCodes(iElement)));
                fprintf(fid,['<a href=\"#', num2str(iElement), '_', listOfFeatures{iElement}, '">', listOfFeatures{iElement}, '</a></li>\n']);
            end
            %            listOfFeatures
            
            
            
            fprintf(fid, '</ul></div>\n');
            
            %% MAIN
            fprintf(fid, '<div class="main">\n');
            
            % In a for loop, we generate all the plots. A section header is
            % shown, followed by all the plots
            for iFeature = 1:length(listOfFeatures)
                fprintf(fid,  ['<div class="module"><h2 id="', num2str(iFeature), '_', listOfFeatures{iFeature}, '">\n']);
                fprintf(fid, DataReporter.getIconBase64Larger(scoreCodes(iFeature)));
                fprintf(fid,  [listOfFeatures{iFeature} '\n']);
                fprintf(fid, '</h2>\n');
                fprintf(fid, '<div id="gallery"><div id="thumbs">\n');
                fprintf(fid, '<div>\n');
                
                % one div per image via y = base64encode(x, alg
                for iPlot = 1:numel(imgsPerFeature{iFeature})
                    fprintf(fid, ['<img src="' imgsPerFeature{iFeature}{iPlot} '" alt="Fig not found" class="eval_img_" openedImage="src_openedImage" />']);
                end
                
                fprintf(fid, '</div>\n'); % div
                fprintf(fid, '</div>\n'); % div thumbs
                fprintf(fid, '</div>\n'); % div gallery
                fprintf(fid, '</div>\n'); % div class module
            end
            
            fprintf(fid, '</div>\n'); % div main
            
            %% FOOTER
            fprintf(fid, '</body></html>');
            
            fclose(fid);
            
        end
        
        
        
        function diastolicInterval = getDI(time, membranePotential)
            % Receives a trace of 2 action potentials, checks if there are
            % three segments under the 90% threshold of APD (pre-1st,
            % between APs 1 and 2, and after 3rd). If 3 present, the length
            % of the middle one is returned.
            
            baseline = mean(membranePotential((end-9):end));
            s = bwconncomp(membranePotential < (baseline + 0.1*(max(membranePotential)-baseline)));
            
            if (s.NumObjects~=3)
                diastolicInterval = -1;
            else
                di = s.PixelIdxList{2};
                diastolicInterval = time(di(end)) - time(di(1));
            end
        end
        
        function apd = getAPD(time, membranePotential, level, varargin)
            % returns APD at the desired level of repolarisation. Can be
            % also used for CaT duration
            
            % Given the data from Szeged sometimes have stimulation
            % artifacts, we chop the first 1 ms here.
            
            if numel(varargin)>0
                removeFirstNms = varargin{1};
            else
                removeFirstNms = 0;
            end
                
            
            membranePotential(time<removeFirstNms) = [];
            time(time < removeFirstNms) = [];
            
            baseline = membranePotential(end); 
            thresh = (baseline + (1-level)*(max(membranePotential)-baseline));
            s = bwconncomp(membranePotential > thresh);
            
            % Now, the first object above threhsold is taken - this is not
            % a problem in simulations.
            interval = s.PixelIdxList{1};

            
            % linear interpolation of the dt after the segment above
            % threshold ends.
            tEndInterval = time(max(interval));
            tAfterEndInterval = time(max(interval)+1);
            vEndInterval = membranePotential(max(interval));
            vAfterEndInterval = membranePotential(max(interval) + 1);
            howFarThreshAfterEnd = (thresh-vEndInterval) / (vAfterEndInterval-vEndInterval);
            interpAdding = (tAfterEndInterval - tEndInterval) * howFarThreshAfterEnd;
            apd = time(interval(end)) - time(interval(1))    +   interpAdding ; % time(interval(1)) should not be used when trimming is applied in varargin. 
            % TODO potentially add code for diagnostic visualisation
        end
        
        function apd = getAPD_Jakub_no_variable_upstroke_time(time, membranePotential, level, varargin)
            % returns APD at the desired level of repolarisation. Can be
            % also used for CaT duration
            
            % Given the data from Szeged sometimes have stimulation
            % artifacts, we chop the first 1 ms here.
            
            if numel(varargin)>0
                removeFirstNms = varargin{1};
            else
                removeFirstNms = 0;
            end
                
            
            membranePotential(time<removeFirstNms) = [];
            time(time < removeFirstNms) = [];
            
            baseline = membranePotential(end); 
            thresh = (baseline + (1-level)*(max(membranePotential)-baseline));
            s = bwconncomp(membranePotential > thresh);
            
            % Now, the first object above threhsold is taken - this is not
            % a problem in simulations.
            interval = s.PixelIdxList{1};

            
            % linear interpolation of the dt after the segment above
            % threshold ends.
            tEndInterval = time(max(interval));
            tAfterEndInterval = time(max(interval)+1);
            vEndInterval = membranePotential(max(interval));
            vAfterEndInterval = membranePotential(max(interval) + 1);
            howFarThreshAfterEnd = (thresh-vEndInterval) / (vAfterEndInterval-vEndInterval);
            interpAdding = (tAfterEndInterval - tEndInterval) * howFarThreshAfterEnd;
            apd = time(interval(end))  +   interpAdding ;
            % TODO potentially add code for diagnostic visualisation
        end
        
        function apd = getAPD_ignore_first_10ms(time_original, membranePotential_original, level, varargin)
            % returns APD at the desired level of repolarisation. Can be
            % also used for CaT duration
            shift_idx = find(time_original > 10);
            membranePotential = membranePotential_original(shift_idx:end);
            time = time_original(shift_idx:end);
            
            % Given the data from Szeged sometimes have stimulation
            % artifacts, we chop the first 1 ms here.
            
            if numel(varargin)>0
                removeFirstNms = varargin{1};
            else
                removeFirstNms = 0;
            end
                
            
            membranePotential(time<removeFirstNms) = [];
            time(time < removeFirstNms) = [];
            
            baseline = membranePotential(end); 
            maxV = max(membranePotential);
            thresh = (baseline + (1-level)*(maxV-baseline)); % It needs the original vms to calculate the threshold
            s = bwconncomp(membranePotential > thresh);
            
            % Now, the first object above threhsold is taken - this is not
            % a problem in simulations.
            interval = s.PixelIdxList{1};

            
            % linear interpolation of the dt after the segment above
            % threshold ends.
            tEndInterval = time(max(interval));
            tAfterEndInterval = time(max(interval)+1);
            vEndInterval = membranePotential(max(interval));
            vAfterEndInterval = membranePotential(max(interval) + 1);
            howFarThreshAfterEnd = (thresh-vEndInterval) / (vAfterEndInterval-vEndInterval);
            interpAdding = (tAfterEndInterval - tEndInterval) * howFarThreshAfterEnd;
            %apd = time(interval(end)) - time(interval(1))    +   interpAdding ;
            apd = time(interval(end)) + interpAdding;
            % TODO potentially add code for diagnostic visualisation
        end
        
        function maxDVDT = getPeakDVDT(dataExperimentTime, membranePotential, ignoreFirstMsInExp)
            % Extracts maximum dV/dt in a signal, ignoring first
            % ignoreFirstMsInExp ms as that contains a synthetic stimulus
            membranePotential = membranePotential(dataExperimentTime > ignoreFirstMsInExp);
            dataExperimentTime = dataExperimentTime(dataExperimentTime > ignoreFirstMsInExp);
            dv = diff(membranePotential);
            dt = diff(dataExperimentTime);
            maxDVDT = max(dv./dt);
        end
        
        
        
        function icon = getIconBase64(type)
            % 1-ok, 2-fail, 3-na
            
            if (type == 1)
                icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAASBSURBVFiFvddtbFNVHMfx7zm37dauw7CH7gmmC2YyNki28rCKCcGQGKOIZDDEiQvIDNMARkUTMVGJBl9AjEKULApOITICGozPJBACbiOwTcEgEBIIGxsqGF0faNf2Hl+wR+7t6KD1vGrO/59zP/3de85tBcOG+4TboSLqWV3oVQJRAmSS+HFNoX5Hsdeu2xta7m+5PlAQAx/Kj5V7gD3AhCQAYo1OoUR1e2V7K4AcBjn4f0AmpRTxpK0a/boOMFEJdcjd4p4FIN0n3A5uJJKaTIREUpexnC+mfkp2NAvfL74BUKou9T2eZo/dEo1E64QQSU3EJm28lbeehwrmAfCH709Un8J/yk/atDRkqiwMasGVUghRlUyIQ3OwtXDzIAQgGAmCBBVWBE4H0IM6QJUESpMFkUKyIfd1prsqRsxHVAQhBYh+0NkAKqjKJDA+WZj1Oet4sGCOYV4JBQKENgx0PpAhGba9EzkWjpvPwsLHTGtOm3MQggQhBSqihEwGZKJ1Ai9NWhOznp7qHAFBghCChGM0ofHu3RtwWB0xe3KcLgMEjcRjlt61mJKM+0btKRp/jwGS8GRyrTmsKnrmln190T4DBAmWRGLWuOpHvT1RFaWhYztb2rcZIAnFlKaWjDjYbh69IS9rDrxMc3erKSTu2ySFZMm4Kl51vojqU6Y9q3NXIWKcEpe93SzdXzsEkUYIWhzJTE4p5rWCdZRlTSGiR/hwRwO9k70I69CFZ6ZNZ6ZrekxIzf4VdAd6hiDSCBk1GYfmYG3mc3w+9WPKsqYAYJEWastq8LZ5UeGhhJZnP2W6Ro/vCsu+XhkXBBlja892VNJU3EjtpBo0qY2oPVGxiHHRdHwdPlRYUWafwizXDMMa/rCfuu+ep8t/OS6IkCbJLE9fxpayzRSk55snZnWwtKKaqD+K/5SfFRlPG3qiKsoLB17h3D/n44aYJtN4bRfvHdlKOBo2xQDUumtIsaZQ7LiXOQUPGOofHP+Iw11HxwRBmGB0m04ju3h85xIuXLtoinE5s1lQ+ij1s+oMO+h4TxsNv24fM0RoAlF+rNx0r+ohHXlKsOXhzcwu8hjq3d4ecpwuNDH0TP0b6mXBvmp6/FfGDDFNZmDIFIk+TbHim3p2dTQZ6vnpeSMgAG8ceXvskP4fWQMPsPkpBkibxF5u582f3+GTE42x2gD48tx+vr/w09ghQ3NKy6vLWw3EfKEITWDNtnKo/TB2kYo7v9zQ81fgKvU/riWk+m4XAoK/JXB61K/cn5BzmpNNbe+z+7e9hvrG1k30Rry3DRFSIKU8IxEYVzdLyCZwlDrYcGwjhy8dHZw/2tXMtxd+uCMIArCwV3iaPfagFjwLTIwHpfoUtktWmh75jNy0HOZ/tZhOX9edQSSdmZbMYgFQ0VpRqYQ6CNjjAoUVBVfzqXTNYOeZ3XcKCQqLmHty7snWwROrH9QEFMYLCl4MoiLq9iGCLou0VHfM62iBm/6meJo99qAlWIdiEVACZN0KFOoM3QCNYdcIKc4g2GcL2ba1zW8LDKz3HxqRgG49XL1ZAAAAAElFTkSuQmCC" alt="[PASS]" class="image_without_zoom" width=23/>';
%             elseif (type == 2)
%                 icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAO5SURBVFiFtZjPbxtFFMc/M7u2184P0yZCVYUCASRSckSkRaISHLhUQhxSUokbiEL/B4TEiTOcQIU7akgPiAsgQKVFAQrihyKlakjIrzY4Su2kdhzbu955HDaN49gbe+3wvYz2vXnPn53ZffvGin2SL0lR1m+BGgc5BQxw9MqCuoXIFJjLaoLSA4faA5nkOZSeBB75HwCaS7GKMhNqnJ/3YHZBvgecSMmcE3jOk4jSxMsLULrbCVIZZV5Q5/lF7W7NbaKsSGKQwtOf0jt8DqWtwCZCYeUa8ZlLJMrzUYFWEDOiKemLkUCA/MjH9D3xcg0EQCn6Hn0R/8wX7LhWeHBzDaH0mxqlxiOFKZve4XOh7tTgCPOVZyi5UXnUuAYZjRQT60fbicPTOg8zswJlL0piGdXAsUgwSrcxx6LiwexqJKABzb7Xuy1Vsphq5dAp7nYmGKswtwaV9oDauc2DErbXZ8K9xmfn3q0aWBUW1oOxlTqAATdzI9RXyPxFtXy/fn4VljZaA3UEE98Kh9lc+qGp3avCnSx4/hHD9G3/iIgJgbkeGuf5sJaDaghQRzDKy1JY+63BLr5HbvHaobG+gfX7zYE6ggHw7n7dYMst3Wh4XsKAstvBeCQwPVvfNNg25r5qO943sFUEI0cA4xRu4rvFOtu9he8i5fANFEo1oI5hkCrFTO258d0ihUx4/QmTESi5ININDOBv1n48n5lBTBuVrRmQgUq1Sxi1swiAiKGY+6ebVEE++VxL62nN5Zo412eF7bKQSPYglS1sC2IW2JbaHdlno85ma4jZtbGrlYlrl7NPefTGPbzSVjepgC5hABIxOHtKkU5F+/g3k91ddC+V9PO41nGeTd3k95l5diod7zo2IETtaQC3fwxz+gpOeogEQeswcuJ95r59j4rXEZBoIBc5zEphxj7DSQ/tmZS2GDz9LidHX6HHib5lImQ1qNmogW56DOehx5r6Eo+/SjIOqcPb5AYppWY1SqaiwpQlHeqLJY+xkRdScXDi7ecUkasaYz5BsRoFJlX8FeM3P4tsLgWN10Y+WB0n1jqfUqwkXXNZqwlKKDMBtQN4K9nuGhvT7wQflH3K//sny9Mf7F1nC5BsASRC2ShzQb1OuXbwv8oZjL4CDIWH1utv8xL28Gs4PcfZXJ5m+acP0aa0W1UVMTuouIP9QdtphINV+Y6lzIWBN5iGA6+0TJJE64uIOr/7l8hgK6A/FmE+I3Xl/iCMbcFAX9DdibAZ0+p2LCZTYsxHJ99m50Gu/wBfK3bpmf5pOwAAAABJRU5ErkJggg==" alt="[WARN]" class="image_without_zoom" width=23/>';
            elseif (type == 2)
                icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAWCSURBVFiFtZhpbFRVGIafc++dTmc6bcEWWiqg0MjSCkIAKwQw4A8kQIIU2URQEJFVFpVAlB+QECGRxYAQlhBRQAgoYsCQICiJQNhkEVoQaCmTsrSUtlNm6czc44/S6Wx3Zmrw/TOTc893znPf737fuTOCIMleOVac8kOELAS6Ahk8ez0CipByHw7TZmG3uxoviABIl9Z9UZS9QNv/AcBId5FyjCh6cBpACQI5FhPEbKaicCJPdPksYdohxHGZl1UAIGSvHCsu/XosED07hyfrd2Bu9wK7urZnjC0JqyKMpv8XlVGrdVFw+qfFAvG374h7x8+kds0nyWbD1L0n+xxunIk4lGKjvmcfngwZgbNgAGgmo5ntSfN+oCFEoeFiJhOudduwPd/E2m3cRI5fOMcBh4e30sxYRKRDMqMVjjmLSBk+iiSLhaSn486SW/DRRKz20si9hChUgHwjFsewQmyduoSM5Y+dQGpWNo/9OgcdHlwy1CFvzz74Dhwn7e13UC2WkGvWDrn4vtpEXTRTJfkK0NIIxtPv9YgxNclM7zkLUARU+3UOBwHV53dH2bIH03PGHSHt5VcoHTQ0WiFkKASVd7iELTXqeLf3p9GmdwEKUK1LjtR5cGe0Qq7bHuFGNLUZN4nzbm/4cyeUWEH+e+XRIRWFQWs2kGS1ogAOoVG3eivmNjkxITwPH2DfvYPKY0fwIrjo9oWkOSaM6fcjhtfSc1+i7/JVKEDvL5aR2ftVw7nue+Vc+ng6fxTkU7R4Pvbtm1GkxIvkShCQkHnZxjUqBNW7D9OiWw/DKbf27CR3zASIUlUAlSdPcHHmFPyPq1CEQKHBAUUQ+G5RBHlmLQ4M4MjtjGXvr2jJ8Z+FcFWdOcXZSaORLheqAUjjuPnpZ0yl3rpO1aI5IJt3DNT8fYlzU8bHBAl2yo+MDwPQ+ugh7q9dmTCIt7aGC9MnoztqmzaOARI8npCytn1NxaEDCc29+vkneOxloZvGBUkgTY0SUtLy6y/jpstbU03d1SvNBxHNcAbANftTw6pplCm9BX0P/kab4SNDU5QASMIwrmGjSB1hfKYGS01Jodv6bXRevgrNkmwIooaBxC1tgPpeBaibf0BNTk6UPSDnjSJuLpyJK5C6JkfUkPQlAOMeNATTyg2oKSnNBglI13n0y49UfLcV11/nAqlRCXZKxIBJS8cxdzG2se8iFINsSsnVpZ/RYeoMrC92TIjLV1mB8/IFPMXXeLR6RQAk8pnJbI2n/yBqlq7Cd/QcqeMnG4MAZd9vx75jG2dHvUnN+TMJwWiZrUgbPAS9pjoUJCRNioJj5yFSuxufQ8Fylt7m5NCBDR0W0JKTyVuzicyhI+LG1t8poXT4QBS3J6L0G2B0HZYtQvr9cReTfj9XFs5CulxN9nrcFM+eQvnWb2IH6zoPlsyLAFEFUgGqGuelFl2mcuPauDAlG9dRe/5MZM/QJWUrlvLPjPeov38v8ibq67m/ZB7uM6cimyGiSsi87BPAgECAEFQuXUWrMROjgpT/tJdrC2chdD2wYENVhPYMzWohY+hIbP36o6WlU3/zBrX7d+Erud3oREgzVAV/CpmXNRfEuhAnFQX7uKlkz5pPUouGV2RPZQUl61dz99stKFLGBGmyPrzRibC4pmaoIRYI2bathTTfdaBdCBBwwQvO3E5ItxPPndImN4I3NAAJaWphIA1xTSAq8q7Hae4sAGTXrNdQxDEklnCgi24vFT7d8MiP/q4S1NSigITECdyaYHBG8cNTCoAoenAaXQ4GyoJhFKBHsolsTW3GO0mQU3FAVKRdE7yRUfzwFIT9TGlImXcaiNE0/CWS2ejQVY+PKr8eAaIagYTefXBclSYoViX7vXXappzycmfj/v8C2833DsQYRuIAAAAASUVORK5CYII=" alt="[FAIL]" class="image_without_zoom" width=23/>';
            elseif (type == 3)
                icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAARYSURBVFiFxZc9TyNHGMf/M7vrMbDYvHhty0CuQECAgheJgi5CVGmRiJQ6SYUs5J5vYKWw3HD5BtEVqaA72ktpjIIuws05GGy8xmAb7xu7Ke5247V3ja0kd39pJT/PMzPPzzP7zM4QdOj4+HhU1/WfAOwBWAYwjf9eMoBLy7Le6Lr+OpVKte0AsX9ks9ltAL8CmP0fAPxUNE1zP5lMvnNgPoG8BRD8jCC2FADfHBwc/E4+Lc17fN4Z6dYHTdO+ppqm/fiFQQDgK0EQfqCU0r0vDGJrj7csa9UrwnEceJ53bF3XYZomCCEIBAKO//n5GYZhePbplKqqfUkIIas8gEmv4OLiInZ2dhz75OQEhUIBsVgM+/v7jv/s7Ay5XA6CIGBtbQ3b29ueydLpNEZGRvrxTFN0lHenTNN02aurqygUCrAsy+XXNA3FYhGGYUAURd9MhBDIstwPhtB+0U69evUKwWAQ19fXPTFd11EsFhEOh337S5KEcrmMWq3m22ZgGADY3NxEq9XyjOm6jomJCcfunsFYLAYAqFQqvkBDw/iJMYZQKOTYtVoNmqY5djQadX5Xq1XU6/V/BzM/P+/6952annZ/xmRZxuPjoyeMH9BQMIQQbGxseMampqZc9u3tLR4eHlwwhPxTK5ZlQZZlV5uhYABgfX3d0989Y5VKxVU9giBgcrJ3F6nVag7Q0DB+5dudqFqt4v7+3uXrXipb9XodjUZjcBi/KrIViURc9t3dXc87YVeUH9DAMBcXF33jkiT1wHTPTDdwt7w/JB7K5/PY2toCpb38hJCeRJlMpqddJBKBaZqeYwBDvDOtVgtXV1eesVAoBMbYi2PEYjE8PT31bIhDwwDA+fm5p7+7rP0kSRIopb5AQ8HkcjlPv1fJHh4e4ujoyJ2MUkSjUZimiXa73QM0FEypVEKlUhkI5ubmxrNtPB4H8PFUoCiKC2jofSafz/f4updJVVUoioJms+kLYwOpquoAkWw2a8LjTEMIQbFYhKZpIISg0WjAMAyMjo4iFAqBEAJKKRRFgWEYkGUZhBDnkWUZlFIkEglwHAdCCDiOg6IoUBQFlFLnZMjzPBhjFg+gBo/LmmVZkCQJl5eXUFUVHMeBUop2uw1VVUEpBaXUScLzPEqlEoCPx0+O4wAA5XLZ6Wv77VhnLsMwHigh5A+/JWGMYWVlBcHgy9cpURQxNzfnu4cMoCtqWdabfi0CgQCWl5cH2kfGxsYwOzvr+joPKkEQfqOapv0CoPgS0NLSkutW4CdRFDEzMzMsUDkej/9MU6lU2zTNfQDtfq0DgQAWFhYGmiFRFJFIJAZaMkKIxhj7PpVKtSkAJJPJd6Zp7gD40K8jYwwLCwsDzdD4+DgSiUTfGaKUVhhj36bT6bcA4LzWp6enf+3u7h5zHFchhIgAxgCMdg/AcRzC4TAajYZzqbOrqvOhlIIxhmAwiGaz6fg4jmvwPP+nIAivKaXfZTKZ9/bYfwN/BqLKrNdTsQAAAABJRU5ErkJggg==" alt="[NA]" class="image_without_zoom" width=23/>';
            end
        end
        
        
        function icon = getIconBase64Larger(type)
            % 1-ok, 2-fail, 3-na
            
            if (type == 1)
                icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAASBSURBVFiFvddtbFNVHMfx7zm37dauw7CH7gmmC2YyNki28rCKCcGQGKOIZDDEiQvIDNMARkUTMVGJBl9AjEKULApOITICGozPJBACbiOwTcEgEBIIGxsqGF0faNf2Hl+wR+7t6KD1vGrO/59zP/3de85tBcOG+4TboSLqWV3oVQJRAmSS+HFNoX5Hsdeu2xta7m+5PlAQAx/Kj5V7gD3AhCQAYo1OoUR1e2V7K4AcBjn4f0AmpRTxpK0a/boOMFEJdcjd4p4FIN0n3A5uJJKaTIREUpexnC+mfkp2NAvfL74BUKou9T2eZo/dEo1E64QQSU3EJm28lbeehwrmAfCH709Un8J/yk/atDRkqiwMasGVUghRlUyIQ3OwtXDzIAQgGAmCBBVWBE4H0IM6QJUESpMFkUKyIfd1prsqRsxHVAQhBYh+0NkAKqjKJDA+WZj1Oet4sGCOYV4JBQKENgx0PpAhGba9EzkWjpvPwsLHTGtOm3MQggQhBSqihEwGZKJ1Ai9NWhOznp7qHAFBghCChGM0ofHu3RtwWB0xe3KcLgMEjcRjlt61mJKM+0btKRp/jwGS8GRyrTmsKnrmln190T4DBAmWRGLWuOpHvT1RFaWhYztb2rcZIAnFlKaWjDjYbh69IS9rDrxMc3erKSTu2ySFZMm4Kl51vojqU6Y9q3NXIWKcEpe93SzdXzsEkUYIWhzJTE4p5rWCdZRlTSGiR/hwRwO9k70I69CFZ6ZNZ6ZrekxIzf4VdAd6hiDSCBk1GYfmYG3mc3w+9WPKsqYAYJEWastq8LZ5UeGhhJZnP2W6Ro/vCsu+XhkXBBlja892VNJU3EjtpBo0qY2oPVGxiHHRdHwdPlRYUWafwizXDMMa/rCfuu+ep8t/OS6IkCbJLE9fxpayzRSk55snZnWwtKKaqD+K/5SfFRlPG3qiKsoLB17h3D/n44aYJtN4bRfvHdlKOBo2xQDUumtIsaZQ7LiXOQUPGOofHP+Iw11HxwRBmGB0m04ju3h85xIuXLtoinE5s1lQ+ij1s+oMO+h4TxsNv24fM0RoAlF+rNx0r+ohHXlKsOXhzcwu8hjq3d4ecpwuNDH0TP0b6mXBvmp6/FfGDDFNZmDIFIk+TbHim3p2dTQZ6vnpeSMgAG8ceXvskP4fWQMPsPkpBkibxF5u582f3+GTE42x2gD48tx+vr/w09ghQ3NKy6vLWw3EfKEITWDNtnKo/TB2kYo7v9zQ81fgKvU/riWk+m4XAoK/JXB61K/cn5BzmpNNbe+z+7e9hvrG1k30Rry3DRFSIKU8IxEYVzdLyCZwlDrYcGwjhy8dHZw/2tXMtxd+uCMIArCwV3iaPfagFjwLTIwHpfoUtktWmh75jNy0HOZ/tZhOX9edQSSdmZbMYgFQ0VpRqYQ6CNjjAoUVBVfzqXTNYOeZ3XcKCQqLmHty7snWwROrH9QEFMYLCl4MoiLq9iGCLou0VHfM62iBm/6meJo99qAlWIdiEVACZN0KFOoM3QCNYdcIKc4g2GcL2ba1zW8LDKz3HxqRgG49XL1ZAAAAAElFTkSuQmCC" alt="[PASS]" class="image_without_zoom">';
            %elseif (type == 2)
            %    icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAO5SURBVFiFtZjPbxtFFMc/M7u2184P0yZCVYUCASRSckSkRaISHLhUQhxSUokbiEL/B4TEiTOcQIU7akgPiAsgQKVFAQrihyKlakjIrzY4Su2kdhzbu955HDaN49gbe+3wvYz2vXnPn53ZffvGin2SL0lR1m+BGgc5BQxw9MqCuoXIFJjLaoLSA4faA5nkOZSeBB75HwCaS7GKMhNqnJ/3YHZBvgecSMmcE3jOk4jSxMsLULrbCVIZZV5Q5/lF7W7NbaKsSGKQwtOf0jt8DqWtwCZCYeUa8ZlLJMrzUYFWEDOiKemLkUCA/MjH9D3xcg0EQCn6Hn0R/8wX7LhWeHBzDaH0mxqlxiOFKZve4XOh7tTgCPOVZyi5UXnUuAYZjRQT60fbicPTOg8zswJlL0piGdXAsUgwSrcxx6LiwexqJKABzb7Xuy1Vsphq5dAp7nYmGKswtwaV9oDauc2DErbXZ8K9xmfn3q0aWBUW1oOxlTqAATdzI9RXyPxFtXy/fn4VljZaA3UEE98Kh9lc+qGp3avCnSx4/hHD9G3/iIgJgbkeGuf5sJaDaghQRzDKy1JY+63BLr5HbvHaobG+gfX7zYE6ggHw7n7dYMst3Wh4XsKAstvBeCQwPVvfNNg25r5qO943sFUEI0cA4xRu4rvFOtu9he8i5fANFEo1oI5hkCrFTO258d0ihUx4/QmTESi5ININDOBv1n48n5lBTBuVrRmQgUq1Sxi1swiAiKGY+6ebVEE++VxL62nN5Zo412eF7bKQSPYglS1sC2IW2JbaHdlno85ma4jZtbGrlYlrl7NPefTGPbzSVjepgC5hABIxOHtKkU5F+/g3k91ddC+V9PO41nGeTd3k95l5diod7zo2IETtaQC3fwxz+gpOeogEQeswcuJ95r59j4rXEZBoIBc5zEphxj7DSQ/tmZS2GDz9LidHX6HHib5lImQ1qNmogW56DOehx5r6Eo+/SjIOqcPb5AYppWY1SqaiwpQlHeqLJY+xkRdScXDi7ecUkasaYz5BsRoFJlX8FeM3P4tsLgWN10Y+WB0n1jqfUqwkXXNZqwlKKDMBtQN4K9nuGhvT7wQflH3K//sny9Mf7F1nC5BsASRC2ShzQb1OuXbwv8oZjL4CDIWH1utv8xL28Gs4PcfZXJ5m+acP0aa0W1UVMTuouIP9QdtphINV+Y6lzIWBN5iGA6+0TJJE64uIOr/7l8hgK6A/FmE+I3Xl/iCMbcFAX9DdibAZ0+p2LCZTYsxHJ99m50Gu/wBfK3bpmf5pOwAAAABJRU5ErkJggg==" alt="[WARN]" class="image_without_zoom"/>';
            elseif (type == 2)
                icon = '<img src="data:image/png;base64,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" alt="[FAIL]" class="image_without_zoom"/>';
            elseif (type == 3)
                icon = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAgCAYAAACYTcH3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFOAAABTgBQSDC0wAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAARYSURBVFiFxZc9TyNHGMf/M7vrMbDYvHhty0CuQECAgheJgi5CVGmRiJQ6SYUs5J5vYKWw3HD5BtEVqaA72ktpjIIuws05GGy8xmAb7xu7Ke5247V3ja0kd39pJT/PMzPPzzP7zM4QdOj4+HhU1/WfAOwBWAYwjf9eMoBLy7Le6Lr+OpVKte0AsX9ks9ltAL8CmP0fAPxUNE1zP5lMvnNgPoG8BRD8jCC2FADfHBwc/E4+Lc17fN4Z6dYHTdO+ppqm/fiFQQDgK0EQfqCU0r0vDGJrj7csa9UrwnEceJ53bF3XYZomCCEIBAKO//n5GYZhePbplKqqfUkIIas8gEmv4OLiInZ2dhz75OQEhUIBsVgM+/v7jv/s7Ay5XA6CIGBtbQ3b29ueydLpNEZGRvrxTFN0lHenTNN02aurqygUCrAsy+XXNA3FYhGGYUAURd9MhBDIstwPhtB+0U69evUKwWAQ19fXPTFd11EsFhEOh337S5KEcrmMWq3m22ZgGADY3NxEq9XyjOm6jomJCcfunsFYLAYAqFQqvkBDw/iJMYZQKOTYtVoNmqY5djQadX5Xq1XU6/V/BzM/P+/6952annZ/xmRZxuPjoyeMH9BQMIQQbGxseMampqZc9u3tLR4eHlwwhPxTK5ZlQZZlV5uhYABgfX3d0989Y5VKxVU9giBgcrJ3F6nVag7Q0DB+5dudqFqt4v7+3uXrXipb9XodjUZjcBi/KrIViURc9t3dXc87YVeUH9DAMBcXF33jkiT1wHTPTDdwt7w/JB7K5/PY2toCpb38hJCeRJlMpqddJBKBaZqeYwBDvDOtVgtXV1eesVAoBMbYi2PEYjE8PT31bIhDwwDA+fm5p7+7rP0kSRIopb5AQ8HkcjlPv1fJHh4e4ujoyJ2MUkSjUZimiXa73QM0FEypVEKlUhkI5ubmxrNtPB4H8PFUoCiKC2jofSafz/f4updJVVUoioJms+kLYwOpquoAkWw2a8LjTEMIQbFYhKZpIISg0WjAMAyMjo4iFAqBEAJKKRRFgWEYkGUZhBDnkWUZlFIkEglwHAdCCDiOg6IoUBQFlFLnZMjzPBhjFg+gBo/LmmVZkCQJl5eXUFUVHMeBUop2uw1VVUEpBaXUScLzPEqlEoCPx0+O4wAA5XLZ6Wv77VhnLsMwHigh5A+/JWGMYWVlBcHgy9cpURQxNzfnu4cMoCtqWdabfi0CgQCWl5cH2kfGxsYwOzvr+joPKkEQfqOapv0CoPgS0NLSkutW4CdRFDEzMzMsUDkej/9MU6lU2zTNfQDtfq0DgQAWFhYGmiFRFJFIJAZaMkKIxhj7PpVKtSkAJJPJd6Zp7gD40K8jYwwLCwsDzdD4+DgSiUTfGaKUVhhj36bT6bcA4LzWp6enf+3u7h5zHFchhIgAxgCMdg/AcRzC4TAajYZzqbOrqvOhlIIxhmAwiGaz6fg4jmvwPP+nIAivKaXfZTKZ9/bYfwN/BqLKrNdTsQAAAABJRU5ErkJggg==" alt="[NA]" class="image_without_zoom"/>';
            end
        end
        
        
        function y = base64encode(x, alg, isChunked, url_safe)
            %BASE64ENCODE Perform base64 encoding on a string.
            % INPUT:
            %   x    - block of data to be encoded.  Can be a string or a numeric
            %          vector containing integers in the range 0-255.
            %   alg  - Algorithm to use: can take values 'java' or 'matlab'. Optional
            %          variable defaulting to 'java' which is a little faster. If
            %          'java' is chosen than core of the code is performed by a call to
            %          a java library. Optionally all operations can be performed using
            %          matleb code.
            %   isChunked - encode output into 76 character blocks. The returned
            %          encoded string is broken into lines of no more than
            %          76 characters each, and each line will end with EOL. Notice that
            %          if resulting string is saved as part of an xml file, those EOL's
            %          are often stripped by xmlwrite funtrion prior to saving.
            %   url_safe - use Modified Base64 for URL applications ('base64url'
            %   encoding) "Base64 alphabet" ([A-Za-z0-9-_=]).
            %
            %
            % OUTPUT:
            %   y    - character array using only "Base64 alphabet" characters
            %
            %   This function may be used to encode strings into the Base64 encoding
            %   specified in RFC 2045 - MIME (Multipurpose Internet Mail Extensions).
            %   The Base64 encoding is designed to represent arbitrary sequences of
            %   octets in a form that need not be humanly readable.  A 65-character
            %   subset ([A-Za-z0-9+/=]) of US-ASCII is used, enabling 6 bits to be
            %   represented per printable character.
            %
            %   See also BASE64DECODE.
            %
            %   Written by Jarek Tuszynski, SAIC, jaroslaw.w.tuszynski_at_saic.com
            %
            %   Matlab version based on 2004 code by Peter J. Acklam
            %   E-mail:      pjacklam@online.no
            %   URL:         http://home.online.no/~pjacklam
            %   http://home.online.no/~pjacklam/matlab/software/util/datautil/base64encode.m
            
            if nargin<2, alg='java';      end
            if nargin<3, isChunked=false; end
            if ~islogical(isChunked)
                if isnumeric(isChunked)
                    isChunked=(isChunked>0);
                else
                    isChunked=false;
                end
            end
            if nargin<4, url_safe=false; end
            if ~islogical(url_safe)
                if isnumeric(url_safe)
                    url_safe=(url_safe>0);
                else
                    url_safe=false;
                end
            end
            
            
            %% if x happen to be a filename than read the file
            if (numel(x)<256)
                if (exist(x, 'file')==2)
                    fid = fopen(x,'rb');
                    x = fread(fid, 'uint8');             % read image file as a raw binary
                    fclose(fid);
                end
            end
            
            %% Perform conversion
            switch (alg)
                case 'java'
                    base64 = org.apache.commons.codec.binary.Base64;
                    y = base64.encodeBase64(x, isChunked);
                    if url_safe
                        y = strrep(y,'=','-');
                        y = strrep(y,'/','_');
                    end
                    
                case 'matlab'
                    
                    %% add padding if necessary, to make the length of x a multiple of 3
                    x   = uint8(x(:));
                    ndbytes = length(x);                 % number of decoded bytes
                    nchunks = ceil(ndbytes / 3);         % number of chunks/groups
                    if rem(ndbytes, 3)>0
                        x(end+1 : 3*nchunks) = 0;          % add padding
                    end
                    x = reshape(x, [3, nchunks]);        % reshape the data
                    y = repmat(uint8(0), 4, nchunks);    % for the encoded data
                    
                    %% Split up every 3 bytes into 4 pieces
                    %    aaaaaabb bbbbcccc ccdddddd
                    % to form
                    %    00aaaaaa 00bbbbbb 00cccccc 00dddddd
                    y(1,:) = bitshift(x(1,:), -2);                  % 6 highest bits of x(1,:)
                    y(2,:) = bitshift(bitand(x(1,:), 3), 4);        % 2 lowest  bits of x(1,:)
                    y(2,:) = bitor(y(2,:), bitshift(x(2,:), -4));   % 4 highest bits of x(2,:)
                    y(3,:) = bitshift(bitand(x(2,:), 15), 2);       % 4 lowest  bits of x(2,:)
                    y(3,:) = bitor(y(3,:), bitshift(x(3,:), -6));   % 2 highest bits of x(3,:)
                    y(4,:) = bitand(x(3,:), 63);                    % 6 lowest  bits of x(3,:)
                    
                    %% Perform the mapping
                    %   0  - 25  ->  A-Z
                    %   26 - 51  ->  a-z
                    %   52 - 61  ->  0-9
                    %   62       ->  +
                    %   63       ->  /
                    map = ['A':'Z', 'a':'z', '0':'9', '+/'];
                    if (url_safe), map(63:64)='-_'; end
                    y = map(y(:)+1);
                    
                    %% Add padding if necessary.
                    npbytes = 3 * nchunks - ndbytes;    % number of padding bytes
                    if npbytes>0
                        y(end-npbytes+1 : end) = '=';     % '=' is used for padding
                    end
                    
                    %% break into lines with length LineLength
                    if (isChunked)
                        eol = sprintf('\n');
                        nebytes = numel(y);
                        nlines  = ceil(nebytes / 76);     % number of lines
                        neolbytes = length(eol);          % number of bytes in eol string
                        
                        % pad data so it becomes a multiple of 76 elements
                        y(nebytes + 1 : 76 * nlines) = 0;
                        y = reshape(y, 76, nlines);
                        
                        % insert eol strings
                        y(end + 1 : end + neolbytes, :) = eol(:, ones(1, nlines));
                        
                        % remove padding, but keep the last eol string
                        m = nebytes + neolbytes * (nlines - 1);
                        n = (76+neolbytes)*nlines - neolbytes;
                        y(m+1 : n) = [];
                    end
            end
            
            %% reshape to a row vector and make it a character array
            y = char(reshape(y, 1, numel(y)));
        end
        
        function [index] = binarySearch(data, num)
            % finds the greatest value lower than num
            left = 1;
            right = length(data);
            
            index = -1;
            
            while left <= right
                mid = ceil((left + right) / 2);
                
                if data(mid) == num
                    index = mid;
                    
                    break;
                else if data(mid) > num
                        right = mid - 1;
                    else
                        left = mid + 1;
                    end
                end
            end
            
            if (index == -1)
                index = right;
            end
            
        end
        
        
        
    end
    
end

