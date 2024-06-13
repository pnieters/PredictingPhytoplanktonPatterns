% This code contains the polynomial fit of the driver functions of
%   temperature
%   salinity
%   light attenuation (Kd)
% orderd by the six blomming pattern
%   (Code to paper: A machine learning based bottom-up approach to derive
%   environmental factors from phytoplankton blooms in the Baltic Sea
%   (Berthold, Nieters, Vortmeyer-Kley, 2024)
%    The code is licensed under an MIT-License (c) 2023, Berthold, Nieters, Vortmeyer-Kley)
%
%  Version June 2024 (rahel.vortmeyer-kley(at)uni-olenburg.de)
%%
%% fitting temperature, salinity, Kd
close all
clear all
clc

% load measurment data for temperature, salinity and Kd of the blomming pattern
Statname = {'OuTrBldriver.h5', 'SuBldriver.h5', 'InTrBldriver.h5', 'InDuBldriver.h5', 'DeSpBldriver.h5', 'AdBlEWdriver.h5'};

for S = 1:6 % loop blooming pattern

    V = h5read(Statname{S},'/Statallsort'); % values: year Light Kd Sal Temp DoY Dayfromfirst
    VS = h5read(Statname{S},'/StartV');     % values for first and last day of teh year from linear fit

    Time = [1 ; V(:,6); 365];                       % tiem as day of the year
    Temp = [VS(4,1)+VS(4,2); V(:,5); VS(4,2)];      % temperature (temp)
    Kd = [VS(2,1)+VS(2,2); V(:,3); VS(2,2)];        % light attenuation (Kd)
    Sal = [VS(3,1)+VS(3,2); V(:,4); VS(3,2)];       % salinity (sal)

    % put to 365d frame and mean and remove doubel entries 
    Time1 = round(Time);
    Time1(:,2) = Temp;
    Time1(:,3) = Kd;
    Time1(:,4) = Sal;

    for k=1:365
        [r, ~] = find(Time1(:,1)==k);
        if r>1
           Time1(r,2) = mean(Time1(r,2)); % temp
           Time1(r,3) = mean(Time1(r,3)); % Kd
           Time1(r,4) = mean(Time1(r,4)); % sal
        end
    end
    Time1 = unique(Time1,'rows');

    TT = nan(365,4);
    TT(:,1) = 1:365;
    TT(Time1(:,1),2) = Time1(:,2); % Temp   => fit with 5th order
    TT(Time1(:,1),3) = Time1(:,3); % Kd     => fit with 4th order
    TT(Time1(:,1),4) = Time1(:,4); % sal    => fit with 4th order

    %-------------------------curve fitting on full data TEMP
    [fT1, g1] = fit(TT(:,1), TT(:,2), 'poly3', 'Exclude', isnan(TT(:,2)));
    [fT2, g2] = fit(TT(:,1), TT(:,2), 'poly4', 'Exclude', isnan(TT(:,2)));
    [fT3, g3] = fit(TT(:,1), TT(:,2), 'poly5', 'Exclude', isnan(TT(:,2)));
% 
%     % goodness of fit overview:
%     gof = struct2table([g1 g2 g3], 'RowNames',["f3" "f4" "f5"])%;

    % write coefficients of polynome to matrix 
    p =[fT3.p1 fT3.p2 fT3.p3 fT3.p4 fT3.p5 fT3.p6];
    PT(S,:) = p;
    clear p
    
    %-------------------curve fitting on full data KD
    [fKd1, g1] = fit(TT(:,1), TT(:,3), 'poly2', 'Exclude', isnan(TT(:,2)));
    [fKd2, g2] = fit(TT(:,1), TT(:,3), 'poly3', 'Exclude', isnan(TT(:,2)));
    [fKd3, g3] = fit(TT(:,1), TT(:,3), 'poly4', 'Exclude', isnan(TT(:,2)));
% 
%     % goodness of fit overview:
%     gof = struct2table([g1 g2 g3], 'RowNames',["f2" "f3" "f4"])%;

    % write coefficients of polynome to matrix 
    p =[fKd3.p1 fKd3.p2 fKd3.p3 fKd3.p4 fKd3.p5];
    PKd(S,:) = p;
    clear p
%     
%     %-------------------curve fitting on full data Salinity
    [fS1, g1] = fit(TT(:,1), TT(:,4), 'poly2', 'Exclude', isnan(TT(:,2)));
    [fS2, g2] = fit(TT(:,1), TT(:,4), 'poly3', 'Exclude', isnan(TT(:,2)));
    [fS3, g3] = fit(TT(:,1), TT(:,4), 'poly4', 'Exclude', isnan(TT(:,2)));
% 
%     % goodness of fit overview:
%     gof = struct2table([g1 g2 g3], 'RowNames',["f2" "f3" "f4"])%;
    
    % write coefficients of polynome to matrix 
    p =[fS3.p1 fS3.p2 fS3.p3 fS3.p4 fS3.p5];
    PS(S,:) = p;
    clear p
    
    clearvars -except Statname PS PKd PT
end

% SAVE DATA TO H5
% order of ploynoms within matrixes rows: {'OuTrBldriver.h5', 'SuBldriver.h5', 'InTrBldriver.h5', 'InDuBldriver.h5', 'DeSpBldriver.h5', 'AdBlEWdriver.h5'};
    % poly 4 for Sal
    h5create('DriverFkt.h5','/PS',[size(PS,1) size(PS,2)])
    h5write('DriverFkt.h5','/PS',PS)
    % poly 5 for Temp
    h5create('DriverFkt.h5','/PT',[size(PT,1) size(PT,2)])
    h5write('DriverFkt.h5','/PT',PT)
    % poly 4 for Kd
    h5create('DriverFkt.h5','/PKd',[size(PKd,1) size(PKd,2)])
    h5write('DriverFkt.h5','/PKd',PKd)