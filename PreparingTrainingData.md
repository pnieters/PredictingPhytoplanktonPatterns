## preparing CWB_df.h5 (training data) and CWB_df_Train_Test.h5 (validation datat) data for training and validation:

1. read /zscore and /ZscoreTest

2. build a matrix from the data with 
	1. column year
	1. column Chl
	1. column WTemp (Temp)
	1. column DIN
	1. column orthoP (DIP)
	1. column totalN
	1. column totalP
	1. column SAL (only for CWB_df.h5, for  CWB_df_Train_Test.h5 produce a dummy column with ones)
	1. column Day of the year (build from date information in Day and Month)
	1. column Day from first (build from date information)

3. remove rows with any NaN entry for column 2. to column 8.

4. cut out the part of the matrix with the blooming type you are interested in (information about bloomingtypes in variable BType) and keep it

5. sort the rows of the matrix produced in step 4 according to the day of the year (9. column)

6. if column 9 contains double entries, shift one of the entries by 1e-3 thus column 9. is an unique column

7. save the matrix as .h5 file under the same name as the blooming type


## Startvalues
additionally only for the training data produce a linear regression for the start and end values

8. take the first 10 rows and the last 10 rows

9. shift the time index (column 9.) by -365 days

10. make a linear regression from these 20 datapoints around time=0 for Chl, Temp, DIN, DIP, totalN and totalP and use the results of the regression for time=1 and time=0 (which is 365 on the original timescale) as first and last value of the timeseries

11. save the regression coefficents as .h5
