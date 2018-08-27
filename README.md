# Vaccine Efficacy

The  project aims to predic the efficacy of the malaria vaccine administered to patients.

###Requirements
- Python >=2.7.15
- Numpy
- Pandas
- sklearn.linear_model

#### Datset used

The data that was used for the project was taken from National Center for Biotechnology Information. The dataset consisted of the expression data from a malaria vaccine trial.

Volunteers were assessed on the day of the vaccination and 24, 72 hours, two weeks after vaccination, and 5 days after challenge. The data set only consisted of 39 volunteers, but we took the number of datapoints to 60f for every time data was collected by adding a very small amount of noise to existing data points and treating them as new datapoints.

The datafiles are stored in the datafiles directory for every time the data was collected.

The link for the original dataset is given below.

[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18323](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18323)

 