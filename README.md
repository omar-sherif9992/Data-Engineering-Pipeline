
<h1 align="center">Welcome to Data Engineering Pipeline</h1>

<div align="center">
    <img src="https://github.com/omar-sherif9992/Data-Engineering-Projects/assets/69806823/5ba521c0-0490-4b5a-a1c1-887d9a1c4a0c" alt="Logo" width="80" height="80">
<br/>


  <h3 align="center">clean, impute, handle outliers, feature engineer, visualize, analyze, containerize, parallelize workload, and build a pipeline </h3>

  <p align="center">
The 4 Milestones aim to build a Data Engineering Pipeline
    <br />
    <br />
	 <a href="https://github.com/omar-sherif9992/Data-Engineering-Projects/tree/main/M4/DE_M4_49-3324_MET_10_2016/dashboard_ss" download target="_blank"><strong>View the Screenshots »</strong></a>
    <br />
   ·	  
   <a href="https://drive.google.com/file/d/1t4xE80t6eL69LLNAr0udsOg1rIoqmAvL/view?usp=drive_link">Demo Video</a>
   ·
   <a href="mailto:osa.helpme@gmail.com?subject=UnExpected%20Error%20Occured&body=Sorry%20for%20the%20inconvenience%2C%20Please%20describe%20Your%20situation%20and%20emphasis%20the%20Endpoint%20!%0A">Report Bug</a>
   	      ·
    <a href="mailto:osa.helpme@gmail.com?subject=I%20want%20to%20be%20a%20Contributor%20to%20Bachelor Thesis&body=Dear%20Omar%20Sherif">Be a Contributer</a>
  </p>
</div>

## 💡 Description

### Milestone 1

The goal of this milestone is to load a csv file, perform exploratory data analysis
with visualization, extract additional data, perform feature engineering and pre-
process the data for downstream cases such as ML and data analysis.
The dataset you will be working on is NYC green taxis dataset. It contains records
about trips conducted in NYC through green taxis. 

There are multiple datasets for this case study(a dataset for each month). 
Download [dataset](https://drive.google.com/drive/folders/1t8nBgbHVaA5roZY4z3RcAG1_JMYlSTqu) from here.

**My dataset was 10/2016**, the code is reproducible and can work with any month/year

### Milestone 2
The objective of this milestone is to package your milestone 1 code in a docker
image that can be run anywhere. In addition, you will load your cleaned and
prepared dataset as well as your lookup table into a PostgreSQL database which
would act as your data warehouse.

### Milestone 3
The goal of this milestone is to preprocess the dataset 'New York yellow taxis' by performing
basic data preparation and basic analysis to gain a better understanding of the data using
PySpark.
Use the same month and year you used for the green taxis in milestone 1. [Datasets](https://drive.google.com/drive/folders/1t8nBgbHVaA5roZY4z3RcAG1_JMYlSTqu) (download
the yellow taxis dataset).


### Milestone 4

For this milestone, we were required to orchestrate the tasks performed in
milestones 1 and 2 using Airflow in Docker. For this milestone, we will primarily
work on the green dataset and pre-process using pandas only for simplicity.
The tasks you have performed in milestones 1 and 2 were as follows.
Read csv(green_taxis) file >> clean and transform >> load to csv(both the
cleaned dataset and the lookup table) >> extract additional resources(GPS
coordinates) >> Integrate with the cleaned dataset and load back to csv >> load
both csv files(lookup and cleaned dataset) to postgres database as 2 separate
tables.
